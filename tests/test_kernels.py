# Metal kernel tests (Apple GPU only): each kernel must compute exactly what the
# plain-PyTorch path in model.py computes, and the whole engine must produce the
# same tokens with kernels on and off. All in fp32, so differences are real bugs.

import dataclasses

import pytest
import torch

from atomserve import kernels
from atomserve.config import EngineConfig
from atomserve.engine import Engine
from atomserve.kv_cache import KVPool, hadamard, quantize_heads
from atomserve.model import (
    apply_rope,
    decode_attention,
    rms_norm,
    rope_cos_sin,
    rope_inv_freq,
    rotate,
)
from atomserve.runner import ModelRunner
from atomserve.scheduler import Scheduler
from atomserve.sequence import SamplingParams
from tests.conftest import TINY_CONFIG, random_weights

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Apple GPU")
MPS = torch.device("mps")


@pytest.mark.parametrize("heads", [(16, 8), (32, 8)])
def test_qk_norm_rope_store_matches_pytorch(heads):
    H, KV = heads
    torch.manual_seed(0)
    positions = torch.tensor([0, 1, 300, 4097, 39999], device=MPS)
    slots = torch.tensor([3, 9, 17, 2, 30], device=MPS)
    qkv = torch.randn(5, H + 2 * KV, 128, device=MPS) * 3
    q_norm, k_norm = torch.randn(128, device=MPS) + 1, torch.randn(128, device=MPS) + 1
    k_pool, v_pool = torch.zeros(64, KV, 128, device=MPS), torch.zeros(64, KV, 128, device=MPS)

    q = kernels.qk_norm_rope_store(
        qkv, q_norm, k_norm, positions, slots, rope_inv_freq(128, 1e6, MPS),
        k_pool, v_pool, H, KV, 1e-6,
    )  # fmt: skip

    cos, sin = rope_cos_sin(positions, 128, 1e6, torch.float32)
    rq, rk, rv = qkv.split([H, KV, KV], dim=1)
    assert torch.allclose(q, apply_rope(rms_norm(rq, q_norm, 1e-6), cos, sin), atol=1e-5)
    assert torch.allclose(
        k_pool[slots], apply_rope(rms_norm(rk, k_norm, 1e-6), cos, sin), atol=1e-5
    )
    assert torch.equal(v_pool[slots], rv)
    untouched = torch.ones(64, dtype=torch.bool)
    untouched[slots.cpu()] = False
    assert k_pool[untouched.to(MPS)].abs().sum() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qk_norm_rope_store_quantizes_like_pytorch(dtype):
    """With an int8 pool, q and k come out rotated and the pool gets the integers and
    scales of kv_cache.quantize_heads (off by one step at most where the kernel's fp32
    k sits on a rounding boundary)."""
    H, KV = 16, 8
    torch.manual_seed(0)
    positions = torch.tensor([0, 1, 300, 4097, 39999], device=MPS)
    slots = torch.tensor([3, 9, 17, 2, 30], device=MPS)
    qkv = (torch.randn(5, H + 2 * KV, 128, device=MPS) * 3).to(dtype)
    qkv[1, H + KV + 2] = 0  # an all-zero value head
    q_norm, k_norm = (
        (torch.randn(128, device=MPS) + 1).to(dtype),
        (torch.randn(128, device=MPS) + 1).to(dtype),
    )
    k_pool = torch.zeros(64, KV, 128, device=MPS, dtype=torch.int8)
    v_pool = torch.zeros_like(k_pool)
    k_scale, v_scale = torch.zeros(64, KV, device=MPS), torch.zeros(64, KV, device=MPS)

    q = kernels.qk_norm_rope_store(
        qkv, q_norm, k_norm, positions, slots, rope_inv_freq(128, 1e6, MPS),
        k_pool, v_pool, H, KV, 1e-6, k_scale, v_scale,
    )  # fmt: skip

    cos, sin = rope_cos_sin(positions, 128, 1e6, torch.float32)
    rq, rk, rv = qkv.float().split([H, KV, KV], dim=1)
    h = hadamard(128, MPS)  # an int8 pool rotates q and k (kv_cache.py)
    ref_q = rotate(apply_rope(rms_norm(rq, q_norm.float(), 1e-6), cos, sin), h)
    assert torch.allclose(q.float(), ref_q, atol=1e-5 if dtype == torch.float32 else 0.05)
    ref_k = rotate(apply_rope(rms_norm(rk, k_norm.float(), 1e-6), cos, sin), h)
    for pool, scale, ref in ((k_pool, k_scale, ref_k), (v_pool, v_scale, rv)):
        ref_q8, ref_scale = quantize_heads(ref)
        assert (pool[slots].int() - ref_q8.int()).abs().max() <= 1
        assert torch.allclose(scale[slots], ref_scale, rtol=1e-5)
    assert v_scale[slots[1], 2] == 0 and v_pool[slots[1], 2].abs().sum() == 0
    untouched = torch.ones(64, dtype=torch.bool)
    untouched[slots.cpu()] = False
    assert k_pool[untouched.to(MPS)].abs().sum() == 0 and k_scale[untouched.to(MPS)].sum() == 0


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("heads", [(16, 8), (32, 8), (8, 8)])
def test_paged_decode_attention_matches_pytorch(heads, quantized):
    """Contexts of 1 key, a partial block, exactly one partition (512) and several
    partitions, on scattered blocks."""
    H, KV = heads
    torch.manual_seed(0)
    bs, lens = 16, [1, 37, 512, 513, 1500]
    k_pool = torch.randn(256 * bs, KV, 128, device=MPS) * 4
    v_pool = torch.randn_like(k_pool)
    scales = {}
    if quantized:  # the kernel reads int8 + scales; the reference their dequantized values
        (k8, k_scale), (v8, v_scale) = quantize_heads(k_pool), quantize_heads(v_pool)
        k_pool, v_pool = k8.float() * k_scale[..., None], v8.float() * v_scale[..., None]
        scales = {"k_scale": k_scale, "v_scale": v_scale}
    tables = [torch.randperm(256)[: -(-n // bs)].tolist() for n in lens]
    width = max(map(len, tables))
    padded = torch.tensor([t + [0] * (width - len(t)) for t in tables], device=MPS)
    q = torch.randn(len(lens), H, 128, device=MPS)

    out = kernels.paged_decode_attention(
        q, k8 if quantized else k_pool, v8 if quantized else v_pool, padded,
        torch.tensor(lens, device=MPS), bs, KV, max(lens), **scales,
    )  # fmt: skip
    for i, (n, table) in enumerate(zip(lens, tables, strict=True)):
        slots = torch.tensor([table[p // bs] * bs + p % bs for p in range(n)], device=MPS)
        ref = decode_attention(q[i : i + 1], k_pool[slots][None], v_pool[slots][None], None)
        assert torch.allclose(out[i], ref[0], atol=1e-5), f"context of {n} keys"


# Big enough heads for the kernels (head_dim 128), small enough to be instant.
KERNEL_CONFIG = dataclasses.replace(TINY_CONFIG, head_dim=128, hidden_size=64)


def engine(weights, use_kernels: bool, quantized_kv: bool = False) -> Engine:
    config = EngineConfig(
        block_size=4, num_blocks=64, max_model_len=64, max_num_batched_tokens=8, max_num_seqs=4
    )
    pool = KVPool(
        KERNEL_CONFIG, config.num_blocks, config.block_size, MPS, torch.float32, quantized_kv
    )
    runner = ModelRunner(weights, KERNEL_CONFIG, pool, use_kernels=use_kernels)
    return Engine(runner, Scheduler(config, frozenset()))


@pytest.mark.parametrize("quantized_kv", [False, True])
def test_engine_outputs_match_with_kernels_on_and_off(quantized_kv):
    """Prompt chunks, decodes of unequal lengths and prefix hits, through both paths
    (with an int8 cache too: the kernels quantize and dequantize like KVPool does)."""
    weights = random_weights(KERNEL_CONFIG, device="mps")
    assert kernels.available(MPS, torch.float32, KERNEL_CONFIG)
    prompts = [list(range(1, 20)), [5, 6, 7], list(range(30, 41)), list(range(1, 20))]
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    with_kernels = engine(weights, True, quantized_kv).generate(prompts, params)
    assert with_kernels == engine(weights, False, quantized_kv).generate(prompts, params)


def test_topk_rows_matches_torch_topk():
    """Exact values (and tokens with those values) for every k, including ties and
    rows with fewer finite entries than k."""
    torch.manual_seed(0)
    rows = [
        torch.randn(7, 151936, device=MPS) * 4,
        torch.randint(-5, 5, (3, 151936), device=MPS).float(),  # heavy ties
        torch.full((2, 151936), float("-inf"), device=MPS).index_fill_(
            1, torch.tensor([3, 9], device=MPS), 1.0
        ),
    ]
    for x in rows:
        for k in (1, 2, 20, 33, 64):
            values, indices = kernels.topk_rows(x, k)
            expected = x.topk(k, dim=-1).values
            assert torch.equal(values, expected)
            assert torch.equal(x.gather(1, indices), expected)


def test_sampler_is_the_same_with_and_without_the_topk_kernel():
    from atomserve.sampler import Sampler
    from atomserve.sequence import Sequence

    logits = torch.randn(6, 151936, device=MPS) * 3
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.7, top_k=20, top_p=0.8, seed=1),
        SamplingParams(temperature=1.0, top_k=64, top_p=1.0, seed=2),
        SamplingParams(temperature=0.6, top_k=5, top_p=0.95, seed=3),
        SamplingParams(temperature=1.3, top_k=1, seed=4),
        SamplingParams(temperature=0.0, presence_penalty=2.0),
    ]
    seqs = [Sequence(i, [0, 7, 7, 9], p, num_prompt_tokens=1) for i, p in enumerate(params)]
    with_kernel = Sampler(MPS, use_kernels=True).sample(logits, seqs)
    assert with_kernel == Sampler(MPS, use_kernels=False).sample(logits, seqs)


def test_silu_mul_matches_pytorch():
    for rows in (1, 7, 300):
        gate_up = torch.randn(rows, 2 * 3072, device=MPS) * 3
        gate, up = gate_up.chunk(2, dim=-1)
        assert torch.allclose(
            kernels.silu_mul(gate_up), torch.nn.functional.silu(gate) * up, atol=1e-5
        )


@pytest.mark.parametrize("rows", [0, 1, 3, 5, 8])
def test_int8_linear_matches_dequantized_matmul(rows):
    from atomserve.quant import quantize_int8

    w = quantize_int8(torch.randn(1000, 512, device=MPS) * 0.05)  # N not a multiple of 32
    x = torch.randn(rows, 512, device=MPS)
    residual = torch.randn(rows, 1000, device=MPS)
    expected = residual + x @ w.dequantize().T
    assert torch.allclose(kernels.int8_linear(x, w.q, w.scale, residual), expected, atol=1e-4)
    assert torch.allclose(kernels.int8_linear(x, w.q, w.scale), expected - residual, atol=1e-4)


@pytest.mark.parametrize("rows", [12, 32, 100])
def test_quantized_linear_paths_agree(rows):
    """Chunked kernel (12, 32 rows) and dequantize + matmul (100 rows) give the same result."""
    from atomserve.quant import linear, quantize_int8

    w = quantize_int8(torch.randn(700, 512, device=MPS) * 0.05)
    x, residual = torch.randn(rows, 512, device=MPS), torch.randn(rows, 700, device=MPS)
    expected = residual + x @ w.dequantize().T
    assert torch.allclose(linear(x, w, residual, use_kernels=True), expected, atol=1e-4)
    deq = kernels.dequantize_int8(w.q, w.scale, torch.float32)
    assert torch.equal(deq, w.dequantize())


def test_quantized_engine_outputs_match_with_kernels_on_and_off():
    """int8 weights: the Metal kernels (int8 matmul, chunked) agree with dequantize + matmul."""
    from atomserve.loader import quantize_weights

    weights = random_weights(KERNEL_CONFIG, device="mps")
    quantize_weights(weights)
    prompts = [list(range(1, 20)), [5, 6, 7], list(range(30, 41)), list(range(1, 20))]
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    with_kernels = engine(weights, use_kernels=True).generate(prompts, params)
    assert with_kernels == engine(weights, use_kernels=False).generate(prompts, params)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_dequantize_kv_matches_pytorch(dtype):
    torch.manual_seed(0)
    values = torch.randint(-127, 128, (64, 8, 128), device=MPS, dtype=torch.int8)
    scale = torch.rand(64, 8, device=MPS)
    slots = torch.tensor([5, 63, 0, 17, 17], device=MPS)
    expected = (values[slots].float() * scale[slots][..., None]).to(dtype)
    assert torch.equal(kernels.gather_dequantize_kv(values, scale, slots, dtype), expected)
    assert kernels.gather_dequantize_kv(values, scale, slots[:0], dtype).shape == (0, 8, 128)
