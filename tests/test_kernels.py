# Metal kernel tests (Apple GPU only): each kernel must compute exactly what the
# plain-PyTorch path in model.py computes, and the whole engine must produce the
# same tokens with kernels on and off. All in fp32, so differences are real bugs.

import dataclasses

import pytest
import torch

from atomserve import kernels
from atomserve.config import EngineConfig
from atomserve.engine import Engine
from atomserve.kv_cache import KVPool
from atomserve.model import apply_rope, decode_attention, rms_norm, rope_cos_sin, rope_inv_freq
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


@pytest.mark.parametrize("heads", [(16, 8), (32, 8), (8, 8)])
def test_paged_decode_attention_matches_pytorch(heads):
    """Contexts of 1 key, a partial block, exactly one partition (512) and several
    partitions, on scattered blocks."""
    H, KV = heads
    torch.manual_seed(0)
    bs, lens = 16, [1, 37, 512, 513, 1500]
    k_pool = torch.randn(256 * bs, KV, 128, device=MPS)
    v_pool = torch.randn_like(k_pool)
    tables = [torch.randperm(256)[: -(-n // bs)].tolist() for n in lens]
    width = max(map(len, tables))
    padded = torch.tensor([t + [0] * (width - len(t)) for t in tables], device=MPS)
    q = torch.randn(len(lens), H, 128, device=MPS)

    out = kernels.paged_decode_attention(
        q, k_pool, v_pool, padded, torch.tensor(lens, device=MPS), bs, KV, max(lens)
    )
    for i, (n, table) in enumerate(zip(lens, tables, strict=True)):
        slots = torch.tensor([table[p // bs] * bs + p % bs for p in range(n)], device=MPS)
        ref = decode_attention(q[i : i + 1], k_pool[slots][None], v_pool[slots][None], None)
        assert torch.allclose(out[i], ref[0], atol=1e-5), f"context of {n} keys"


# Big enough heads for the kernels (head_dim 128), small enough to be instant.
KERNEL_CONFIG = dataclasses.replace(TINY_CONFIG, head_dim=128, hidden_size=64)


def engine(weights, use_kernels: bool) -> Engine:
    config = EngineConfig(
        block_size=4, num_blocks=64, max_model_len=64, max_num_batched_tokens=8, max_num_seqs=4
    )
    pool = KVPool(KERNEL_CONFIG, config.num_blocks, config.block_size, MPS, torch.float32)
    runner = ModelRunner(weights, KERNEL_CONFIG, pool, use_kernels=use_kernels)
    return Engine(runner, Scheduler(config, frozenset()))


def test_engine_outputs_match_with_kernels_on_and_off():
    """Prompt chunks, decodes of unequal lengths and prefix hits, through both paths."""
    weights = random_weights(KERNEL_CONFIG, device="mps")
    assert kernels.available(MPS, torch.float32, KERNEL_CONFIG)
    prompts = [list(range(1, 20)), [5, 6, 7], list(range(30, 41)), list(range(1, 20))]
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    with_kernels = engine(weights, use_kernels=True).generate(prompts, params)
    assert with_kernels == engine(weights, use_kernels=False).generate(prompts, params)
