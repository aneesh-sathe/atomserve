# KV pool tests. The fast ones check storage addressing on a toy pool (CPU); the
# slow ones prove, on the real model, that caching changes speed and nothing else,
# and that an int8 cache costs almost no quality.

import pytest
import torch

from atomserve.config import EngineConfig
from atomserve.engine import create_engine
from atomserve.kv_cache import KVPool, hadamard, quantize_heads
from atomserve.runner import ModelRunner, slot_of
from atomserve.sequence import SamplingParams, Sequence
from tests.conftest import TINY_CONFIG

CPU = torch.device("cpu")


def test_slot_of_follows_the_block_table():
    """Position p lives at offset p % block_size inside block block_table[p // block_size]."""
    table = [7, 2, 5]
    assert [slot_of(table, p, 4) for p in (0, 3, 4, 5, 11)] == [28, 31, 8, 9, 23]


def test_write_then_read_round_trips_through_scattered_slots(tiny_config):
    """Whatever slots a sequence's tokens land in, reading them back gives them in order."""
    pool = KVPool(tiny_config, num_blocks=6, block_size=4, device=CPU, dtype=torch.float32)
    k = torch.randn(5, tiny_config.num_kv_heads, tiny_config.head_dim)
    v = torch.randn_like(k)
    slots = torch.tensor([20, 21, 22, 23, 4])  # block 5 then block 1
    pool.write(1, k, v, slots)
    k_out, v_out = pool.read(1, slots)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)


def test_layers_do_not_share_storage(tiny_config):
    """Writing layer 0's slot 3 leaves layer 1's slot 3 untouched."""
    pool = KVPool(tiny_config, num_blocks=2, block_size=4, device=CPU, dtype=torch.float32)
    pool.k.zero_()
    pool.v.zero_()
    ones = torch.ones(1, tiny_config.num_kv_heads, tiny_config.head_dim)
    pool.write(0, ones, ones, torch.tensor([3]))
    assert torch.equal(pool.read(0, torch.tensor([3]))[0], ones)
    assert pool.read(1, torch.tensor([3]))[0].abs().sum() == 0


PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is"


@pytest.mark.slow
def test_prefill_then_decode_logits_equal_full_recompute(weights_fp32, config, full_logits):
    """At every step, the cached forward pass yields the same logits as recomputing
    the whole sequence from scratch (fp32, so any mismatch is a real bug). With
    block_size 4 the 13-token run crosses three block boundaries."""
    device = weights_fp32.embed.device
    pool = KVPool(config, num_blocks=4, block_size=4, device=device, dtype=torch.float32)
    runner = ModelRunner(weights_fp32, config, pool)
    seq = Sequence(0, token_ids=list(PROMPT), block_table=[3, 0, 2, 1])

    num_new = len(PROMPT)
    for _ in range(9):
        logits = runner.run([seq], [num_new]).cpu()
        assert (logits - full_logits(weights_fp32, config, seq.token_ids)[-1:]).abs().max() < 1e-3
        seq.num_computed_tokens += num_new
        seq.token_ids.append(int(logits.argmax()))
        num_new = 1


@pytest.mark.slow
def test_engine_greedy_equals_argmax_of_full_recompute(weights_fp32, config, greedy, full_logits):
    """32 greedy tokens through the cached engine equal recomputing the whole sequence
    from scratch at every step and taking the argmax (the O(N^2) way)."""
    [cached] = greedy(weights_fp32, config, [PROMPT], 32)
    ids = list(PROMPT)
    for _ in range(32):
        ids.append(int(full_logits(weights_fp32, config, ids)[-1].argmax()))
    assert cached == ids[len(PROMPT) :]


def test_pool_starts_finite(tiny_config):
    """Batched decode attention pads short contexts with never-written slots and masks
    them; masking only works if those slots can't hold NaN (0 * NaN = NaN)."""
    pool = KVPool(tiny_config, num_blocks=4, block_size=4, device=CPU, dtype=torch.float32)
    assert torch.isfinite(pool.k).all() and torch.isfinite(pool.v).all()


def test_int8_heads_round_to_within_half_a_step():
    """Each head vector uses its own scale, so tiny and huge vectors are both precise."""
    x = torch.randn(6, 2, 128) * torch.logspace(-3, 2, 6)[:, None, None]
    x[0, 1] = 0.0  # an all-zero head: scale 0, no division by zero
    q, scale = quantize_heads(x)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    assert torch.equal(q[0, 1], torch.zeros(128, dtype=torch.int8)) and scale[0, 1] == 0
    error = (q.float() * scale[..., None] - x).abs().amax(dim=-1)
    assert (error <= scale / 2 * (1 + 1e-5)).all()
    assert (q.abs().amax(dim=-1)[scale > 0] == 127).all()  # the full int8 range is used


def test_hadamard_is_a_rotation():
    h = hadamard(128, CPU)
    assert torch.equal(h, h.T)
    assert torch.allclose(h @ h, torch.eye(128), atol=1e-6)
    h4 = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]  # Sylvester order
    assert torch.equal(hadamard(4, CPU) * 2, torch.tensor(h4, dtype=torch.float32))
    with pytest.raises(ValueError):
        hadamard(96, CPU)


def test_rotating_keys_with_outlier_channels_makes_int8_scores_precise():
    """Keys with a few huge channels (as real keys have): quantized as is, the outliers
    set a coarse step for every other channel; rotated first (queries too), the scores
    q . k come out several times more precise, though the rotation changes no score."""
    torch.manual_seed(0)
    k = torch.randn(256, 1, 128)
    k[..., [3, 70]] *= 40  # two outlier channels
    q = torch.randn(128)
    h = hadamard(128, CPU)

    def score_error(rotation):
        k_r = k @ rotation
        q8, scale = quantize_heads(k_r)
        return ((q8.float() * scale[..., None] - k_r) @ (rotation.T @ q)).abs().mean()

    assert torch.allclose((k @ h) @ (h @ q), k @ q, atol=1e-3)  # rotation changes no score
    assert score_error(h) < score_error(torch.eye(128)) / 4


def test_int8_pool_round_trips_within_rounding(tiny_config):
    pool = KVPool(tiny_config, 6, 4, CPU, torch.float32, quantized=True)
    assert pool.k.dtype == torch.int8 and torch.isfinite(pool.k_scale).all()
    k = torch.randn(5, tiny_config.num_kv_heads, tiny_config.head_dim)
    v = torch.randn_like(k) * 10
    slots = torch.tensor([20, 21, 22, 23, 4])
    pool.write(1, k, v, slots)
    k_out, v_out = pool.read(1, slots)
    assert k_out.dtype == torch.float32
    for x, out in ((k, k_out), (v, v_out)):
        step = x.abs().amax(dim=-1, keepdim=True) / 127
        assert ((out - x).abs() <= step / 2 * (1 + 1e-5)).all()
    assert pool.read(0, slots)[0].abs().sum() == 0  # other layers untouched


def test_int8_pool_takes_132_of_256_bytes_per_head():
    """head_dim 128: 128 int8 values + one fp32 scale instead of 128 bf16 values."""
    import dataclasses

    config = dataclasses.replace(TINY_CONFIG, head_dim=128)
    plain = KVPool.bytes_per_block(config, 16, torch.bfloat16)
    int8 = KVPool.bytes_per_block(config, 16, torch.bfloat16, quantized=True)
    assert int8 / plain == 132 / 256
    pool = KVPool(config, 3, 16, CPU, torch.bfloat16, quantized=True)
    tensors = (pool.k, pool.v, pool.k_scale, pool.v_scale)
    assert sum(t.numel() * t.element_size() for t in tensors) == 3 * int8


INT8_PROMPTS = [[1, 2, 3], list(range(4, 20)), [12], [13, 14, 15, 16, 17], list(range(4, 20))]


def int8_engine(tiny_weights, **kwargs):
    config = EngineConfig(
        **{"block_size": 4, "num_blocks": 64, "max_model_len": 64, "kv_dtype": "int8", **kwargs}
    )
    return create_engine(tiny_weights, TINY_CONFIG, config, frozenset())


def test_int8_cache_mostly_keeps_the_greedy_tokens(tiny_weights):
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    int8 = int8_engine(tiny_weights).generate(INT8_PROMPTS, params)
    plain = int8_engine(tiny_weights, kv_dtype="auto").generate(INT8_PROMPTS, params)
    same = sum(a == b for p, r in zip(plain, int8, strict=True) for a, b in zip(p, r, strict=True))
    assert same >= 0.75 * 12 * len(INT8_PROMPTS)


@pytest.mark.parametrize(
    "setting",
    [
        {"max_num_batched_tokens": 8, "max_num_seqs": 4},  # chunked prefill
        {"num_blocks": 12, "max_model_len": 40},  # preemption and recompute
        {"enable_prefix_caching": False},  # the repeated prompt recomputed, not reused
    ],
)
def test_int8_cache_outputs_do_not_depend_on_scheduling(tiny_weights, setting):
    """Each token's K/V is quantized on its own, so chunking, preemption and prefix
    reuse store the same integers and change nothing."""
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    expected = int8_engine(tiny_weights).generate(INT8_PROMPTS, params)
    assert int8_engine(tiny_weights, **setting).generate(INT8_PROMPTS, params) == expected


TEXT = (
    "The history of the printing press begins in the fifteenth century, when Johannes "
    "Gutenberg combined a screw press with movable metal type, oil-based ink and a mould "
    "for casting letters. Before that, books in Europe were copied by hand, a slow and "
    "costly process that kept them rare. Within a few decades, printing shops had spread "
    "to more than two hundred cities, and millions of volumes were in circulation."
)


@pytest.mark.slow
def test_int8_cache_keeps_real_model_quality(model_dir, config, weights_bf16, full_logits):
    """Next-token distributions on natural text barely move from a bf16 cache (mean KL
    divergence), the top-1 token almost always agrees, and perplexity stays within 1%.
    (Without the key rotation, Qwen3-0.6B's perplexity rises ~5% and KL is ~10x.)"""
    from atomserve.tokenizer import ChatTokenizer

    ids = ChatTokenizer(model_dir).encode(TEXT)
    ref = full_logits(weights_bf16, config, ids)
    int8 = full_logits(weights_bf16, config, ids, quantized=True)
    target = torch.tensor(ids[1:])

    def perplexity(logits):
        return torch.nn.functional.cross_entropy(logits[:-1], target).exp().item()

    kl = torch.nn.functional.kl_div(
        int8.log_softmax(-1), ref.log_softmax(-1), log_target=True, reduction="batchmean"
    ).item()
    print(f"KL {kl:.4f}, perplexity {perplexity(ref):.3f} -> {perplexity(int8):.3f}")
    assert kl < KL_LIMIT
    assert (int8.argmax(-1) == ref.argmax(-1)).float().mean() > 0.95
    assert perplexity(int8) == pytest.approx(perplexity(ref), rel=0.01)


KL_LIMIT = 0.02
