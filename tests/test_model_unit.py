# Fast tests for the model's building blocks, on small random tensors (CPU, no weights).
# Each test pins down one mathematical property the block must have.

import math

import pytest
import torch

from atomserve.config import ModelConfig
from atomserve.model import apply_rope, causal_attention, rms_norm, rope_cos_sin

torch.manual_seed(0)


def test_rms_norm_gives_unit_rms():
    """With a weight of ones, every output vector has root-mean-square 1."""
    x = torch.randn(4, 64) * 7.0
    out = rms_norm(x, torch.ones(64), eps=1e-6)
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-5)


def test_rms_norm_applies_weight_per_channel():
    """The learned weight scales each channel independently after normalising."""
    x = torch.randn(3, 8)
    w = torch.arange(1.0, 9.0)
    assert torch.allclose(rms_norm(x, w, 1e-6), rms_norm(x, torch.ones(8), 1e-6) * w)


def test_rope_is_identity_at_position_zero():
    """Rotation angle is zero at position 0, so vectors pass through unchanged."""
    x = torch.randn(1, 2, 16)
    cos, sin = rope_cos_sin(torch.tensor([0]), 16, 10_000.0, torch.float32)
    assert torch.allclose(apply_rope(x, cos, sin), x)


def test_rope_preserves_vector_length():
    """RoPE is a rotation, so it never changes a head vector's norm."""
    x = torch.randn(5, 3, 16)
    cos, sin = rope_cos_sin(torch.arange(100, 105), 16, 10_000.0, torch.float32)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_relative_distance():
    """q.k after RoPE is unchanged when both positions shift by the same amount."""
    q, k = torch.randn(1, 1, 16), torch.randn(1, 1, 16)

    def score(m: int, n: int) -> float:
        cq, sq = rope_cos_sin(torch.tensor([m]), 16, 10_000.0, torch.float32)
        ck, sk = rope_cos_sin(torch.tensor([n]), 16, 10_000.0, torch.float32)
        return float((apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum())

    assert score(7, 3) == pytest.approx(score(107, 103), abs=1e-4)


def naive_causal_attention(q, k, v):
    """Reference attention written out longhand: repeat kv heads, mask, softmax, mix."""
    T, num_heads, head_dim = q.shape
    group = num_heads // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(head_dim)  # [H, T, T]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    probs = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return torch.einsum("hts,shd->thd", probs, v)


def test_causal_attention_matches_naive_reference_with_gqa():
    """SDPA with enable_gqa equals the longhand definition (4 query heads over 2 kv heads)."""
    q, k, v = torch.randn(6, 4, 8), torch.randn(6, 2, 8), torch.randn(6, 2, 8)
    assert torch.allclose(causal_attention(q, k, v), naive_causal_attention(q, k, v), atol=1e-5)


def test_first_token_attends_only_to_itself():
    """Under the causal mask, token 0 sees only itself, so its output is exactly v[0]."""
    q, k, v = torch.randn(5, 4, 8), torch.randn(5, 2, 8), torch.randn(5, 2, 8)
    out = causal_attention(q, k, v)
    assert torch.allclose(out[0], v[0].repeat_interleave(2, dim=0), atol=1e-6)


def test_config_rejects_other_architectures():
    """Only Qwen3ForCausalLM is accepted; anything else fails before loading weights."""
    with pytest.raises(ValueError, match="only supports"):
        ModelConfig.from_dict({"architectures": ["LlamaForCausalLM"]})
