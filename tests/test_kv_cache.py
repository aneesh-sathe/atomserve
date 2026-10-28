# KV cache tests. The fast ones check the bookkeeping on a tiny config (CPU); the
# slow ones prove, on the real model, that caching changes speed and nothing else.

import pytest
import torch

from atomserve.config import ModelConfig
from atomserve.generate import generate_greedy
from atomserve.kv_cache import KVCache
from atomserve.model import compute_logits, forward

TINY = ModelConfig(
    vocab_size=10,
    hidden_size=8,
    intermediate_size=16,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=4,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
    tie_word_embeddings=True,
    max_position_embeddings=64,
)


def test_append_returns_full_history_and_advance_moves_length():
    """append() exposes prefix + new tokens; length only moves once all layers wrote."""
    cache = KVCache(TINY, max_len=8, device=torch.device("cpu"), dtype=torch.float32)
    first = torch.randn(3, 2, 4)
    for layer in range(TINY.num_layers):
        k_all, _ = cache.append(layer, first, first)
        assert k_all.shape == (3, 2, 4)
    cache.advance(3)

    second = torch.randn(1, 2, 4)
    k_all, v_all = cache.append(0, second, second)
    assert cache.length == 3
    assert torch.equal(k_all, torch.cat([first, second]))
    assert torch.equal(v_all, torch.cat([first, second]))


def test_append_past_capacity_raises():
    """Writing beyond max_len is an error, never a silent overwrite."""
    cache = KVCache(TINY, max_len=4, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="KV cache full"):
        cache.append(0, torch.randn(5, 2, 4), torch.randn(5, 2, 4))


PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is"


@pytest.mark.slow
def test_prefill_then_decode_logits_equal_full_recompute(weights_fp32, config):
    """At every step, the cached forward pass yields the same logits as recomputing
    the whole sequence from scratch (fp32, so any mismatch is a real bug)."""
    device = weights_fp32.embed.device
    cache = KVCache(config, max_len=32, device=device, dtype=torch.float32)
    ids = list(PROMPT)

    def full_logits() -> torch.Tensor:
        hidden = forward(
            weights_fp32,
            config,
            torch.tensor(ids, device=device),
            torch.arange(len(ids), device=device),
        )
        return compute_logits(weights_fp32, hidden)

    with torch.inference_mode():
        # Prefill: every prompt position must match.
        hidden = forward(
            weights_fp32,
            config,
            torch.tensor(ids, device=device),
            torch.arange(len(ids), device=device),
            cache,
        )
        assert (compute_logits(weights_fp32, hidden) - full_logits()).abs().max() < 1e-3

        # Decode: feed one token at a time and compare the newest position.
        for _ in range(8):
            next_id = int(compute_logits(weights_fp32, hidden[-1:]).argmax())
            ids.append(next_id)
            pos = torch.tensor([len(ids) - 1], device=device)
            hidden = forward(
                weights_fp32, config, torch.tensor([next_id], device=device), pos, cache
            )
            assert (compute_logits(weights_fp32, hidden) - full_logits()[-1:]).abs().max() < 1e-3
    assert cache.length == len(ids)


@pytest.mark.slow
def test_cached_and_uncached_greedy_agree(weights_fp32, config):
    """32 greedy tokens are identical with and without the cache."""
    cached = generate_greedy(weights_fp32, config, PROMPT, 32, use_cache=True)
    uncached = generate_greedy(weights_fp32, config, PROMPT, 32, use_cache=False)
    assert cached == uncached
