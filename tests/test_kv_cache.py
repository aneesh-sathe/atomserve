# KV pool tests. The fast ones check storage addressing on a toy pool (CPU); the
# slow ones prove, on the real model, that caching changes speed and nothing else.

import pytest
import torch

from atomserve.kv_cache import KVPool
from atomserve.runner import ModelRunner, slot_of
from atomserve.sequence import Sequence

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
