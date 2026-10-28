# Runner tests: the flat batch must be a pure packing trick. Where a sequence's
# blocks live and which other sequences share its forward pass may change
# speed, never results. Fast tests use the tiny random model on CPU; the slow
# one repeats the batching check on the real model.

import pytest
import torch

from atomserve.block_allocator import BlockAllocator
from atomserve.generate import generate_greedy, reserve_blocks
from atomserve.kv_cache import KVPool
from atomserve.runner import ModelRunner
from atomserve.sequence import Sequence

CPU = torch.device("cpu")


def make_runner(weights, config, num_blocks: int, block_size: int) -> ModelRunner:
    device, dtype = weights.embed.device, weights.embed.dtype
    return ModelRunner(weights, config, KVPool(config, num_blocks, block_size, device, dtype))


def test_prepare_flattens_a_decode_and_a_prompt_chunk(tiny_weights, tiny_config):
    """Hand-checked metadata for a decode step next to a partial prefill (block_size 4)."""
    runner = make_runner(tiny_weights, tiny_config, num_blocks=8, block_size=4)
    decoding = Sequence(token_ids=[1, 2, 3, 4, 5, 6], num_computed_tokens=5, block_table=[7, 2])
    chunked = Sequence(token_ids=list(range(10, 20)), block_table=[5])
    batch = runner.prepare([decoding, chunked], [1, 3])

    assert batch.input_ids.tolist() == [6, 10, 11, 12]
    assert batch.positions.tolist() == [5, 0, 1, 2]
    assert batch.slot_mapping.tolist() == [9, 20, 21, 22]  # block 2 offset 1; block 5 offsets 0-2
    assert batch.cu_seqlens == [0, 1, 4]
    assert batch.context_slots.tolist() == [28, 29, 30, 31, 8, 9, 20, 21, 22]
    assert batch.cu_ctxlens == [0, 6, 9]
    assert batch.logits_idx.tolist() == [0]  # the chunk has not reached its last token


def test_prepare_rejects_a_block_table_that_is_too_short(tiny_weights, tiny_config):
    runner = make_runner(tiny_weights, tiny_config, num_blocks=8, block_size=4)
    with pytest.raises(ValueError, match="block table"):
        runner.prepare([Sequence(token_ids=list(range(5)), block_table=[0])], [5])


def greedy_logits(runner: ModelRunner, seq: Sequence, steps: int) -> torch.Tensor:
    """Prefill `seq`, then decode greedily; return each step's logits, [steps, vocab]."""
    rows = []
    num_new = len(seq.token_ids)
    for _ in range(steps):
        logits = runner.run([seq], [num_new])
        seq.num_computed_tokens += num_new
        seq.token_ids.append(int(logits[0].argmax()))
        rows.append(logits[0])
        num_new = 1
    return torch.stack(rows)


def test_block_placement_does_not_change_results(tiny_weights, tiny_config):
    """Contiguous blocks and blocks scattered across the pool give bitwise-equal logits."""
    prompt = [3, 14, 15, 9, 26, 5, 35]
    ordered = make_runner(tiny_weights, tiny_config, num_blocks=4, block_size=4)
    scattered = make_runner(tiny_weights, tiny_config, num_blocks=16, block_size=4)
    a = greedy_logits(ordered, Sequence(token_ids=list(prompt), block_table=[0, 1, 2, 3]), 9)
    b = greedy_logits(scattered, Sequence(token_ids=list(prompt), block_table=[13, 2, 9, 6]), 9)
    assert torch.equal(a, b)


def test_mixed_batch_matches_each_sequence_alone(tiny_weights, tiny_config, full_logits):
    """Prefills, prompt chunks and decodes of different sequences share one forward pass
    and a pool of interleaved blocks, yet every logits row equals the uncached
    single-sequence reference."""
    runner = make_runner(tiny_weights, tiny_config, num_blocks=16, block_size=4)
    allocator = BlockAllocator(16)
    a = Sequence(token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = Sequence(token_ids=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    c = Sequence(token_ids=[31, 32, 33])

    def step(seqs: list[Sequence], num_new: list[int]) -> torch.Tensor:
        for seq in seqs:
            reserve_blocks(seq, allocator, 4)  # interleaves the sequences' blocks
        logits = runner.run(seqs, num_new)
        for seq, n in zip(seqs, num_new, strict=True):
            seq.num_computed_tokens += n
        return logits

    def reference(seq: Sequence) -> torch.Tensor:
        return full_logits(tiny_weights, tiny_config, seq.token_ids)[-1]

    # Step 1: A prefills fully, B prefills only its first 7 tokens (no prediction yet).
    logits = step([a, b], [9, 7])
    assert logits.shape[0] == 1
    assert torch.allclose(logits[0], reference(a), atol=1e-5)
    a.token_ids.append(int(logits[0].argmax()))

    # Step 2: A decodes, B finishes its prompt, C prefills from scratch.
    logits = step([a, b, c], [1, 5, 3])
    for row, seq in zip(logits, [a, b, c], strict=True):
        assert torch.allclose(row, reference(seq), atol=1e-5)
        seq.token_ids.append(int(row.argmax()))

    # Steps 3-6: all three decode together, crossing block boundaries.
    for _ in range(4):
        logits = step([a, b, c], [1, 1, 1])
        for row, seq in zip(logits, [a, b, c], strict=True):
            assert torch.allclose(row, reference(seq), atol=1e-5)
            seq.token_ids.append(int(row.argmax()))


PROMPTS = [
    [785, 6722, 315, 9625, 374],  # "The capital of France is"
    [9707, 11, 847, 829, 374],  # "Hello, my name is"
    [16, 11, 220, 17, 11, 220, 18, 11],  # "1, 2, 3,"
]


@pytest.mark.slow
def test_batched_greedy_matches_alone_on_real_model(weights_fp32, config):
    """Three prompts prefilled and decoded together produce the same 24 greedy tokens
    as each generated on its own."""
    alone = [generate_greedy(weights_fp32, config, p, 24) for p in PROMPTS]

    runner = make_runner(weights_fp32, config, num_blocks=32, block_size=4)
    allocator = BlockAllocator(32)
    seqs = [Sequence(token_ids=list(p)) for p in PROMPTS]
    num_new = [len(p) for p in PROMPTS]
    for _ in range(24):
        for seq in seqs:
            reserve_blocks(seq, allocator, 4)
        logits = runner.run(seqs, num_new)
        for seq, n, row in zip(seqs, num_new, logits, strict=True):
            seq.num_computed_tokens += n
            seq.token_ids.append(int(row.argmax()))
        num_new = [1] * len(seqs)

    assert [seq.token_ids[len(p) :] for seq, p in zip(seqs, PROMPTS, strict=True)] == alone
