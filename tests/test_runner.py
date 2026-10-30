# Runner tests: the flat batch must be a pure packing trick. Where a sequence's
# blocks live and which other sequences share its forward pass may change
# speed, never results. All on the tiny random model on CPU.

import pytest
import torch

from atomserve.block_allocator import BlockAllocator, blocks_for_tokens
from atomserve.kv_cache import KVPool
from atomserve.runner import ModelRunner, Row, rows_for
from atomserve.sequence import Sequence

CPU = torch.device("cpu")


def make_runner(weights, config, num_blocks: int, block_size: int) -> ModelRunner:
    device, dtype = weights.embed.device, weights.embed.dtype
    return ModelRunner(weights, config, KVPool(config, num_blocks, block_size, device, dtype))


def reserve_blocks(seq: Sequence, allocator: BlockAllocator, block_size: int) -> None:
    """Grow seq's block table to cover all its tokens (the scheduler's job in the engine)."""
    needed = blocks_for_tokens(seq.num_tokens, block_size) - len(seq.block_table)
    seq.block_table.extend(allocator.allocate(max(needed, 0)))


def test_prepare_flattens_a_decode_and_a_prompt_chunk(tiny_weights, tiny_config):
    """Hand-checked metadata for a decode step next to a long partial prefill (block_size 4)."""
    runner = make_runner(tiny_weights, tiny_config, num_blocks=8, block_size=4)
    decoding = Sequence(0, token_ids=[1, 2, 3, 4, 5, 6], num_computed_tokens=5, block_table=[7, 2])
    chunked = Sequence(1, token_ids=list(range(10, 30)), block_table=[5, 0, 1])
    batch = runner.prepare(rows_for([decoding, chunked], [1, 10]))

    assert batch.input_ids.tolist() == [6, *range(10, 20)]
    assert batch.positions.tolist() == [5, *range(10)]
    assert batch.slot_mapping.tolist() == [9, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    assert batch.logits_idx.tolist() == [0]  # the chunk has not reached its last token
    # The decode: row 0, attending to positions 0-5 (blocks 7 and 2).
    assert batch.decode_rows.tolist() == [0]
    assert batch.decode_slots.tolist() == [[28, 29, 30, 31, 8, 9]]
    assert batch.decode_mask is None  # one decode: nothing to pad
    # The chunk: rows 1-10, attending to positions 0-9 (blocks 5, 0, 1).
    [span] = batch.prefills
    assert (span.rows.start, span.rows.stop) == (1, 11)
    assert span.context_slots.tolist() == [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]


def test_short_rows_become_one_decode_query_per_token(tiny_weights, tiny_config):
    """A 3-token row (a small chunk, or a decode plus drafts to verify) is attended as three
    single queries; query j sees the keys up to its own position, which is the causal mask."""
    runner = make_runner(tiny_weights, tiny_config, num_blocks=8, block_size=4)
    row = Row(tokens=[7, 8, 9], start=5, block_table=[3, 6], num_logits=3)
    batch = runner.prepare([row])
    assert batch.prefills == []
    assert batch.decode_rows.tolist() == [0, 1, 2]
    assert batch.decode_mask.tolist() == [
        [True] * 6 + [False] * 2,
        [True] * 7 + [False],
        [True] * 8,
    ]
    assert batch.logits_idx.tolist() == [0, 1, 2]


def test_decodes_of_different_lengths_are_padded_and_masked(tiny_weights, tiny_config):
    runner = make_runner(tiny_weights, tiny_config, num_blocks=8, block_size=4)
    short = Sequence(0, token_ids=[1, 2], num_computed_tokens=1, block_table=[3])
    long = Sequence(1, token_ids=list(range(6)), num_computed_tokens=5, block_table=[1, 6])
    batch = runner.prepare(rows_for([short, long], [1, 1]))
    assert batch.decode_slots.shape == (2, 6)
    assert batch.decode_slots[1].tolist() == [4, 5, 6, 7, 24, 25]
    assert batch.decode_mask.tolist() == [[True] * 2 + [False] * 4, [True] * 6]


def test_prepare_rejects_a_block_table_that_is_too_short(tiny_weights, tiny_config):
    runner = make_runner(tiny_weights, tiny_config, num_blocks=8, block_size=4)
    with pytest.raises(ValueError, match="block table"):
        runner.prepare(rows_for([Sequence(0, token_ids=list(range(5)), block_table=[0])], [5]))


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
    a = greedy_logits(ordered, Sequence(0, token_ids=list(prompt), block_table=[0, 1, 2, 3]), 9)
    b = greedy_logits(scattered, Sequence(0, token_ids=list(prompt), block_table=[13, 2, 9, 6]), 9)
    assert torch.equal(a, b)


def test_mixed_batch_matches_each_sequence_alone(tiny_weights, tiny_config, full_logits):
    """Prefills, prompt chunks and decodes of different sequences share one forward pass
    and a pool of interleaved blocks, yet every logits row equals the uncached
    single-sequence reference."""
    runner = make_runner(tiny_weights, tiny_config, num_blocks=16, block_size=4)
    allocator = BlockAllocator(16)
    a = Sequence(0, token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = Sequence(1, token_ids=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    c = Sequence(2, token_ids=[31, 32, 33])

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
