# Block allocator tests: pure bookkeeping, so every rule can be checked exactly.

import pytest

from atomserve.block_allocator import BlockAllocator, OutOfBlocksError, blocks_for_tokens


def test_blocks_for_tokens_rounds_up():
    """A partly filled block still occupies a whole block."""
    assert [blocks_for_tokens(n, 16) for n in (0, 1, 15, 16, 17, 32, 33)] == [0, 1, 1, 1, 2, 2, 3]


def test_allocate_hands_out_distinct_blocks():
    allocator = BlockAllocator(8)
    a, b = allocator.allocate(3), allocator.allocate(5)
    assert len(set(a + b)) == 8
    assert set(a + b) == set(range(8))
    assert allocator.num_free == 0


def test_out_of_blocks_raises_and_takes_nothing():
    """A request that cannot be met fully is refused without a partial allocation."""
    allocator = BlockAllocator(4)
    allocator.allocate(3)
    with pytest.raises(OutOfBlocksError):
        allocator.allocate(2)
    assert allocator.num_free == 1


def test_freed_blocks_are_reused():
    allocator = BlockAllocator(2)
    blocks = allocator.allocate(2)
    allocator.free(blocks)
    assert allocator.num_free == 2
    assert set(allocator.allocate(2)) == set(blocks)


def test_double_free_and_foreign_blocks_are_rejected():
    """Freeing a block twice would let two sequences share it; that must fail loudly."""
    allocator = BlockAllocator(4)
    blocks = allocator.allocate(2)
    allocator.free(blocks)
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free(blocks[:1])
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free([99])
