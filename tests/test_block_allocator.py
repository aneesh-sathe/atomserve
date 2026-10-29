# Block allocator tests: pure bookkeeping, so every rule (allocation, refcounts, the
# prefix cache and its eviction order) can be checked exactly.

import pytest

from atomserve.block_allocator import (
    BlockAllocator,
    OutOfBlocksError,
    block_hash,
    blocks_for_tokens,
)


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


H1, H2, H3 = (block_hash(b"", [i]) for i in (1, 2, 3))


def test_block_hash_chains_on_the_whole_prefix():
    """The same block of tokens after different prefixes gets different hashes."""
    tokens = [5, 6, 7, 8]
    assert block_hash(b"", tokens) == block_hash(b"", list(tokens))
    assert block_hash(H1, tokens) != block_hash(H2, tokens)


def test_shared_block_stays_in_use_until_every_owner_frees_it():
    allocator = BlockAllocator(4)
    [block] = allocator.allocate(1)
    allocator.register(block, H1)
    allocator.acquire(allocator.lookup(H1))  # a second sequence shares it
    allocator.free([block])
    assert allocator.num_free == 3  # still used by the second owner
    allocator.free([block])
    assert allocator.num_free == 4


def test_freed_block_stays_cached_and_can_be_revived():
    allocator = BlockAllocator(4)
    [block] = allocator.allocate(1)
    allocator.register(block, H1)
    allocator.free([block])
    assert allocator.num_free == 4 and allocator.lookup(H1) == block
    allocator.acquire(block)
    assert allocator.num_free == 3


def test_allocation_evicts_the_least_recently_freed_cached_block():
    allocator = BlockAllocator(3)
    a, b, c = allocator.allocate(3)
    for block, h in ((a, H1), (b, H2), (c, H3)):
        allocator.register(block, h)
    allocator.free([b])
    allocator.free([a, c])  # released last block first: c, then a
    assert allocator.allocate(2) == [b, c]
    assert allocator.lookup(H2) is None and allocator.lookup(H3) is None
    assert allocator.lookup(H1) == a  # the earliest block of its prefix survives longest


def test_register_keeps_the_first_copy_of_a_prefix():
    allocator = BlockAllocator(2)
    first, second = allocator.allocate(2)
    allocator.register(first, H1)
    allocator.register(second, H1)  # computed concurrently with the same content
    assert allocator.lookup(H1) == first
    assert allocator.num_cached == 1
