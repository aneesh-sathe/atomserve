# The block allocator: which pages of the KV pool are free.
#
# The KV pool (kv_cache.py) is carved into fixed-size blocks of `block_size`
# tokens. A sequence owns a list of blocks, its *block table*, which grows one
# block at a time as the sequence gets longer. Because any free block will do,
# memory never fragments and nothing has to be reserved up front for a
# sequence's maximum length: that is the whole idea of PagedAttention.
#
# This file is pure bookkeeping over block ids (plain Python, no torch), so the
# scheduler can decide who gets memory without touching the GPU.

from collections import deque


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """Number of blocks needed to hold `num_tokens` tokens (ceiling division)."""
    return -(-num_tokens // block_size)


class OutOfBlocksError(RuntimeError):
    """Raised when more blocks are requested than are free."""


class BlockAllocator:
    """Hands out and takes back physical block ids in the range [0, num_blocks).

    Every block is either free (in `_free`) or owned by exactly one sequence
    (in `_used`); keeping both lets us catch double frees and foreign ids, which
    would otherwise silently let two sequences overwrite each other's K/V.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        """Blocks available right now."""
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        """Take `n` free blocks, or raise OutOfBlocksError and take none."""
        if n > len(self._free):
            raise OutOfBlocksError(f"need {n} KV blocks, only {len(self._free)} free")
        blocks = [self._free.popleft() for _ in range(n)]
        self._used.update(blocks)
        return blocks

    def free(self, blocks: list[int]) -> None:
        """Return blocks to the pool. Their contents become garbage, to be overwritten."""
        for block in blocks:
            if block not in self._used:
                raise ValueError(f"block {block} is not allocated")
            self._used.remove(block)
            self._free.append(block)
