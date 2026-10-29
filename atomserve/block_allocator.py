# The block allocator: which pages of the KV pool are free, and which hold reusable prefixes.
#
# The KV pool (kv_cache.py) is carved into fixed-size blocks of `block_size`
# tokens. A sequence owns a list of blocks, its *block table*, which grows one
# block at a time as the sequence gets longer. Because any free block will do,
# memory never fragments and nothing has to be reserved up front for a
# sequence's maximum length: that is the whole idea of PagedAttention.
#
# Prefix caching. A full block's K/V depend only on its tokens and on every
# token before it, so a full block can be named by a *chained* content hash:
#
#     hash_i = sha256(hash_{i-1} || tokens of block i)
#
# Equal hashes mean equal prefixes, so equal K/V. When a new request starts
# with a prefix some earlier request already computed (the same system prompt,
# the earlier turns of a chat), its block table simply points at those blocks
# and prefill starts after them. Blocks are reference-counted, since several
# sequences may share one. A block whose last user finished keeps its contents
# and its hash: it is "free but cached", reusable by lookup until allocate()
# needs the memory, which evicts the least recently freed block first.
#
# Only full blocks are ever shared, and full blocks are never written again,
# so sharing needs no copy-on-write.
#
# This file is pure bookkeeping over block ids (plain Python, no torch), so the
# scheduler can decide who gets memory without touching the GPU.

import hashlib
from array import array
from collections import OrderedDict


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """Number of blocks needed to hold `num_tokens` tokens (ceiling division)."""
    return -(-num_tokens // block_size)


def block_hash(parent: bytes, tokens: list[int]) -> bytes:
    """Content address of a full block: commits to its tokens and every token before it.

    parent: hash of the previous block (b"" for the first block)

    SHA-256 rather than Python's hash(): a collision would silently give a
    request someone else's K/V, so it must be practically impossible.
    """
    return hashlib.sha256(parent + array("I", tokens).tobytes()).digest()


class OutOfBlocksError(RuntimeError):
    """Raised when more blocks are requested than are free."""


class BlockAllocator:
    """Hands out physical block ids in [0, num_blocks), with refcounts and a prefix cache.

    A block is in exactly one state:
      in use          refcount > 0, owned by one or more sequences
      free, cached    refcount 0, still has a hash: reusable via lookup()
      free, empty     refcount 0, no hash
    Both free states live in `_free`, oldest first, and count as free memory.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._refcount = [0] * num_blocks
        self._free: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks))
        self._hash_of: dict[int, bytes] = {}  # block -> hash, for cached blocks
        self._block_of: dict[bytes, int] = {}  # hash -> block, the cache index

    @property
    def num_free(self) -> int:
        """Blocks available to allocate right now (cached ones are evicted on demand)."""
        return len(self._free)

    @property
    def num_cached(self) -> int:
        """Blocks holding a reusable prefix, in use or not."""
        return len(self._block_of)

    def allocate(self, n: int) -> list[int]:
        """Take `n` free blocks (oldest first), or raise OutOfBlocksError and take none.

        A taken block that was cached is evicted: its contents are about to be
        overwritten, so its hash must no longer find it.
        """
        if n > len(self._free):
            raise OutOfBlocksError(f"need {n} KV blocks, only {len(self._free)} free")
        blocks = []
        for _ in range(n):
            block, _ = self._free.popitem(last=False)
            if (h := self._hash_of.pop(block, None)) is not None:
                del self._block_of[h]
            self._refcount[block] = 1
            blocks.append(block)
        return blocks

    def free(self, blocks: list[int]) -> None:
        """Drop one reference to each block; a block nobody uses becomes free (still cached).

        Pass a sequence's blocks in order: they are released last block first,
        so a prefix's later blocks get evicted before its earlier ones, which
        are the ones more requests share.
        """
        for block in reversed(blocks):
            if not 0 <= block < self.num_blocks or self._refcount[block] == 0:
                raise ValueError(f"block {block} is not allocated")
            self._refcount[block] -= 1
            if self._refcount[block] == 0:
                self._free[block] = None  # newest end: evicted last

    def lookup(self, h: bytes) -> int | None:
        """The block holding the prefix with hash `h`, if it is still cached."""
        return self._block_of.get(h)

    def acquire(self, block: int) -> None:
        """Add a reference to a cached block found by lookup(), reviving it if free."""
        if self._refcount[block] == 0:
            del self._free[block]
        self._refcount[block] += 1

    def register(self, block: int, h: bytes) -> None:
        """Record that `block` (full, computed) holds the prefix with hash `h`.

        If another block already holds that prefix (two requests prefilled the
        same text at once), the first one stays the cached copy.
        """
        if h not in self._block_of and block not in self._hash_of:
            self._block_of[h] = block
            self._hash_of[block] = h
