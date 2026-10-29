# The scheduler: decides, every step, which sequences run and how many tokens each.
#
# This is continuous (iteration-level) batching. Instead of collecting a batch
# of requests and running it to completion, the batch is rebuilt every step:
# a request that finishes leaves immediately and a new one joins at once, so
# the GPU never waits for the slowest request in a batch.
#
# Each step, under two limits (max_num_seqs sequences, max_num_batched_tokens
# tokens per forward pass):
#   1. Running sequences go first, oldest first: one token for a decode, or the
#      next chunk of a prompt still being prefilled. If the pool has no block
#      for one, the most recently admitted sequence is *preempted*: its blocks
#      are freed and it goes back to the front of the queue, to be recomputed
#      from its tokens later. Recomputing is cheaper than it sounds (prefill is
#      fast) and needs no swap space.
#   2. Waiting sequences are admitted first come, first served, while the
#      token budget, the sequence limit and free blocks allow.
#
# Chunked prefill: a prompt never has to fit the budget in one piece. It takes
# whatever budget is left this step and continues next step. So a long prompt
# neither waits for an empty step (blocking everyone queued behind it) nor
# hogs one giant step: every step stays bounded by the budget, which bounds
# how long running streams wait for their next token.
#
# Prefix caching: on admission, a sequence first looks up its leading full
# blocks in the allocator's prefix cache and starts with those already
# "computed"; after every step, its newly completed blocks are registered so
# later requests can reuse them. A preempted sequence usually finds its own
# blocks still cached, so recomputing it is cheap too.
#
# It only does bookkeeping on Sequence objects and block ids, plain Python
# with no torch, so every policy decision is testable without a GPU.

from collections import deque
from dataclasses import dataclass, field

from atomserve.block_allocator import BlockAllocator, block_hash, blocks_for_tokens
from atomserve.config import EngineConfig
from atomserve.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """The plan for one engine step.

    seqs[i] runs its next num_new_tokens[i] tokens: its whole prompt, a chunk
    of it (prefill), or its newest token (decode). `preempted` lists sequences evicted to make
    room this step; they are waiting again, not part of this step.
    """

    seqs: list[Sequence] = field(default_factory=list)
    num_new_tokens: list[int] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(self.num_new_tokens)


class Scheduler:
    """Owns the waiting queue, the running list and the block allocator."""

    def __init__(self, config: EngineConfig, eos_token_ids: frozenset[int]):
        self.config = config
        self.eos_token_ids = eos_token_ids
        self.allocator = BlockAllocator(config.num_blocks)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order

    def add(self, seq: Sequence) -> None:
        """Queue a new request."""
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        """Pick this step's sequences and token counts, reserving their KV blocks."""
        budget = self.config.max_num_batched_tokens
        out = SchedulerOutput()

        # 1. Running sequences: each needs one token, plus a block when it crosses
        #    a block boundary. Iterate over a copy: preemption shrinks the list.
        for seq in list(self.running):
            if seq.status is not SequenceStatus.RUNNING:
                continue  # preempted earlier in this loop to make room
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)
            if n == 0:
                continue  # budget spent; it simply waits for the next step
            while not self._reserve(seq, n):
                victim = self.running[-1]  # newest first: it has the least work to redo
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            if seq.status is SequenceStatus.RUNNING:
                out.seqs.append(seq)
                out.num_new_tokens.append(n)
                budget -= n

        # 2. Admit waiting sequences. Skipped after a preemption: memory is short,
        #    and admitting would just evict someone again next step.
        while self.waiting and not out.preempted and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            if self.config.enable_prefix_caching:
                self._use_cached_prefix(seq)
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)  # prompt, or a chunk
            if n == 0 or not self._reserve(seq, n):
                self._release_blocks(seq)  # undo any prefix hit; retry next step
                seq.num_computed_tokens = seq.num_cached_tokens = 0
                break
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.seqs.append(seq)
            out.num_new_tokens.append(n)
            budget -= n
        return out

    def update(self, out: SchedulerOutput, sampled: list[int]) -> list[tuple[Sequence, int]]:
        """Apply a finished step: advance sequences, append samples, retire finished ones.

        sampled: one token per sequence of `out` that reached the end of its
                 known tokens (see Sequence.reaches_end), in order
        returns  (sequence, new token) pairs; a finished sequence has status FINISHED
        """
        samples = iter(sampled)
        new_tokens = []
        for seq, n in zip(out.seqs, out.num_new_tokens, strict=True):
            sampled_now = seq.reaches_end(n)  # decided before advancing
            seq.num_computed_tokens += n
            if self.config.enable_prefix_caching:
                self._register_full_blocks(seq)
            if not sampled_now:
                continue
            token = next(samples)
            seq.token_ids.append(token)
            new_tokens.append((seq, token))
            if reason := self._finish_reason(seq, token):
                self._finish(seq, reason)
        if next(samples, None) is not None:
            raise ValueError("more sampled tokens than sampling sequences")
        return new_tokens

    def abort(self, seq_id: int) -> Sequence | None:
        """Cancel a request wherever it is; returns it, or None if unknown or already done."""
        for seq in [*self.waiting, *self.running]:
            if seq.id == seq_id:
                if seq in self.waiting:
                    self.waiting.remove(seq)
                self._finish(seq, "abort")
                return seq
        return None

    def _finish_reason(self, seq: Sequence, token: int) -> str | None:
        """Why `seq` should stop after appending `token`, or None to continue."""
        if token in self.eos_token_ids and not seq.params.ignore_eos:
            return "stop"
        if len(seq.output_ids) >= seq.params.max_tokens:
            return "length"
        if seq.num_tokens >= self.config.max_model_len:
            return "length"
        return None

    def _reserve(self, seq: Sequence, num_new_tokens: int) -> bool:
        """Grow seq's block table to hold `num_new_tokens` more tokens. All or nothing."""
        needed = blocks_for_tokens(seq.num_computed_tokens + num_new_tokens, self.config.block_size)
        missing = needed - len(seq.block_table)
        if missing > self.allocator.num_free:
            return False
        if missing > 0:
            seq.block_table.extend(self.allocator.allocate(missing))
        return True

    def _use_cached_prefix(self, seq: Sequence) -> None:
        """Start a sequence with KV blocks already in the prefix cache, if any.

        Walks the sequence's full blocks in order and stops at the first miss.
        At least one token is always left to compute: its logits are needed to
        sample the next token.
        """
        bs = self.config.block_size
        parent = b""
        for i in range((seq.num_tokens - 1) // bs):
            h = block_hash(parent, seq.token_ids[i * bs : (i + 1) * bs])
            if (block := self.allocator.lookup(h)) is None:
                break
            self.allocator.acquire(block)
            seq.block_table.append(block)
            seq.block_hashes.append(h)
            parent = h
        seq.num_computed_tokens = len(seq.block_table) * bs
        if not seq.output_ids:  # count hits on the prompt, not on recomputing a preempted reply
            seq.num_cached_tokens = seq.num_computed_tokens

    def _register_full_blocks(self, seq: Sequence) -> None:
        """Publish blocks this step completed (all tokens computed) to the prefix cache."""
        bs = self.config.block_size
        while len(seq.block_hashes) < seq.num_computed_tokens // bs:
            i = len(seq.block_hashes)
            parent = seq.block_hashes[-1] if i else b""
            h = block_hash(parent, seq.token_ids[i * bs : (i + 1) * bs])
            self.allocator.register(seq.block_table[i], h)
            seq.block_hashes.append(h)

    def _release_blocks(self, seq: Sequence) -> None:
        self.allocator.free(seq.block_table)
        seq.block_table, seq.block_hashes = [], []

    def _preempt(self, seq: Sequence) -> None:
        """Evict a running sequence: free its KV and requeue it at the front."""
        self.running.remove(seq)
        self._release_blocks(seq)
        seq.num_computed_tokens = 0  # its K/V are gone; recompute prompt + output so far
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def _finish(self, seq: Sequence, reason: str) -> None:
        if seq in self.running:
            self.running.remove(seq)
        self._release_blocks(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
