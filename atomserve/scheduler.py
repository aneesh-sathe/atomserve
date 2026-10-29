# The scheduler: decides, every step, which sequences run and how many tokens each.
#
# This is continuous (iteration-level) batching. Instead of collecting a batch
# of requests and running it to completion, the batch is rebuilt every step:
# a request that finishes leaves immediately and a new one joins at once, so
# the GPU never waits for the slowest request in a batch.
#
# Each step, under two limits (max_num_seqs sequences, max_num_batched_tokens
# tokens per forward pass):
#   1. Running sequences go first, one decode token each, oldest first. If the
#      pool has no block for one, the most recently admitted sequence is
#      *preempted*: its blocks are freed and it goes back to the front of the
#      queue, to be recomputed from its tokens later. Recomputing is cheaper
#      than it sounds (prefill is fast) and needs no swap space.
#   2. Waiting sequences are admitted first come, first served, while the
#      token budget, the sequence limit and free blocks allow.
#
# It only does bookkeeping on Sequence objects and block ids, plain Python
# with no torch, so every policy decision is testable without a GPU.

from collections import deque
from dataclasses import dataclass, field

from atomserve.block_allocator import BlockAllocator, blocks_for_tokens
from atomserve.config import EngineConfig
from atomserve.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """The plan for one engine step.

    seqs[i] runs its next num_new_tokens[i] tokens: its whole prompt (prefill)
    or its newest token (decode). `preempted` lists sequences evicted to make
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
            n = seq.num_tokens - seq.num_computed_tokens
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
            n = seq.num_tokens - seq.num_computed_tokens  # whole prompt (or recompute)
            # A prompt larger than the whole budget may only run on an empty step.
            if (n > budget and out.seqs) or not self._reserve(seq, n):
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

    def _release_blocks(self, seq: Sequence) -> None:
        self.allocator.free(seq.block_table)
        seq.block_table = []

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
