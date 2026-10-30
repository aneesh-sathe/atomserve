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
# Speculative decoding: a decoding sequence also gets `num_lookahead` extra KV
# slots (and token budget) for the tokens a draft model will propose; the
# runner reports how many were accepted, and the rest are simply never
# counted as computed. Both models' KV pools share one block table, so a block
# enters the prefix cache only once *both* have computed it.
#
# It only does bookkeeping on Sequence objects and block ids, plain Python
# with no torch, so every policy decision is testable without a GPU.

from collections import deque
from dataclasses import dataclass, field

from atomserve.block_allocator import BlockAllocator, block_hash, blocks_for_tokens
from atomserve.config import EngineConfig
from atomserve.sequence import Sequence, SequenceStatus, StepResult


@dataclass
class SchedulerOutput:
    """The plan for one engine step.

    seqs[i] runs its next num_new_tokens[i] tokens: its whole prompt, a chunk
    of it (prefill), or its newest token (decode). A decode may also have
    num_lookahead[i] > 0: KV slots reserved for that many speculative tokens.
    `preempted` lists sequences evicted to make room this step; they are
    waiting again, not part of this step.
    """

    seqs: list[Sequence] = field(default_factory=list)
    num_new_tokens: list[int] = field(default_factory=list)
    num_lookahead: list[int] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(self.num_new_tokens) + sum(self.num_lookahead)


class Scheduler:
    """Owns the waiting queue, the running list and the block allocator."""

    def __init__(self, config: EngineConfig, eos_token_ids: frozenset[int]):
        self.config = config
        self.eos_token_ids = eos_token_ids
        self.allocator = BlockAllocator(config.num_blocks)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order
        self._speculate = False  # decided at the start of each schedule()

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
        # Speculation pays off only while the batch is small: a big batch already
        # shares each weight read between many sequences, so drafting is pure overhead.
        self._speculate = len(self.running) <= self.config.max_speculative_batch

        # 1. Running sequences: each needs one token, plus a block when it crosses
        #    a block boundary. Iterate over a copy: preemption shrinks the list.
        for seq in list(self.running):
            if seq.status is not SequenceStatus.RUNNING:
                continue  # preempted earlier in this loop to make room
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)
            if n == 0:
                continue  # budget spent; it simply waits for the next step
            lookahead = self._lookahead(seq, n, budget)
            while not self._reserve(seq, n + lookahead):
                victim = self.running[-1]  # newest first: it has the least work to redo
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            if seq.status is SequenceStatus.RUNNING:
                out.seqs.append(seq)
                out.num_new_tokens.append(n)
                out.num_lookahead.append(lookahead)
                budget -= n + lookahead

        # 2. Admit waiting sequences. Skipped after a preemption: memory is short,
        #    and admitting would just evict someone again next step.
        while self.waiting and not out.preempted and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            if self.config.enable_prefix_caching:
                self._use_cached_prefix(seq)
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)  # prompt, or a chunk
            if n == 0 or not self._reserve(seq, n):
                self._release_blocks(seq)  # undo any prefix hit; retry next step
                seq.num_computed_tokens = seq.draft_num_computed = seq.num_cached_tokens = 0
                break
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.seqs.append(seq)
            out.num_new_tokens.append(n)
            out.num_lookahead.append(0)
            budget -= n
        return out

    def update(
        self, out: SchedulerOutput, results: list[StepResult]
    ) -> list[tuple[Sequence, list[int]]]:
        """Apply a finished step: advance sequences, append new tokens, retire finished ones.

        results[i] is what the runner did for out.seqs[i]
        returns  (sequence, its new tokens) for every sequence that got any; a finished
                 sequence has status FINISHED and only the tokens up to its stop
        """
        new_tokens = []
        for seq, result in zip(out.seqs, results, strict=True):
            appended = []
            for token in result.tokens:
                seq.token_ids.append(token)
                appended.append(token)
                if reason := self._finish_reason(seq, token):
                    self._finish(seq, reason)
                    break
            if appended:
                new_tokens.append((seq, appended))
            if seq.status is SequenceStatus.FINISHED:
                continue
            # K/V exist only for tokens that are in token_ids, and the newest token is
            # never computed yet (its K/V come with the next step).
            seq.num_computed_tokens = min(result.num_computed, seq.num_tokens - 1)
            seq.draft_num_computed = min(result.draft_num_computed, seq.num_computed_tokens)
            if self.config.enable_prefix_caching:
                self._register_full_blocks(seq)
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

    def _lookahead(self, seq: Sequence, n: int, budget: int) -> int:
        """Speculative tokens to reserve for `seq` this step (0 when not speculating).

        Only decodes speculate, only while the batch is small (max_speculative_batch),
        and never past the sequence's limits or the step's token budget. A presence
        penalty depends on every token generated before each position, which drafted
        tokens would change; those requests don't speculate.
        """
        k = self.config.num_speculative_tokens
        if k == 0 or not self._speculate or n != 1 or seq.num_tokens - seq.num_computed_tokens != 1:
            return 0
        if seq.params.presence_penalty != 0:
            return 0
        remaining_output = seq.params.max_tokens - len(seq.output_ids) - 1
        remaining_context = self.config.max_model_len - seq.num_tokens - 1
        return max(0, min(k, budget - n, remaining_output, remaining_context))

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
        seq.num_computed_tokens = seq.draft_num_computed = len(seq.block_table) * bs
        if not seq.output_ids:  # count hits on the prompt, not on recomputing a preempted reply
            seq.num_cached_tokens = seq.num_computed_tokens

    def _register_full_blocks(self, seq: Sequence) -> None:
        """Publish blocks this step completed (all tokens computed) to the prefix cache.

        With a draft model, a block counts as computed only once both models'
        pools hold its K/V: a later request reusing it skips both.
        """
        bs = self.config.block_size
        computed = seq.num_computed_tokens
        if self.config.num_speculative_tokens:
            computed = min(computed, seq.draft_num_computed)
        while len(seq.block_hashes) < computed // bs:
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
        seq.num_computed_tokens = seq.draft_num_computed = 0  # K/V gone: recompute it all
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def _finish(self, seq: Sequence, reason: str) -> None:
        if seq in self.running:
            self.running.remove(seq)
        self._release_blocks(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
