# The engine: one background thread that owns the GPU and runs the step loop.
#
#     loop:
#         take new requests / aborts from the inbox -> scheduler
#         plan    = scheduler.schedule()        what runs this step (pure Python)
#         results = runner.execute(plan)        forward pass(es) + sampling (GPU)
#         scheduler.update(plan, results)       advance sequences, retire finished ones
#         push each new token to its request's handle
#
# Clients (the REPL, HTTP handler threads) never touch the scheduler or the
# GPU. They talk to the engine through two kinds of thread-safe queues: one
# inbox for submit/abort, and one outbox per request that streams its tokens
# back. The engine deals only in token ids; text belongs to the clients.

import itertools
import queue
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass

from atomserve.config import EngineConfig, ModelConfig
from atomserve.kv_cache import KVPool
from atomserve.model import Weights
from atomserve.runner import ModelRunner
from atomserve.scheduler import Scheduler
from atomserve.sequence import SamplingParams, Sequence, SequenceStatus
from atomserve.speculative import SpeculativeRunner


@dataclass
class _Finished:
    """Last event on a request's outbox."""

    reason: str  # "stop", "length", "abort" or "error"
    error: BaseException | None = None


class RequestHandle:
    """The client's end of one request: iterate it to receive token ids as they are made.

    After iteration ends, `finish_reason` says why, and the timing fields give
    the request's latency: time to first token (TTFT) and decode speed.
    """

    def __init__(self, request_id: int, engine: Engine, num_prompt_tokens: int):
        self.id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = 0
        self.num_cached_tokens = 0  # prompt tokens served from the prefix cache
        self.finish_reason: str | None = None
        self.submitted_at = time.perf_counter()
        self.first_token_at: float | None = None
        self.finished_at: float | None = None
        self._engine = engine
        self._outbox: queue.SimpleQueue[int | _Finished] = queue.SimpleQueue()

    def __iter__(self) -> Iterator[int]:
        """Yield token ids until the request finishes. Raises if the engine failed."""
        while True:
            event = self._outbox.get()
            if isinstance(event, _Finished):
                self.finish_reason = event.reason
                if event.error is not None:
                    raise RuntimeError("engine step failed") from event.error
                return
            yield event

    def abort(self) -> None:
        """Stop generating (e.g. the user pressed Ctrl-C or the HTTP client went away)."""
        self._engine.abort(self.id)

    @property
    def ttft(self) -> float | None:
        """Seconds from submit to the first token."""
        return None if self.first_token_at is None else self.first_token_at - self.submitted_at

    @property
    def decode_tokens_per_s(self) -> float | None:
        """Generation speed after the first token."""
        if self.first_token_at is None or self.finished_at is None or self.num_output_tokens < 2:
            return None
        return (self.num_output_tokens - 1) / max(self.finished_at - self.first_token_at, 1e-9)


class Engine:
    """Accepts requests from any thread; runs them on its own thread (or via `step`)."""

    def __init__(self, runner: ModelRunner | SpeculativeRunner, scheduler: Scheduler):
        self.runner = runner
        self.scheduler = scheduler
        self._inbox: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        self._handles: dict[int, RequestHandle] = {}
        self._ids = itertools.count()
        self._thread: threading.Thread | None = None
        self._stopping = False
        # Counters for /stats; written by the engine thread, read by anyone.
        self.num_steps = 0
        self.last_batch_seqs = 0
        self.last_batch_tokens = 0
        self.num_prompt_tokens = 0
        self.num_cached_prompt_tokens = 0  # of finished requests: served by the prefix cache
        self.num_generated_tokens = 0

    # ---- client side (any thread) -------------------------------------------------

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> RequestHandle:
        """Queue a request; returns its handle immediately."""
        limit = self.scheduler.config.max_model_len
        if not 0 < len(prompt_ids) < limit:
            raise ValueError(f"prompt has {len(prompt_ids)} tokens; must be 1 to {limit - 1}")
        if params.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        request_id = next(self._ids)
        handle = RequestHandle(request_id, self, len(prompt_ids))
        self._inbox.put(("add", Sequence(request_id, list(prompt_ids), params), handle))
        return handle

    def abort(self, request_id: int) -> None:
        """Cancel a request. Safe to call for one that already finished."""
        self._inbox.put(("abort", request_id))

    def start(self) -> None:
        """Run the step loop on a daemon thread."""
        self._thread = threading.Thread(target=self._loop, name="atomserve-engine", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the engine thread after its current step."""
        self._stopping = True
        self._inbox.put(("wake",))
        if self._thread is not None:
            self._thread.join()

    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[list[int]]:
        """Run prompts to completion on the calling thread; return each one's output ids.

        For tests, benchmarks and warm-up, when the engine thread is not running.
        All prompts are submitted first, so they are batched together.
        """
        handles = [self.submit(p, params) for p in prompts]
        while self.step():
            pass
        return [list(h) for h in handles]

    def stats(self) -> dict:
        """A snapshot of engine counters (approximate if read while a step runs)."""
        s = self.scheduler
        used = s.allocator.num_blocks - s.allocator.num_free
        return {
            "running": len(s.running),
            "waiting": len(s.waiting),
            "kv_blocks_used": used,
            "kv_blocks_total": s.allocator.num_blocks,
            "kv_usage": used / s.allocator.num_blocks,
            "kv_blocks_cached": s.allocator.num_cached,
            "steps": self.num_steps,
            "last_batch_seqs": self.last_batch_seqs,
            "last_batch_tokens": self.last_batch_tokens,
            "prompt_tokens": self.num_prompt_tokens,
            "cached_prompt_tokens": self.num_cached_prompt_tokens,
            "generated_tokens": self.num_generated_tokens,
            "draft_tokens": getattr(self.runner, "num_drafted", 0),
            "accepted_draft_tokens": getattr(self.runner, "num_accepted", 0),
        }

    # ---- engine side (engine thread only) -------------------------------------------

    def step(self) -> bool:
        """Run one scheduling step. Returns False when there is nothing to do."""
        self._drain_inbox(block=False)
        if not self.scheduler.has_work:
            return False
        out = self.scheduler.schedule()
        if not out.seqs:
            raise RuntimeError("scheduler has work but scheduled nothing")
        results = self.runner.execute(out)
        new_tokens = self.scheduler.update(out, results)

        now = time.perf_counter()
        self.num_steps += 1
        self.last_batch_seqs, self.last_batch_tokens = len(out.seqs), out.num_tokens
        for seq, tokens in new_tokens:
            handle = self._handles[seq.id]
            if handle.first_token_at is None:
                handle.first_token_at = now
            handle.num_output_tokens += len(tokens)
            self.num_generated_tokens += len(tokens)
            for token in tokens:
                handle._outbox.put(token)
            if seq.status is SequenceStatus.FINISHED:
                self._close(seq)
        return True

    def _drain_inbox(self, block: bool) -> None:
        """Move queued submits/aborts into the scheduler. `block` waits for the first one."""
        while True:
            try:
                message = self._inbox.get(block=block)
            except queue.Empty:
                return
            block = False
            if message[0] == "add":
                _, seq, handle = message
                self._handles[seq.id] = handle
                self.num_prompt_tokens += seq.num_prompt_tokens
                self.scheduler.add(seq)
            elif message[0] == "abort":
                if (seq := self.scheduler.abort(message[1])) is not None:
                    self._close(seq)

    def _close(self, seq: Sequence, error: BaseException | None = None) -> None:
        """Deliver a finished sequence's final event and drop its per-request state."""
        handle = self._handles.pop(seq.id)
        handle.num_cached_tokens = seq.num_cached_tokens
        self.num_cached_prompt_tokens += seq.num_cached_tokens
        handle.finished_at = time.perf_counter()
        self.runner.sampler.release(seq.id)
        handle._outbox.put(_Finished(seq.finish_reason or "error", error))

    def _loop(self) -> None:
        """Engine thread body: step while there is work, sleep on the inbox when idle."""
        while not self._stopping:
            try:
                if not self.step():
                    self._drain_inbox(block=True)
            except Exception as error:  # fail the requests, keep the engine alive
                traceback.print_exc(file=sys.stderr)
                for seq in [*self.scheduler.running, *self.scheduler.waiting]:
                    self.scheduler.abort(seq.id)
                    seq.finish_reason = "error"
                    self._close(seq, error)


def create_engine(
    weights: Weights,
    model_config: ModelConfig,
    engine_config: EngineConfig,
    eos_token_ids: frozenset[int],
    draft: tuple[Weights, ModelConfig] | None = None,
) -> Engine:
    """Wire up KV pool(s), runner and scheduler around loaded weights (thread not started).

    With `draft` (a smaller model of the same family) and
    engine_config.num_speculative_tokens > 0, decoding is speculative: the draft gets
    its own KV pool with the same number of blocks, indexed by the same block tables.
    """
    if (draft is not None) != (engine_config.num_speculative_tokens > 0):
        raise ValueError("speculative decoding needs both a draft model and num_speculative_tokens")

    def runner_for(w: Weights, config: ModelConfig) -> ModelRunner:
        device, dtype = w.embed.device, w.embed.dtype
        pool = KVPool(config, engine_config.num_blocks, engine_config.block_size, device, dtype)
        return ModelRunner(w, config, pool)

    runner = runner_for(weights, model_config)
    if draft is not None:
        runner = SpeculativeRunner(runner, runner_for(*draft))
    return Engine(runner, Scheduler(engine_config, eos_token_ids))
