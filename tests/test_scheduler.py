# Scheduler tests with a fake runner: pure Python, no model, no GPU.
#
# The fake runner simulates the KV pool slot by slot. Each step it "writes" the
# scheduled tokens into the slots their block tables point to, and it predicts
# the next token from what it *reads back* from the slots, not from the
# sequence's own token list. So if the scheduler ever hands a block to two
# sequences, forgets to recompute after preemption, or mis-sizes a block
# table, the outputs change and the tests fail, just as real K/V would.

from atomserve.config import EngineConfig
from atomserve.runner import slot_of
from atomserve.scheduler import Scheduler
from atomserve.sequence import SamplingParams, Sequence, SequenceStatus

EOS = 999
GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


class FakeRunner:
    """A deterministic 'model' whose next token depends on every token in its context."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.slots: dict[int, int] = {}  # slot -> token stored there

    def execute(self, seqs: list[Sequence], num_new_tokens: list[int]) -> list[int]:
        sampled = []
        for seq, n in zip(seqs, num_new_tokens, strict=True):
            start, end = seq.num_computed_tokens, seq.num_computed_tokens + n
            assert end <= len(seq.block_table) * self.block_size, "block table too small"
            for p in range(start, end):
                self.slots[slot_of(seq.block_table, p, self.block_size)] = seq.token_ids[p]
            if seq.reaches_end(n):
                context = [
                    self.slots[slot_of(seq.block_table, p, self.block_size)] for p in range(end)
                ]
                sampled.append(next_token(context))
        return sampled


def next_token(context: list[int]) -> int:
    """Any function of the whole context works; this one mixes in position too."""
    return sum((i + 1) * t for i, t in enumerate(context)) % 500


def run(
    prompts: list[list[int]], params=GREEDY, **config
) -> tuple[list[list[int]], Scheduler, int]:
    """Drive the scheduler to completion; return outputs, scheduler and preemption count."""
    cfg = {"block_size": 4, "num_blocks": 64, "max_model_len": 64, **config}
    scheduler = Scheduler(EngineConfig(**cfg), frozenset({EOS}))
    runner = FakeRunner(cfg["block_size"])
    seqs = [Sequence(i, list(p), params) for i, p in enumerate(prompts)]
    for seq in seqs:
        scheduler.add(seq)
    preemptions = 0
    while scheduler.has_work:
        out = scheduler.schedule()
        assert out.seqs, "no progress"
        assert len(out.seqs) <= scheduler.config.max_num_seqs
        assert_blocks_exclusive(scheduler)
        preemptions += len(out.preempted)
        scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    assert scheduler.allocator.num_free == scheduler.allocator.num_blocks, "leaked blocks"
    return [s.output_ids for s in seqs], scheduler, preemptions


def assert_blocks_exclusive(scheduler: Scheduler) -> None:
    owned = [b for s in scheduler.running for b in s.block_table]
    assert len(owned) == len(set(owned)), "two sequences share a block"


PROMPTS = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11], [12, 13, 14, 15, 16], [17, 18]]


def alone(prompt: list[int]) -> list[int]:
    return run([prompt])[0][0]


def test_continuous_batching_matches_running_each_request_alone():
    outputs, _, preemptions = run(PROMPTS)
    assert outputs == [alone(p) for p in PROMPTS]
    assert preemptions == 0
    assert all(len(o) == 12 for o in outputs)


def test_max_num_seqs_queues_the_rest():
    outputs, _, _ = run(PROMPTS, max_num_seqs=2, max_num_batched_tokens=16)
    assert outputs == [alone(p) for p in PROMPTS]


def test_block_exhaustion_preempts_and_recomputes_without_changing_outputs():
    """8 blocks of 4 tokens hold 32 tokens; 5 sequences growing to ~20 tokens cannot all fit."""
    outputs, _, preemptions = run(PROMPTS, num_blocks=8, max_model_len=32)
    assert preemptions > 0
    assert outputs == [alone(p) for p in PROMPTS]


def test_preemption_evicts_the_newest_and_requeues_it_first():
    scheduler = Scheduler(EngineConfig(block_size=4, num_blocks=2, max_model_len=8), frozenset())
    old, new = Sequence(0, [1, 2, 3], GREEDY), Sequence(1, [4, 5, 6], GREEDY)
    runner = FakeRunner(4)
    for seq in (old, new):
        scheduler.add(seq)
    for _ in range(2):  # prefill 3 tokens, then decode the 4th: one full block each
        out = scheduler.schedule()
        scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    out = scheduler.schedule()  # both now need a second block; only one can grow
    assert out.preempted == [new]
    assert out.seqs == [old]
    assert scheduler.waiting[0] is new
    assert new.status is SequenceStatus.WAITING
    assert new.num_computed_tokens == 0 and new.block_table == []


def test_admission_respects_the_token_budget():
    scheduler = Scheduler(
        EngineConfig(
            block_size=4, num_blocks=64, max_model_len=64, max_num_batched_tokens=8, max_num_seqs=4
        ),
        frozenset(),
    )
    for i, prompt in enumerate([[1] * 5, [2] * 3, [3] * 2]):
        scheduler.add(Sequence(i, prompt, GREEDY))
    out = scheduler.schedule()
    assert out.num_new_tokens == [5, 3]  # the 2-token prompt would exceed 8
    assert len(scheduler.waiting) == 1


def test_a_prompt_larger_than_the_budget_runs_alone():
    outputs, _, _ = run([[1] * 20, [2, 3]], max_num_batched_tokens=8, max_num_seqs=4)
    assert outputs == [alone([1] * 20), alone([2, 3])]


def test_finish_reasons_and_eos():
    """EOS stops a request unless ignore_eos; max_tokens and max_model_len give 'length'."""
    scheduler = Scheduler(
        EngineConfig(block_size=4, num_blocks=16, max_model_len=16), frozenset({7})
    )
    stops = Sequence(0, [1], SamplingParams(temperature=0.0, max_tokens=5))
    ignores = Sequence(1, [1], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True))
    too_long = Sequence(
        2, [1] * 14, SamplingParams(temperature=0.0, max_tokens=99, ignore_eos=True)
    )
    for seq in (stops, ignores, too_long):
        scheduler.add(seq)
    while scheduler.has_work:
        out = scheduler.schedule()
        scheduler.update(
            out,
            [7] * sum(s.reaches_end(n) for s, n in zip(out.seqs, out.num_new_tokens, strict=True)),
        )
    assert (stops.finish_reason, stops.output_ids) == ("stop", [7])
    assert (ignores.finish_reason, ignores.output_ids) == ("length", [7, 7, 7])
    assert (too_long.finish_reason, too_long.num_tokens) == ("length", 16)
    assert scheduler.allocator.num_free == 16


def test_abort_frees_running_and_waiting_requests():
    scheduler = Scheduler(
        EngineConfig(
            block_size=4, num_blocks=16, max_model_len=16, max_num_seqs=1, max_num_batched_tokens=8
        ),
        frozenset(),
    )
    running, waiting = Sequence(0, [1, 2], GREEDY), Sequence(1, [3], GREEDY)
    scheduler.add(running)
    scheduler.add(waiting)
    scheduler.schedule()
    assert running.status is SequenceStatus.RUNNING and running.block_table
    assert scheduler.abort(0) is running and scheduler.abort(1) is waiting
    assert scheduler.abort(1) is None  # already gone
    assert running.finish_reason == waiting.finish_reason == "abort"
    assert not scheduler.has_work
    assert scheduler.allocator.num_free == 16
