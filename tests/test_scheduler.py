# Scheduler tests with a fake runner: pure Python, no model, no GPU.
#
# The fake runner simulates the KV pool slot by slot. Each step it "writes" the
# scheduled tokens into the slots their block tables point to, and it predicts
# the next token from what it *reads back* from the slots, not from the
# sequence's own token list. So if the scheduler ever hands a block to two
# sequences wrongly, forgets to recompute after preemption, reuses a cached
# prefix that doesn't match, or mis-sizes a block table, the outputs change
# and the tests fail, just as real K/V would.

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


def make(**config) -> tuple[Scheduler, FakeRunner]:
    cfg = {"block_size": 4, "num_blocks": 64, "max_model_len": 64, **config}
    return Scheduler(EngineConfig(**cfg), frozenset({EOS})), FakeRunner(cfg["block_size"])


def drive(scheduler: Scheduler, runner: FakeRunner) -> int:
    """Step until no work is left, checking invariants every step; returns preemptions."""
    preemptions = 0
    while scheduler.has_work:
        out = scheduler.schedule()
        assert out.seqs, "no progress"
        assert len(out.seqs) <= scheduler.config.max_num_seqs
        assert_sharing_is_safe(scheduler)
        preemptions += len(out.preempted)
        scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    assert scheduler.allocator.num_free == scheduler.allocator.num_blocks, "leaked blocks"
    return preemptions


def run(
    prompts: list[list[int]], params=GREEDY, **config
) -> tuple[list[list[int]], Scheduler, int]:
    """Run prompts arriving together; return outputs, the scheduler and the preemption count."""
    scheduler, runner = make(**config)
    seqs = [Sequence(i, list(p), params) for i, p in enumerate(prompts)]
    for seq in seqs:
        scheduler.add(seq)
    preemptions = drive(scheduler, runner)
    return [s.output_ids for s in seqs], scheduler, preemptions


def assert_sharing_is_safe(scheduler: Scheduler) -> None:
    """A block may be shared only as the same, fully computed, logical block of each owner."""
    bs = scheduler.config.block_size
    owners: dict[int, list[tuple[Sequence, int]]] = {}
    for seq in scheduler.running:
        for i, block in enumerate(seq.block_table):
            owners.setdefault(block, []).append((seq, i))
    for sharers in owners.values():
        if len(sharers) > 1:
            assert len({i for _, i in sharers}) == 1, "shared at different positions"
            assert all((i + 1) * bs <= s.num_computed_tokens for s, i in sharers), (
                "shared too early"
            )


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


def test_long_prompt_is_prefilled_in_chunks_alongside_decodes():
    """A 20-token prompt with an 8-token budget runs as chunks of 7, 7, 6 (one budget
    token goes to the running decode each step), and the decode never misses a step."""
    cfg = EngineConfig(
        block_size=4, num_blocks=64, max_model_len=64, max_num_batched_tokens=8, max_num_seqs=4
    )
    scheduler, runner = Scheduler(cfg, frozenset()), FakeRunner(4)
    short, long = Sequence(0, [2, 3], GREEDY), Sequence(1, [1] * 20, GREEDY)
    scheduler.add(short)
    out = scheduler.schedule()
    scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    scheduler.add(long)
    plans = []
    while long.num_computed_tokens < long.num_prompt_tokens:
        out = scheduler.schedule()
        plans.append(list(zip(out.seqs, out.num_new_tokens, strict=True)))
        scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    assert plans == [[(short, 1), (long, 7)], [(short, 1), (long, 7)], [(short, 1), (long, 6)]]
    assert len(long.output_ids) == 1  # sampled only once its last chunk ran


def test_chunked_prefill_does_not_change_outputs():
    prompts = [[1] * 20, [2, 3], list(range(30, 47))]
    outputs, _, _ = run(prompts, max_num_batched_tokens=8, max_num_seqs=4)
    assert outputs == [alone(p) for p in prompts]


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


def test_repeated_prompt_reuses_its_cached_blocks():
    """The second run of a 10-token prompt starts with its 2 full blocks computed."""
    scheduler, runner = make()
    prompt = list(range(40, 50))
    first = Sequence(0, list(prompt), GREEDY)
    scheduler.add(first)
    drive(scheduler, runner)
    again = Sequence(1, list(prompt), GREEDY)
    scheduler.add(again)
    drive(scheduler, runner)
    assert (first.num_cached_tokens, again.num_cached_tokens) == (0, 8)
    assert again.output_ids == first.output_ids


def test_fully_cached_prompt_still_computes_its_last_token():
    """An 8-token prompt is exactly 2 blocks; only the first is reused so the last token
    can run and produce logits."""
    scheduler, runner = make()
    for i in range(2):
        seq = Sequence(i, list(range(8)), GREEDY)
        scheduler.add(seq)
        drive(scheduler, runner)
    assert seq.num_cached_tokens == 4


def test_running_requests_share_a_common_prefix():
    """A request arriving while another with the same 12-token system prompt is still
    decoding shares its 3 blocks, and both outputs equal running alone."""
    system = list(range(100, 112))
    a, b = Sequence(0, system + [1, 2], GREEDY), Sequence(1, system + [3], GREEDY)
    scheduler, runner = make()
    scheduler.add(a)
    for _ in range(2):  # prefill a, then one decode step
        out = scheduler.schedule()
        scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    scheduler.add(b)
    out = scheduler.schedule()
    assert b.block_table[:3] == a.block_table[:3]  # shared, refcount 2
    assert out.num_new_tokens == [1, 1]  # b only computes its 1 uncached token
    scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    drive(scheduler, runner)
    assert a.output_ids == alone(a.token_ids[: a.num_prompt_tokens])
    assert b.output_ids == alone(b.token_ids[: b.num_prompt_tokens])


def test_prefix_caching_under_memory_pressure_keeps_outputs_and_frees_everything():
    """Shared prefixes, preemption and eviction all at once: outputs are unchanged."""
    system = [7, 7, 7, 7, 8, 8, 8, 8]
    prompts = [system + [i, i + 1] for i in range(6)]
    outputs, _, preemptions = run(prompts, num_blocks=10, max_model_len=40)
    assert preemptions > 0
    assert outputs == [alone(p) for p in prompts]


def test_prefix_caching_can_be_disabled():
    scheduler, runner = make(enable_prefix_caching=False)
    for i in range(2):
        seq = Sequence(i, list(range(10)), GREEDY)
        scheduler.add(seq)
        drive(scheduler, runner)
    assert seq.num_cached_tokens == 0
    assert scheduler.allocator.num_cached == 0


def test_only_fully_computed_blocks_enter_the_prefix_cache():
    """A is aborted after prefilling 14 of its 20 tokens: blocks 0-2 are complete, block 3
    holds K/V for only 2 of its 4 tokens. B, with the same prompt, may reuse the
    complete blocks and nothing more."""
    prompt = list(range(60, 80))
    scheduler, runner = make(max_num_batched_tokens=14, max_num_seqs=4)
    a = Sequence(0, list(prompt), GREEDY)
    scheduler.add(a)
    out = scheduler.schedule()
    scheduler.update(out, runner.execute(out.seqs, out.num_new_tokens))
    assert a.num_computed_tokens == 14
    scheduler.abort(a.id)
    b = Sequence(1, list(prompt), GREEDY)
    scheduler.add(b)
    drive(scheduler, runner)
    assert b.num_cached_tokens == 12
    assert b.output_ids == alone(prompt)
