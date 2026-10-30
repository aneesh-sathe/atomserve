# Speculative decoding tests. The contract: a draft model may change speed, never
# results. Greedy output must equal the target model alone, token for token; with
# sampling, every token must still follow the target's distribution.

import dataclasses

import pytest
import torch

from atomserve.config import EngineConfig
from atomserve.engine import create_engine
from atomserve.kv_cache import KVPool
from atomserve.runner import ModelRunner
from atomserve.scheduler import Scheduler
from atomserve.sequence import SamplingParams, Sequence
from atomserve.speculative import SpeculativeRunner
from tests.conftest import TINY_CONFIG, random_weights

PROMPTS = [[1, 2, 3], list(range(4, 20)), [12], [13, 14, 15, 16, 17], [18, 19]]
GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


def tiny_engine(target, draft=None, k=3, **kwargs):
    config = EngineConfig(
        **{
            "block_size": 4,
            "num_blocks": 64,
            "max_model_len": 64,
            "num_speculative_tokens": k if draft else 0,
            "max_speculative_batch": 8,  # speculate even with all of PROMPTS running
            **kwargs,
        }
    )
    draft_pair = (draft, TINY_CONFIG) if draft else None
    return create_engine(target, TINY_CONFIG, config, frozenset(), draft_pair)


@pytest.fixture(scope="module")
def models():
    """A target and a different (so often wrong) draft, same vocabulary."""
    return random_weights(TINY_CONFIG, seed=0), random_weights(TINY_CONFIG, seed=1)


@pytest.mark.parametrize(
    "setting",
    [
        {},
        {"k": 1},
        {"k": 6},
        {"max_num_batched_tokens": 8, "max_num_seqs": 4},  # chunked prefill, budget limits
        {"num_blocks": 12, "max_model_len": 40},  # preemption
        {"enable_prefix_caching": False},
    ],
)
def test_greedy_speculation_equals_the_target_alone(models, setting):
    target, draft = models
    expected = tiny_engine(target).generate(PROMPTS, GREEDY)
    engine = tiny_engine(target, draft, **setting)
    assert engine.generate(PROMPTS, GREEDY) == expected
    stats = engine.stats()
    assert stats["draft_tokens"] > 0
    assert stats["kv_blocks_used"] == 0  # everything freed


def test_speculation_pauses_for_big_batches_and_the_draft_catches_up(models):
    """With max_speculative_batch=2, five requests decode plainly (the draft idles and
    falls behind); as the short ones finish, the rest speculate after the draft catches
    up. The tokens never change."""
    target, draft = models
    params = [
        SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in (3, 4, 5, 20, 24)
    ]

    def run(engine):
        handles = [engine.submit(p, sp) for p, sp in zip(PROMPTS, params, strict=True)]
        while engine.step():
            pass
        return [list(h) for h in handles]

    expected = run(tiny_engine(target))
    engine = tiny_engine(target, draft, max_speculative_batch=2)
    assert run(engine) == expected
    assert engine.stats()["draft_tokens"] > 0


def test_a_perfect_draft_is_always_accepted(models):
    """Draft == target: every proposal is kept, and each step yields k+1 tokens."""
    target, _ = models
    engine = tiny_engine(target, target, k=3)
    params = SamplingParams(temperature=0.0, max_tokens=13, ignore_eos=True)
    assert engine.generate([[1, 2, 3]], params) == tiny_engine(target).generate([[1, 2, 3]], params)
    stats = engine.stats()
    assert stats["accepted_draft_tokens"] == stats["draft_tokens"] == 9
    assert stats["steps"] == 1 + 3  # prefill (first token), then 3 steps of 4 tokens


def test_repeated_prompts_reuse_blocks_computed_by_both_models(models):
    """Prefix-cache hits in speculative mode: both pools hold the reused blocks."""
    target, draft = models
    engine = tiny_engine(target, draft)
    first = engine.generate(PROMPTS, GREEDY)
    assert engine.generate(PROMPTS, GREEDY) == first
    assert engine.stats()["cached_prompt_tokens"] > 0


def test_rejection_sampling_follows_the_target_distribution():
    """With a draft that disagrees with the target, the kept-or-fixed first token is
    still distributed exactly like the target's p, not like the draft's q."""
    config = dataclasses.replace(TINY_CONFIG, vocab_size=5)
    w = random_weights(config)
    pool = KVPool(config, 1, 4, torch.device("cpu"), torch.float32)
    runner = SpeculativeRunner(ModelRunner(w, config, pool), ModelRunner(w, config, pool))

    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    q = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.5])  # nearly the reverse
    n = 40_000
    params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0)
    seqs = [Sequence(i, [0], params) for i in range(n)]
    drafts = torch.multinomial(q, n, replacement=True)
    logits = p.log().repeat(2 * n, 1)  # two verified positions per sequence
    accepted, fixes = runner._verify(
        seqs, list(range(n)), [[int(d)] for d in drafts], [q.repeat(n, 1)], logits,
        [2 * i for i in range(n)],
    )  # fmt: skip
    first = torch.where(accepted >= 1, drafts, fixes)
    assert torch.allclose(first.bincount(minlength=5) / n, p, atol=0.01)
    acceptance = accepted.float().mean().item()
    assert acceptance == pytest.approx(torch.minimum(p, q).sum().item(), abs=0.01)


def test_lookahead_respects_limits():
    """No speculation for prompt chunks, presence penalties, or past a request's limits."""
    config = EngineConfig(
        block_size=4, num_blocks=64, max_model_len=20, num_speculative_tokens=4,
        max_num_batched_tokens=64, max_num_seqs=8, max_speculative_batch=8,
    )  # fmt: skip
    scheduler = Scheduler(config, frozenset())
    decoding = Sequence(0, [1, 2, 3], SamplingParams(max_tokens=100), num_computed_tokens=2)
    near_max_tokens = Sequence(1, [1, 2, 3], SamplingParams(max_tokens=3), num_prompt_tokens=1)
    near_max_tokens.num_computed_tokens = 2  # 2 outputs so far: room for 1 more
    near_context = Sequence(2, [1] * 18, SamplingParams(max_tokens=100), num_computed_tokens=17)
    penalised = Sequence(3, [1, 2], SamplingParams(presence_penalty=1.0), num_computed_tokens=1)
    prefilling = Sequence(4, list(range(10)))
    for seq in (decoding, near_max_tokens, near_context, penalised):
        seq.status = seq.status.RUNNING
        scheduler.running.append(seq)
    scheduler.add(prefilling)
    out = scheduler.schedule()
    assert dict(zip([s.id for s in out.seqs], out.num_lookahead, strict=True)) == {
        0: 4,
        1: 0,
        2: 1,
        3: 0,
        4: 0,
    }


@pytest.mark.slow
def test_real_draft_model_greedy_equals_target_alone(weights_fp32, config):
    """Qwen3-1.7B verifying Qwen3-0.6B drafts (fp32) gives exactly 1.7B's own greedy text."""
    from atomserve.config import ModelConfig
    from atomserve.loader import default_device, load_weights, resolve_model_dir

    target_dir = resolve_model_dir("Qwen/Qwen3-1.7B")
    target_config = ModelConfig.load(target_dir)
    target = load_weights(target_dir, target_config, default_device(), torch.float32)
    prompts = [
        [785, 6722, 315, 9625, 374],  # "The capital of France is"
        [16, 11, 220, 17, 11, 220, 18, 11],  # "1, 2, 3,"
    ]
    params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)

    def engine(draft):
        cfg = EngineConfig(
            num_blocks=128, max_model_len=2048, num_speculative_tokens=4 if draft else 0
        )
        return create_engine(target, target_config, cfg, frozenset(), draft)

    alone = engine(None).generate(prompts, params)
    spec = engine((weights_fp32, config))
    assert spec.generate(prompts, params) == alone
    stats = spec.stats()
    assert stats["accepted_draft_tokens"] / stats["draft_tokens"] > 0.3


def test_no_speculation_above_max_speculative_batch():
    config = EngineConfig(
        block_size=4, num_blocks=64, max_model_len=64, num_speculative_tokens=4,
        max_speculative_batch=2,
    )  # fmt: skip
    scheduler = Scheduler(config, frozenset())
    for n_running in (2, 3):
        scheduler.running = []
        for i in range(n_running):
            seq = Sequence(i, [1, 2, 3], SamplingParams(max_tokens=100), num_computed_tokens=2)
            seq.status = seq.status.RUNNING
            scheduler.running.append(seq)
        out = scheduler.schedule()
        assert set(out.num_lookahead) == ({4} if n_running == 2 else {0})
        for seq in out.seqs:
            scheduler.allocator.free(seq.block_table)
            seq.block_table = []
