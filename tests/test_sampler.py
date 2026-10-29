# Sampler tests on hand-made logits (CPU). Each pins down one rule of the pipeline:
# greedy, top-k, top-p, the draw's distribution, seeding and the presence penalty.

import pytest
import torch

from atomserve.sampler import Sampler
from atomserve.sequence import SamplingParams, Sequence

CPU = torch.device("cpu")
PROBS = torch.tensor([0.4, 0.3, 0.2, 0.1])  # token 0 most likely


def seqs(n: int, output_ids: list[int] | None = None, **params) -> list[Sequence]:
    """n sequences with a 1-token prompt, the given generated tokens and sampling params."""
    tokens = [0, *(output_ids or [])]
    return [
        Sequence(i, list(tokens), SamplingParams(**params), num_prompt_tokens=1) for i in range(n)
    ]


def draws(n: int, **params) -> torch.Tensor:
    """Sample n times from PROBS (one batch of n identical rows)."""
    logits = PROBS.log().repeat(n, 1)
    return torch.tensor(Sampler(CPU).sample(logits, seqs(n, **params)))


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 0.2], [5.0, 1.0, 0.0]])
    assert Sampler(CPU).sample(logits, seqs(2, temperature=0.0)) == [1, 0]


def test_draws_follow_the_distribution():
    """With no truncation, token frequencies converge to the softmax probabilities."""
    counts = draws(20_000, temperature=1.0, top_k=0, top_p=1.0).bincount(minlength=4)
    assert torch.allclose(counts / 20_000, PROBS, atol=0.015)


def test_top_k_limits_support():
    assert set(draws(2_000, temperature=1.0, top_k=2, top_p=1.0).tolist()) == {0, 1}
    assert set(draws(200, temperature=1.0, top_k=1, top_p=1.0).tolist()) == {0}


def test_top_p_keeps_the_smallest_set_reaching_p():
    """p=0.6: token 0 (0.4) does not reach it, so token 1 joins; then 0.7 >= 0.6 stops."""
    assert set(draws(2_000, temperature=1.0, top_k=0, top_p=0.6).tolist()) == {0, 1}
    assert set(draws(200, temperature=1.0, top_k=0, top_p=0.01).tolist()) == {0}


def test_seeded_requests_repeat_regardless_of_batch_neighbours():
    """Same seed, same draws, whether the request runs alone or next to others."""
    logits = torch.randn(1, 1000)
    alone = Sampler(CPU)
    crowd = Sampler(CPU)
    seeded = seqs(1, temperature=1.0, top_k=50, top_p=1.0, seed=7)[0]
    other = seqs(2, temperature=0.8, top_k=500, top_p=0.9)
    other[0].id, other[1].id = 10, 11
    a = [alone.sample(logits, [seeded])[0] for _ in range(20)]
    b = [crowd.sample(logits.repeat(3, 1), [other[0], seeded, other[1]])[1] for _ in range(20)]
    assert a == b
    assert len(set(a)) > 1  # actually sampling, not stuck on one token


def test_presence_penalty_discourages_generated_tokens():
    """Token 0 is barely ahead; once generated and penalised, greedy picks token 1."""
    logits = torch.tensor([[2.0, 1.9, 0.0]])
    plain = seqs(1, output_ids=[0], temperature=0.0)
    penalised = seqs(1, output_ids=[0], temperature=0.0, presence_penalty=0.5)
    assert Sampler(CPU).sample(logits, plain) == [0]
    assert Sampler(CPU).sample(logits, penalised) == [1]


@pytest.mark.parametrize("seed", [None, 3])
def test_greedy_and_sampled_rows_mix_in_one_batch(seed):
    logits = torch.tensor([[0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    batch = seqs(1, temperature=0.0) + seqs(1, temperature=1.0, top_k=1, seed=seed)
    batch[1].id = 1
    assert Sampler(CPU).sample(logits, batch) == [1, 2]


def test_a_step_with_nothing_to_sample():
    """A step that only advanced prompt chunks hands the sampler zero rows."""
    assert Sampler(CPU).sample(torch.empty(0, 10), []) == []
