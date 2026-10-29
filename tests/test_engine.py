# Engine tests: the whole stack (engine -> scheduler -> runner -> model) on the
# tiny random model on CPU, plus the batching invariant on the real model.

import threading

import pytest

from atomserve.config import EngineConfig
from atomserve.engine import create_engine
from atomserve.sequence import SamplingParams

PROMPTS = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11], [12], [13, 14, 15, 16, 17], [18, 19]]
GREEDY = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True)


def tiny_engine(tiny_weights, tiny_config, **kwargs):
    config = EngineConfig(**{"block_size": 4, "num_blocks": 64, "max_model_len": 64, **kwargs})
    return create_engine(tiny_weights, tiny_config, config, frozenset())


def test_batched_generation_matches_each_prompt_alone(tiny_weights, tiny_config):
    alone = [tiny_engine(tiny_weights, tiny_config).generate([p], GREEDY)[0] for p in PROMPTS]
    together = tiny_engine(tiny_weights, tiny_config).generate(PROMPTS, GREEDY)
    assert together == alone


def test_preemption_under_memory_pressure_does_not_change_outputs(tiny_weights, tiny_config):
    alone = [tiny_engine(tiny_weights, tiny_config).generate([p], GREEDY)[0] for p in PROMPTS]
    engine = tiny_engine(tiny_weights, tiny_config, num_blocks=8, max_model_len=32)
    assert engine.generate(PROMPTS, GREEDY) == alone


def test_chunked_prefill_matches_whole_prompt_prefill(tiny_weights, tiny_config):
    """With a 4-token budget every prompt is split into chunks; the outputs don't change."""
    alone = [tiny_engine(tiny_weights, tiny_config).generate([p], GREEDY)[0] for p in PROMPTS]
    engine = tiny_engine(tiny_weights, tiny_config, max_num_batched_tokens=4, max_num_seqs=4)
    assert engine.generate(PROMPTS, GREEDY) == alone


def test_engine_thread_serves_concurrent_clients(tiny_weights, tiny_config):
    """Requests submitted from several threads stream back the same tokens as alone."""
    alone = [tiny_engine(tiny_weights, tiny_config).generate([p], GREEDY)[0] for p in PROMPTS]
    engine = tiny_engine(tiny_weights, tiny_config)
    engine.start()
    results: dict[int, list[int]] = {}

    def client(i: int) -> None:
        handle = engine.submit(PROMPTS[i], GREEDY)
        results[i] = list(handle)
        assert handle.finish_reason == "length"
        assert handle.ttft is not None and handle.ttft >= 0

    threads = [threading.Thread(target=client, args=(i,)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    engine.shutdown()
    assert [results[i] for i in range(len(PROMPTS))] == alone


def test_abort_stops_a_stream_and_frees_its_blocks(tiny_weights, tiny_config):
    engine = tiny_engine(tiny_weights, tiny_config)
    engine.start()
    handle = engine.submit(
        [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=50, ignore_eos=True)
    )
    tokens = []
    for token in handle:
        tokens.append(token)
        if len(tokens) == 3:
            handle.abort()
    engine.shutdown()
    assert handle.finish_reason == "abort"
    assert 3 <= len(tokens) < 50
    assert engine.stats()["kv_blocks_used"] == 0


def test_submit_validates_prompt_length(tiny_weights, tiny_config):
    engine = tiny_engine(tiny_weights, tiny_config)
    with pytest.raises(ValueError, match="prompt"):
        engine.submit([], GREEDY)
    with pytest.raises(ValueError, match="prompt"):
        engine.submit([1] * 64, GREEDY)


PROMPTS_REAL = [
    [785, 6722, 315, 9625, 374],  # "The capital of France is"
    [9707, 11, 847, 829, 374],  # "Hello, my name is"
    [16, 11, 220, 17, 11, 220, 18, 11],  # "1, 2, 3,"
]


@pytest.mark.slow
def test_chunked_prefill_matches_on_real_model(weights_fp32, config, greedy):
    """A ~500-token prompt prefilled in 64-token chunks gives the same greedy tokens."""
    prompt = (PROMPTS_REAL[0] + PROMPTS_REAL[1]) * 50
    [whole] = greedy(weights_fp32, config, [prompt], 16)
    [chunked] = greedy(weights_fp32, config, [prompt], 16, max_num_batched_tokens=64)
    assert chunked == whole


@pytest.mark.slow
def test_batched_greedy_matches_alone_on_real_model(weights_fp32, config, greedy):
    """Three prompts batched produce the same 24 greedy tokens as each run on its own."""
    alone = [greedy(weights_fp32, config, [p], 24)[0] for p in PROMPTS_REAL]
    assert greedy(weights_fp32, config, PROMPTS_REAL, 24) == alone
