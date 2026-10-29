# In-process engine benchmark: decode and prefill speed without HTTP or tokenization.
#
# The yardstick for optimization work. Every change should be measured with
# this before and after, on an otherwise idle machine.
#
#   decode  B sequences (distinct prompts of --context tokens) generate 128 tokens each with
#           the default thinking-mode sampling; reports aggregate tok/s and the
#           average step time.
#   prefill one cold prompt of N tokens, one output token: time to first token.
#
# `--profile` splits the average decode step into its phases: scheduling,
# building the batch on the host, launching the forward pass, waiting for the
# GPU, sampling, and bookkeeping.
#
# Usage:
#   uv run python bench/engine.py
#   uv run python bench/engine.py --batch 1,32 --profile

import argparse
import time
from collections import defaultdict

import torch

from atomserve.config import EngineConfig, ModelConfig
from atomserve.engine import Engine, create_engine
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import forward
from atomserve.sequence import THINKING_PARAMS, SamplingParams


def make_engine(weights, config) -> Engine:
    """One engine for every scenario, as in a server: a fresh 3.5 GB KV pool would make
    the first run pay for touching new GPU memory. Prefix caching is off so repeated
    runs stay cold."""
    engine_config = EngineConfig(num_blocks=2048, max_model_len=16384, enable_prefix_caching=False)
    return create_engine(weights, config, engine_config, frozenset())


def prompts(batch: int, length: int) -> list[list[int]]:
    """Distinct prompts, so the prefix cache can't make later ones cheaper."""
    return [[(1000 + 37 * b + i) % 150000 for i in range(length)] for b in range(batch)]


def bench_decode(
    engine: Engine, batch: int, new_tokens: int, context: int = 100
) -> tuple[float, float]:
    """Returns (aggregate decode tok/s, average decode step in ms)."""
    params = SamplingParams(
        **{**vars(THINKING_PARAMS), "max_tokens": new_tokens, "ignore_eos": True}
    )
    handles = [engine.submit(p, params) for p in prompts(batch, context)]
    while any(h.first_token_at is None for h in handles):  # prefill (maybe several chunks)
        engine.step()
    torch.mps.synchronize()
    start, steps = time.perf_counter(), 0
    while engine.step():
        steps += 1
    elapsed = time.perf_counter() - start
    assert all(len(list(h)) == new_tokens for h in handles)
    return batch * steps / elapsed, elapsed / steps * 1000


def bench_prefill(engine: Engine, length: int) -> float:
    """Time to first token (ms) for one cold prompt."""
    params = SamplingParams(temperature=0.0, max_tokens=1)
    start = time.perf_counter()
    engine.generate(prompts(1, length), params)
    return (time.perf_counter() - start) * 1000


def profile_decode(
    engine: Engine, batch: int, new_tokens: int, context: int = 100
) -> dict[str, float]:
    """Average ms per decode step spent in each phase (synchronizing after the forward)."""
    runner, scheduler = engine.runner, engine.scheduler
    params = SamplingParams(
        **{**vars(THINKING_PARAMS), "max_tokens": new_tokens, "ignore_eos": True}
    )
    handles = [engine.submit(p, params) for p in prompts(batch, context)]
    while any(h.first_token_at is None for h in handles):
        engine.step()
    phases: dict[str, float] = defaultdict(float)
    steps = 0
    with torch.inference_mode():
        while scheduler.has_work:
            t0 = time.perf_counter()
            out = scheduler.schedule()
            t1 = time.perf_counter()
            batch_ = runner.prepare(out.seqs, out.num_new_tokens)
            t2 = time.perf_counter()
            logits = forward(runner.weights, runner.config, batch_, runner.pool)
            t3 = time.perf_counter()
            torch.mps.synchronize()
            t4 = time.perf_counter()
            sampling = [
                s for s, n in zip(out.seqs, out.num_new_tokens, strict=True) if s.reaches_end(n)
            ]
            tokens = runner.sampler.sample(logits, sampling)
            t5 = time.perf_counter()
            scheduler.update(out, tokens)
            t6 = time.perf_counter()
            for name, dt in zip(
                ["schedule", "prepare", "forward launch", "gpu wait", "sample", "update"],
                [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5],
                strict=True,
            ):
                phases[name] += dt
            steps += 1
    return {name: total / steps * 1000 for name, total in phases.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Engine decode/prefill benchmark.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch", default="1,8,32", help="decode batch sizes")
    parser.add_argument("--prefill", default="1000,8000", help="prefill prompt lengths")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=100, help="prompt length for decode runs")
    parser.add_argument("--profile", action="store_true", help="per-phase decode step breakdown")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    config = ModelConfig.load(model_dir)
    weights = load_weights(model_dir, config, default_device(), torch.bfloat16)
    engine = make_engine(weights, config)
    bench_decode(engine, 2, 8)  # warm-up: compile MPS kernels, touch the KV pool
    for length in (600, 8000):
        bench_prefill(engine, length)

    for batch in [int(b) for b in args.batch.split(",")]:
        rate, step_ms = bench_decode(engine, batch, args.new_tokens, args.context)
        print(f"decode  B={batch:<3} {rate:7.1f} tok/s   {step_ms:6.2f} ms/step", flush=True)
        if args.profile:
            phases = profile_decode(engine, batch, args.new_tokens, args.context)
            print("        " + "  ".join(f"{k} {v:.2f}" for k, v in phases.items()), flush=True)
    for length in [int(n) for n in args.prefill.split(",") if n]:
        print(f"prefill {length:>5} tokens: TTFT {bench_prefill(engine, length):7.0f} ms")


if __name__ == "__main__":
    main()
