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
# `--chat` decodes from real chat prompts instead of synthetic token ids; use it
# with `--draft-model` (speculative decoding), whose speed depends on how often
# the draft guesses the text right, which needs real text.
#
# Usage:
#   uv run python bench/engine.py
#   uv run python bench/engine.py --batch 1,32 --profile
#   uv run python bench/engine.py --model Qwen/Qwen3-4B --chat --draft-model Qwen/Qwen3-0.6B

import argparse
import dataclasses
import time
from collections import defaultdict

import torch

from atomserve.config import EngineConfig, ModelConfig
from atomserve.engine import Engine, create_engine
from atomserve.kv_cache import KVPool
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import forward
from atomserve.runner import rows_for
from atomserve.sequence import THINKING_PARAMS, SamplingParams, StepResult
from atomserve.tokenizer import ChatTokenizer

CHAT_QUESTIONS = [
    "Explain how a hash map works.",
    "Write a short poem about the sea.",
    "What causes the seasons on Earth?",
    "Give three tips for writing clean Python code.",
    "Summarize the plot of Romeo and Juliet.",
    "How does a bicycle stay upright?",
    "What is the difference between TCP and UDP?",
    "Describe how bread is made.",
]


def make_engine(
    weights, config, draft=None, num_speculative_tokens: int = 0, kv_mem_gb: float = 4.0
) -> Engine:
    """One engine for every scenario, as in a server: a fresh KV pool would make the
    first run pay for touching new GPU memory. Prefix caching is off so repeated runs
    stay cold. The pools get kv_mem_gb in total (shared with the draft model's), like
    the CLI: more would push big models past the GPU's working-set limit."""
    configs = [config] + ([draft[1]] if draft else [])
    block_bytes = sum(KVPool.bytes_per_block(c, 16, torch.bfloat16) for c in configs)
    num_blocks = int(kv_mem_gb * 2**30 // block_bytes)
    engine_config = EngineConfig(
        num_blocks=num_blocks,
        max_model_len=min(16384, num_blocks * 16),
        enable_prefix_caching=False,
        num_speculative_tokens=num_speculative_tokens,
    )
    return create_engine(weights, config, engine_config, frozenset(), draft)


def chat_prompts(model_dir, batch: int) -> list[list[int]]:
    """Real chat prompts (thinking off), cycling through CHAT_QUESTIONS."""
    tokenizer = ChatTokenizer(model_dir)
    return [
        tokenizer.encode_chat(
            [{"role": "user", "content": CHAT_QUESTIONS[b % len(CHAT_QUESTIONS)]}],
            enable_thinking=False,
        )
        for b in range(batch)
    ]


def prompts(batch: int, length: int) -> list[list[int]]:
    """Distinct prompts, so the prefix cache can't make later ones cheaper."""
    return [[(1000 + 37 * b + i) % 150000 for i in range(length)] for b in range(batch)]


def bench_decode(
    engine: Engine,
    batch: int,
    new_tokens: int,
    context: int = 100,
    prompt_ids: list[list[int]] | None = None,
    greedy: bool = False,
) -> tuple[float, float]:
    """Returns (aggregate decode tok/s, average decode step in ms)."""
    base = SamplingParams(temperature=0.0) if greedy else THINKING_PARAMS
    params = dataclasses.replace(base, max_tokens=new_tokens, ignore_eos=True)
    batch_prompts = prompt_ids or prompts(batch, context)
    handles = [engine.submit(p, params) for p in batch_prompts]
    while any(h.first_token_at is None for h in handles):  # prefill (maybe several chunks)
        engine.step()
    torch.mps.synchronize()
    start, steps = time.perf_counter(), 0
    tokens_before = engine.num_generated_tokens
    while engine.step():
        steps += 1
    elapsed = time.perf_counter() - start
    assert all(len(list(h)) == new_tokens for h in handles)
    # Count tokens, not steps: with speculative decoding a step yields several.
    return (engine.num_generated_tokens - tokens_before) / elapsed, elapsed / steps * 1000


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
            batch_ = runner.prepare(rows_for(out.seqs, out.num_new_tokens))
            t2 = time.perf_counter()
            logits = forward(runner.weights, runner.config, batch_, runner.pool)
            t3 = time.perf_counter()
            torch.mps.synchronize()
            t4 = time.perf_counter()
            sampling = [
                s for s, n in zip(out.seqs, out.num_new_tokens, strict=True) if s.reaches_end(n)
            ]
            sampled = iter(runner.sampler.sample(logits, sampling))
            t5 = time.perf_counter()
            results = [
                StepResult([next(sampled)] if s.reaches_end(n) else [], s.num_computed_tokens + n)
                for s, n in zip(out.seqs, out.num_new_tokens, strict=True)
            ]
            scheduler.update(out, results)
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
    parser.add_argument("--chat", action="store_true", help="decode from real chat prompts")
    parser.add_argument("--greedy", action="store_true", help="greedy instead of default sampling")
    parser.add_argument("--draft-model", help="speculative decoding with this draft model")
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--quantize", choices=["int8"], help="int8 weights (target and draft)")
    parser.add_argument("--kv-mem-gb", type=float, default=4.0, help="KV cache, all pools")
    args = parser.parse_args()

    def load(model_dir, config):
        return load_weights(model_dir, config, default_device(), torch.bfloat16, args.quantize)

    model_dir = resolve_model_dir(args.model)
    config = ModelConfig.load(model_dir)
    weights = load(model_dir, config)
    draft, k = None, 0
    if args.draft_model:
        draft_dir = resolve_model_dir(args.draft_model)
        draft_config = ModelConfig.load(draft_dir)
        draft = (load(draft_dir, draft_config), draft_config)
        k = args.num_speculative_tokens
    engine = make_engine(weights, config, draft, k, args.kv_mem_gb)
    bench_decode(engine, 2, 8)  # warm-up: compile MPS kernels, touch the KV pool
    longest = engine.scheduler.config.max_model_len - 1
    for length in (600, 8000):
        bench_prefill(engine, min(length, longest))

    for batch in [int(b) for b in args.batch.split(",")]:
        before = engine.stats()
        prompt_ids = chat_prompts(model_dir, batch) if args.chat else None
        rate, step_ms = bench_decode(
            engine, batch, args.new_tokens, args.context, prompt_ids, args.greedy
        )
        line = f"decode  B={batch:<3} {rate:7.1f} tok/s   {step_ms:6.2f} ms/step"
        after = engine.stats()
        if drafted := after["draft_tokens"] - before["draft_tokens"]:
            accepted = after["accepted_draft_tokens"] - before["accepted_draft_tokens"]
            line += f"   drafts accepted {accepted / drafted:.0%}"
        print(line, flush=True)
        if args.profile and not draft:
            phases = profile_decode(engine, batch, args.new_tokens, args.context)
            print("        " + "  ".join(f"{k} {v:.2f}" for k, v in phases.items()), flush=True)
    for length in [int(n) for n in args.prefill.split(",") if n]:
        print(f"prefill {length:>5} tokens: TTFT {bench_prefill(engine, length):7.0f} ms")


if __name__ == "__main__":
    main()
