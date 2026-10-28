# The simplest possible generation loop: one sequence, greedy.
#
# Scaffolding for the early milestones. It shows the core autoregressive loop
# (predict one token, append it, repeat) before any engine machinery exists.
# The scheduler and engine replace it in M5.
#
# Try it:
#   uv run python -m atomserve.generate Qwen/Qwen3-0.6B "The capital of France is"
#   uv run python -m atomserve.generate Qwen/Qwen3-0.6B "The capital of France is" --no-cache

import argparse
import time
from collections.abc import Iterator

import torch
from tokenizers import Tokenizer

from atomserve.config import ModelConfig
from atomserve.kv_cache import KVCache
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import Weights, compute_logits, forward

# Peak unified-memory bandwidth of the M4 Pro, used for the decode speed ceiling.
M4_PRO_BANDWIDTH_GBS = 273.0


@torch.inference_mode()
def iter_greedy(
    weights: Weights,
    config: ModelConfig,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_ids: frozenset[int] = frozenset(),
    use_cache: bool = True,
) -> Iterator[int]:
    """Greedily extend `prompt_ids`, yielding each new token id as soon as it is chosen.

    With the cache: one forward pass over the whole prompt fills the cache
    (prefill), then every step feeds only the single newest token (decode).
    Without it: every step re-runs the model over prompt + everything generated
    so far, recomputing keys/values that have not changed (O(N^2) work).
    """
    device = weights.embed.device
    ids = list(prompt_ids)
    cache = None
    if use_cache:
        cache = KVCache(config, len(ids) + max_new_tokens, device, weights.embed.dtype)

    for _ in range(max_new_tokens):
        start = cache.length if cache is not None else 0  # first position not yet processed
        input_ids = torch.tensor(ids[start:], device=device)
        positions = torch.arange(start, len(ids), device=device)
        hidden = forward(weights, config, input_ids, positions, cache)
        next_id = int(compute_logits(weights, hidden[-1:]).argmax(dim=-1))
        ids.append(next_id)
        yield next_id
        if next_id in stop_ids:
            return


def generate_greedy(
    weights: Weights,
    config: ModelConfig,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_ids: frozenset[int] = frozenset(),
    use_cache: bool = True,
) -> list[int]:
    """Greedy continuation of `prompt_ids` as a list of new token ids (see `iter_greedy`)."""
    return list(iter_greedy(weights, config, prompt_ids, max_new_tokens, stop_ids, use_cache))


def weight_bytes(weights: Weights) -> int:
    """Bytes of all distinct parameter tensors (a tied lm_head is counted once).

    A decode step multiplies one token's activations by every weight matrix, so
    this is (almost exactly) how many bytes each decode step must read from memory.
    """
    tensors = [weights.embed, weights.final_norm, weights.lm_head]
    for layer in weights.layers:
        tensors.extend(vars(layer).values())
    unique = {t.data_ptr(): t for t in tensors}.values()
    return sum(t.numel() * t.element_size() for t in unique)


def main() -> None:
    """Greedy continuation of a raw (un-templated) prompt, with prefill/decode timings."""
    parser = argparse.ArgumentParser(description="Greedy single-sequence generation demo.")
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-cache", action="store_true", help="recompute everything each step")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    config = ModelConfig.load(model_dir)
    weights = load_weights(model_dir, config, default_device(), torch.bfloat16)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_ids = tokenizer.encode(args.prompt).ids

    # Warm-up: the first MPS run compiles kernels, which would distort the timings.
    generate_greedy(weights, config, prompt_ids, 2, use_cache=not args.no_cache)

    new_ids: list[int] = []
    start = time.perf_counter()
    for token in iter_greedy(
        weights, config, prompt_ids, args.max_new_tokens, use_cache=not args.no_cache
    ):
        if not new_ids:
            first_token_at = time.perf_counter()
        new_ids.append(token)
    end = time.perf_counter()

    print(args.prompt + tokenizer.decode(new_ids))
    ttft_ms = (first_token_at - start) * 1000
    decode_rate = (len(new_ids) - 1) / (end - first_token_at)
    gb_per_step = weight_bytes(weights) / 1e9
    print(f"\n[cache: {'off' if args.no_cache else 'on'}]")
    print(f"  prompt: {len(prompt_ids)} tokens, time to first token: {ttft_ms:.0f} ms")
    print(f"  decode: {len(new_ids) - 1} tokens at {decode_rate:.1f} tok/s")
    bw = M4_PRO_BANDWIDTH_GBS
    print(
        f"  ceiling: {gb_per_step:.2f} GB of weights read per step / {bw:.0f} GB/s"
        f" = {bw / gb_per_step:.0f} tok/s"
    )


if __name__ == "__main__":
    main()
