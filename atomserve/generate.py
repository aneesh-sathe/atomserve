# The simplest possible generation loop: one sequence, greedy, no cache.
#
# Scaffolding for the early milestones. It shows the core autoregressive loop
# (predict one token, append it, repeat) before any engine machinery exists.
# The scheduler and engine replace it in M5.
#
# Try it:  uv run python -m atomserve.generate Qwen/Qwen3-0.6B "The capital of France is"

import argparse
import time

import torch
from tokenizers import Tokenizer

from atomserve.config import ModelConfig
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import Weights, compute_logits, forward


@torch.inference_mode()
def generate_greedy(
    weights: Weights,
    config: ModelConfig,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_ids: set[int] = frozenset(),
) -> list[int]:
    """Greedily extend `prompt_ids` by up to `max_new_tokens` tokens; returns only the new ones.

    Each step re-runs the full model over prompt + everything generated so far and
    keeps the argmax of the last position's logits. Correct but O(N^2): every step
    recomputes keys/values for tokens whose keys/values have not changed.
    """
    device = weights.embed.device
    ids = list(prompt_ids)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(ids, device=device)
        positions = torch.arange(len(ids), device=device)
        hidden = forward(weights, config, input_ids, positions)
        next_id = int(compute_logits(weights, hidden[-1:]).argmax(dim=-1))
        ids.append(next_id)
        new_ids.append(next_id)
        if next_id in stop_ids:
            break
    return new_ids


def main() -> None:
    """Load a model and print a greedy continuation of a raw (un-templated) prompt."""
    parser = argparse.ArgumentParser(description="Greedy, cache-less generation demo.")
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    config = ModelConfig.load(model_dir)
    weights = load_weights(model_dir, config, default_device(), torch.bfloat16)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    prompt_ids = tokenizer.encode(args.prompt).ids
    start = time.perf_counter()
    new_ids = generate_greedy(weights, config, prompt_ids, args.max_new_tokens)
    elapsed = time.perf_counter() - start

    print(args.prompt + tokenizer.decode(new_ids))
    rate = len(new_ids) / elapsed
    print(f"\n[{len(new_ids)} tokens in {elapsed:.2f}s = {rate:.1f} tok/s, no cache]")


if __name__ == "__main__":
    main()
