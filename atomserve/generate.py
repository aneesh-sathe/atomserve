# The simplest possible generation loop: one sequence, greedy.
#
# Scaffolding for the early milestones. It shows the core autoregressive loop
# (predict one token, append it, repeat) on top of the paged KV pool and the
# runner, doing by hand what the scheduler and engine take over in M5:
# growing the block table and advancing the sequence after each step.
#
# Try it:
#   uv run python -m atomserve.generate Qwen/Qwen3-0.6B "The capital of France is"
#   uv run python -m atomserve.generate Qwen/Qwen3-0.6B "The capital of France is" --no-cache

import argparse
import time
from collections.abc import Iterator

import torch
from tokenizers import Tokenizer

from atomserve.block_allocator import BlockAllocator, blocks_for_tokens
from atomserve.config import ModelConfig
from atomserve.kv_cache import KVPool
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import Weights
from atomserve.runner import ModelRunner
from atomserve.sequence import Sequence

# Peak unified-memory bandwidth of the M4 Pro, used for the decode speed ceiling.
M4_PRO_BANDWIDTH_GBS = 273.0
BLOCK_SIZE = 16  # tokens per KV block


def reserve_blocks(seq: Sequence, allocator: BlockAllocator, block_size: int) -> None:
    """Grow `seq.block_table` until it has a slot for every one of its tokens.

    A sequence needs a new block only every `block_size` tokens; in between,
    decode steps write into the free tail of its last block.
    """
    needed = blocks_for_tokens(len(seq.token_ids), block_size) - len(seq.block_table)
    if needed > 0:
        seq.block_table.extend(allocator.allocate(needed))


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

    With the cache: the first step runs the whole prompt (prefill), then every
    step feeds only the single newest token (decode), its K/V appended to the pool.
    Without it: every step frees the sequence's blocks and re-runs prompt +
    everything generated so far, recomputing K/V that have not changed (O(N^2) work).
    """
    num_blocks = blocks_for_tokens(len(prompt_ids) + max_new_tokens, BLOCK_SIZE)
    pool = KVPool(config, num_blocks, BLOCK_SIZE, weights.embed.device, weights.embed.dtype)
    allocator = BlockAllocator(num_blocks)
    runner = ModelRunner(weights, config, pool)
    seq = Sequence(token_ids=list(prompt_ids))

    for _ in range(max_new_tokens):
        if not use_cache:
            allocator.free(seq.block_table)
            seq.block_table, seq.num_computed_tokens = [], 0
        reserve_blocks(seq, allocator, BLOCK_SIZE)
        num_new = len(seq.token_ids) - seq.num_computed_tokens
        logits = runner.run([seq], [num_new])  # [1, vocab]
        seq.num_computed_tokens += num_new
        next_id = int(logits[0].argmax())
        seq.token_ids.append(next_id)
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
