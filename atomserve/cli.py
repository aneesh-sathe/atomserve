# Command-line entrypoint: `atomserve <hf_model>`.
#
# This is the top of the dependency graph. It parses arguments and wires the
# pieces together; it owns no inference logic itself:
#
#     load config + weights + tokenizer  ->  size the KV pool  ->  build the engine
#     ->  warm up  ->  start the engine thread  ->  start the HTTP server  ->  run the chat REPL
#
# The REPL and the HTTP server are both just clients of the one engine thread,
# so requests from either are batched together.

import argparse
import readline  # noqa: F401  (gives input() line editing and history)
import sys
import threading
import time

import torch

from atomserve import __version__
from atomserve.config import EngineConfig, ModelConfig
from atomserve.engine import create_engine
from atomserve.kv_cache import KVPool
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import Weights
from atomserve.repl import Repl
from atomserve.sequence import SamplingParams
from atomserve.server import OpenAIServer
from atomserve.tokenizer import ChatTokenizer

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface: one positional model id, plus engine knobs."""
    parser = argparse.ArgumentParser(
        prog="atomserve",
        description="Run a Hugging Face model locally and chat with it in the terminal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="Hugging Face model id (e.g. Qwen/Qwen3-0.6B) or a local model directory",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP API address")
    parser.add_argument("--port", type=int, default=11435, help="HTTP API port")
    parser.add_argument(
        "--no-repl", action="store_true", help="serve the HTTP API only, no terminal chat"
    )
    parser.add_argument(
        "--quantize",
        choices=["int8"],
        help="store weights as int8 (half the memory, faster small-batch decode)",
    )
    parser.add_argument("--dtype", choices=DTYPES, default="bf16", help="weights and KV cache")
    parser.add_argument(
        "--kv-mem-gb", type=float, default=4.0, help="GiB of memory for the KV cache"
    )
    parser.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="most requests running at once"
    )
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        default=512,
        help="token budget of one forward pass; smaller bounds the stall a long prompt "
        "causes running streams, larger prefills a little faster",
    )
    parser.add_argument(
        "--no-prefix-caching", action="store_true", help="always recompute shared prompt prefixes"
    )
    parser.add_argument(
        "--draft-model",
        help="smaller model of the same family for speculative decoding (e.g. Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=3,
        help="tokens the draft model proposes per step (with --draft-model)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def engine_config_for(
    args: argparse.Namespace, model_config: ModelConfig, draft_config: ModelConfig | None = None
) -> EngineConfig:
    """Size the KV pool(s) from the memory budget; cap sequence length by pool and model.

    With a draft model, every block exists in both models' pools, so a block costs both.
    """
    configs = [model_config] + ([draft_config] if draft_config else [])
    dtype = DTYPES[args.dtype]
    block_bytes = sum(KVPool.bytes_per_block(c, args.block_size, dtype) for c in configs)
    num_blocks = int(args.kv_mem_gb * 2**30 // block_bytes)
    return EngineConfig(
        block_size=args.block_size,
        num_blocks=num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_len=min(model_config.max_position_embeddings, num_blocks * args.block_size),
        enable_prefix_caching=not args.no_prefix_caching,
        num_speculative_tokens=args.num_speculative_tokens if draft_config else 0,
    )


def fit_kv_budget(args: argparse.Namespace, weights: list[Weights], device: torch.device) -> None:
    """Shrink --kv-mem-gb if weights + KV cache would exceed the GPU's working set.

    Past torch.mps.recommended_max_memory(), macOS starts paging GPU memory and
    every step slows down severalfold, with no error. 1 GiB stays free for activations.
    """
    if device.type != "mps":
        return
    weight_bytes = sum(w.nbytes for w in weights)
    room_gb = (torch.mps.recommended_max_memory() - weight_bytes) / 2**30 - 1.0
    if args.kv_mem_gb > room_gb:
        if room_gb < 0.25:
            raise SystemExit("atomserve: the model(s) leave no GPU memory for a KV cache.")
        print(f"Note: KV cache reduced to {room_gb:.1f} GiB to fit in GPU memory.")
        args.kv_mem_gb = room_gb


def main(argv: list[str] | None = None) -> int:
    """Load the model, start the engine and chat. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    model_dir = resolve_model_dir(args.model)
    model_config = ModelConfig.load(model_dir)
    device = default_device()
    weight_format = f"{args.quantize} weights" if args.quantize else args.dtype
    print(f"Loading {args.model} ({weight_format}, {device.type}) ...", flush=True)
    weights = load_weights(model_dir, model_config, device, DTYPES[args.dtype], args.quantize)
    tokenizer = ChatTokenizer(model_dir)
    draft = None
    if args.draft_model:
        draft_dir = resolve_model_dir(args.draft_model)
        if (draft_dir / "tokenizer.json").read_bytes() != (
            model_dir / "tokenizer.json"
        ).read_bytes():
            print(f"atomserve: {args.draft_model} has a different tokenizer from {args.model}.")
            return 1
        print(f"Loading draft model {args.draft_model} ...", flush=True)
        draft_config = ModelConfig.load(draft_dir)
        draft_weights = load_weights(
            draft_dir, draft_config, device, DTYPES[args.dtype], args.quantize
        )
        draft = (draft_weights, draft_config)
    fit_kv_budget(args, [weights] + ([draft[0]] if draft else []), device)
    engine_config = engine_config_for(args, model_config, draft[1] if draft else None)
    engine = create_engine(weights, model_config, engine_config, tokenizer.eos_token_ids, draft)

    # The first MPS run compiles kernels; pay that now rather than on the first reply.
    engine.generate([tokenizer.encode("Hello")], SamplingParams(temperature=0.0, max_tokens=2))
    try:
        server = OpenAIServer(
            engine, tokenizer, args.model, args.host, args.port, log_requests=args.no_repl
        )
    except OSError as error:
        print(f"atomserve: cannot listen on {args.host}:{args.port} ({error.strerror}).")
        print("Is another atomserve running? Pick another port with --port.")
        return 1
    engine.start()
    server.start()
    print(
        f"Ready in {time.perf_counter() - start:.1f}s. KV cache: "
        f"{engine_config.num_blocks * engine_config.block_size:,} tokens, "
        f"context up to {engine_config.max_model_len:,}."
    )
    print(f"OpenAI-compatible API on http://{args.host}:{server.port}/v1", flush=True)

    if args.no_repl:
        print("Press Ctrl-C to stop.", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print()
    else:
        print("Type a message, /help for commands, Ctrl-D to exit.")
        Repl(engine, tokenizer).run()
    server.shutdown()
    engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
