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
    parser.add_argument("--dtype", choices=DTYPES, default="bf16", help="weights and KV cache")
    parser.add_argument(
        "--kv-mem-gb", type=float, default=4.0, help="GiB of memory for the KV cache (default 4)"
    )
    parser.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="most requests running at once"
    )
    parser.add_argument(
        "--max-batched-tokens", type=int, default=2048, help="token budget of one forward pass"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def engine_config_for(args: argparse.Namespace, model_config: ModelConfig) -> EngineConfig:
    """Size the KV pool from the memory budget; cap sequence length by pool and model."""
    block_bytes = KVPool.bytes_per_block(model_config, args.block_size, DTYPES[args.dtype])
    num_blocks = int(args.kv_mem_gb * 2**30 // block_bytes)
    return EngineConfig(
        block_size=args.block_size,
        num_blocks=num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_len=min(model_config.max_position_embeddings, num_blocks * args.block_size),
    )


def main(argv: list[str] | None = None) -> int:
    """Load the model, start the engine and chat. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    model_dir = resolve_model_dir(args.model)
    model_config = ModelConfig.load(model_dir)
    device = default_device()
    print(f"Loading {args.model} ({args.dtype}, {device.type}) ...", flush=True)
    weights = load_weights(model_dir, model_config, device, DTYPES[args.dtype])
    tokenizer = ChatTokenizer(model_dir)
    engine_config = engine_config_for(args, model_config)
    engine = create_engine(weights, model_config, engine_config, tokenizer.eos_token_ids)

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
