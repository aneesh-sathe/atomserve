# Command-line entrypoint: `atomserve <hf_model>`.
#
# This is the top of the dependency graph. It parses arguments and wires the
# pieces together; it owns no inference logic itself. Flags are added here only
# as the milestone that needs them lands.

import argparse
import sys

from atomserve import __version__


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface: one positional model id, plus global flags."""
    parser = argparse.ArgumentParser(
        prog="atomserve",
        description="Run a Hugging Face model locally and chat with it in the terminal.",
    )
    parser.add_argument(
        "model",
        help="Hugging Face model id (e.g. Qwen/Qwen3-0.6B) or a local model directory",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run atomserve. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    print(f"atomserve: serving '{args.model}' is not implemented yet.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
