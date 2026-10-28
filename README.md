# atomserve

A minimal, from-scratch LLM inference engine for Apple silicon, written in plain PyTorch (MPS).
Built to learn how inference engines work: KV cache, paged memory, scheduling, continuous batching.

The code is the documentation: start at `atomserve/cli.py` and follow the imports.

## Install

Requires macOS on Apple silicon and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run atomserve Qwen/Qwen3-0.6B
```

## Development

```sh
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest                # fast tests
uv run pytest -m slow        # tests that need model weights / the GPU
```
