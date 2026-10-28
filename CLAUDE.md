# atomserve - project notes

Minimal, learning-focused LLM inference engine for Apple silicon (PyTorch MPS).
Full spec and milestones: `~/.claude/plans/i-want-to-build-tidy-stream.md`.

## Decisions
- Single model family: Qwen3 (`Qwen3ForCausalLM`), developed against `Qwen/Qwen3-0.6B`. Fail fast on anything else.
- Chat only, no tool calling.
- Plain tensors and functions; `transformers` is a dev dependency for golden tests only, never imported by `atomserve/`.
- One process: engine thread + chat REPL + OpenAI-compatible HTTP (stdlib `http.server`). No asyncio.
- Code is the doc: every file has a header comment on its concept; every function a docstring with
  purpose, tensor shapes and the non-obvious "why". No separate design docs; README is install + usage.
- Built milestone by milestone (M0-M8); the user reviews each before the next starts.

## Commands
- `uv sync` - install (Python 3.14 pinned in `.python-version`; torch ships cp314 macOS wheels)
- `uv run ruff check .` and `uv run ruff format --check .` - lint/format
- `uv run pytest` - fast tests (default excludes `slow`)
- `uv run pytest -m slow` - tests needing model weights / GPU

- `uv run python -m atomserve.generate Qwen/Qwen3-0.6B "prompt"` - greedy demo (temporary until the engine lands)

## Pitfalls / verified facts
- Qwen3-0.6B ships `lm_head.weight` even though `tie_word_embeddings=true`; follow the config (use the embedding).
- Golden tests: reference = transformers fp32 on CPU, eager attention. Ours fp32 on MPS differs by ~1.3e-4 max logit.
  bf16 vs fp32 reference: cosine ~0.999, occasional top-1 flips at individual positions (expected).
- `F.scaled_dot_product_attention(..., enable_gqa=True)` works on MPS.
- Importing `transformers` prints a harmless torch.distributed "Redirects" warning on macOS; the engine never imports it.

## Conventions
- Commit messages: under 10 words, tagged (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `delete:`). No co-author lines.
- Never use em dashes in code, comments or docs.
