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

## Conventions
- Commit messages: under 10 words, tagged (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `delete:`). No co-author lines.
- Never use em dashes in code, comments or docs.
