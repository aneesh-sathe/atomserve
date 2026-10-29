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
- M4 was built before M3 at the user's request.
- `block_allocator.py` is separate from `kv_cache.py` (plan had them together) so the scheduler
  stays pure Python: the allocator tracks block ids, `KVPool` only holds tensors.
- The runner never mutates a `Sequence`; only the scheduler does (on the engine thread).
- Engine deals in token ids only; detokenization and stop strings live in the clients (REPL, server).

## Commands
- `uv sync` - install (Python 3.14 pinned in `.python-version`; torch ships cp314 macOS wheels)
- `uv run ruff check .` and `uv run ruff format --check .` - lint/format
- `uv run pytest` - fast tests (default excludes `slow`)
- `uv run pytest -m slow` - tests needing model weights / GPU

- `uv run atomserve Qwen/Qwen3-0.6B` - chat REPL (`/help` inside) + HTTP API on :11435
- `uv run atomserve Qwen/Qwen3-0.6B --no-repl --port 11499` - API only (logs requests)
- `uv run python bench/load.py --url http://127.0.0.1:11499 --concurrency 1,4,16,32` - load test

## Pitfalls / verified facts
- Qwen3-0.6B ships `lm_head.weight` even though `tie_word_embeddings=true`; follow the config (use the embedding).
- Golden tests: reference = transformers fp32 on CPU, eager attention. Ours fp32 on MPS differs by ~1.3e-4 max logit.
  bf16 vs fp32 reference: cosine ~0.999, occasional top-1 flips at individual positions (expected).
- `F.scaled_dot_product_attention(..., enable_gqa=True)` works on MPS.
- SDPA's `is_causal=True` aligns the mask top-left, which is wrong once queries sit after a cached prefix;
  `model.attention` builds an explicit position-based mask instead (skipped for single-token decode).
- Baseline perf (M2, bf16, single seq, M4 Pro): decode ~80 tok/s vs a ~229 tok/s bandwidth ceiling
  (1.19 GB weights/step at 273 GB/s), so decode is overhead-bound. 1000-token prompt: prefill ~270 ms;
  decode 74 tok/s with cache vs 3.6 tok/s without. Always warm up MPS before timing.
- M4 paged path (bf16): single seq decode ~74 tok/s (short) / ~67 tok/s (1000 ctx), ~10% below M2
  because each layer gathers the sequence's K/V out of its blocks (`index_select` copy).
  Lockstep decode, 100-token prompts, aggregate tok/s: B=1 73, B=4 244, B=8 396, B=16 460, B=32 637.
  Step time grows with B mostly from the per-sequence Python loop in `paged_attention`.
- `tensor.view(-1, ...)` of a pool layer + `index_copy_` / `index_select` scatter/gather slots correctly on MPS.
- Fast tests run the full paged model on a tiny random-weight config on CPU (`tiny_weights` fixture);
  use it for engine invariants instead of the real model where possible.
- M6 bench (bf16, 128 output tokens, HTTP streaming): aggregate 72 / 221 / 445 / 609 tok/s at
  concurrency 1 / 4 / 16 / 32; TTFT p50 23 / 61 / 171 / 250 ms.
- M7 chunked prefill, 8 streams + one 8k-token prompt: before, short-request TTFT p99 ~4.9 s
  (head-of-line blocking) and long TTFT 6.4 s; after, TTFT p99 ~95 ms, long TTFT 3.7 s. ITL max is
  ~1150 ms at budget 2048 vs ~325 ms at 512, while a lone 8k prefill is only ~2% slower at 512,
  hence the 512 default.
- M8 prefix caching, chat about a ~4k-token document: turn 2 (30 output tokens) takes 616 ms
  with caching vs 1944 ms without; turn-1 cost is unchanged.
- Prefix-cache invariants: only *fully computed* blocks are hashed (never the partly filled
  last block); at least one prompt token is always recomputed for its logits; blocks are freed
  last-first so a prefix's head is evicted last. Mutation-tested in tests/test_scheduler.py.
- Greedy outputs of identical requests can differ under bf16 when batch composition changes
  (different matmul shapes round differently). In fp32 they are identical, so invariance tests use fp32.
- E2E terminal tests need the pty as the child's *controlling* tty or Ctrl-C never becomes SIGINT.
  Don't use `pty.fork()` inside pytest (forking a multi-threaded process can deadlock the child);
  `tests/test_repl.py` spawns via subprocess + `TIOCSCTTY`. Pause ~0.3 s after the prompt before
  typing, and never wait on the child without a timeout.
- Qwen3-0.6B sometimes loops forever while thinking on trick prompts; that is the model, not the engine.
- Importing `transformers` prints a harmless torch.distributed "Redirects" warning on macOS; the engine never imports it.

## Conventions
- Commit messages: under 10 words, tagged (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `delete:`). No co-author lines.
- Never use em dashes in code, comments or docs.
