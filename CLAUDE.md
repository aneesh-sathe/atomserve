# atomserve - project notes

Minimal, learning-focused LLM inference engine for Apple silicon (PyTorch MPS).
Full spec and milestones: `~/.claude/plans/i-want-to-build-tidy-stream.md`.

## Decisions
- Single model family: Qwen3 (`Qwen3ForCausalLM`), developed against `Qwen/Qwen3-0.6B`; 1.7B (fp32),
  4B and 8B (bf16, 8B also int8) verified against transformers too. Fail fast on anything else.
  Big-model reference checks: run transformers in its own process (CPU, bf16) and save logits first;
  8B bf16 needs 16 GB of RAM for the reference, and our bf16 8B only fits with a tiny KV pool.
- Chat only, no tool calling.
- Plain tensors and functions; `transformers` is a dev dependency for golden tests only, never imported by `atomserve/`.
- One process: engine thread + chat REPL + OpenAI-compatible HTTP (stdlib `http.server`). No asyncio.
- Code is the doc: every file has a header comment on its concept; every function a docstring with
  purpose, tensor shapes and the non-obvious "why". No separate design docs.
- README.md is the one overview doc: comprehensive, succinct, technically detailed (user request).
  Keep its numbers and module map in sync when behavior or measurements change.
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
- `uv run python bench/engine.py --profile` - in-process decode/prefill benchmark with step breakdown
- `uv run python bench/engine.py --model Qwen/Qwen3-4B --chat --quantize int8 --draft-model Qwen/Qwen3-0.6B`
  - bigger model, int8, speculative (use `--chat`: draft acceptance needs real text)

## Pitfalls / verified facts
- Qwen3-0.6B ships `lm_head.weight` even though `tie_word_embeddings=true`; follow the config (use the embedding).
- Golden tests: reference = transformers fp32 on CPU, eager attention. Ours fp32 on MPS differs by ~1.3e-4 max logit.
  bf16 vs fp32 reference: cosine ~0.999, occasional top-1 flips at individual positions (expected).
- `F.scaled_dot_product_attention(..., enable_gqa=True)` works on MPS.
- SDPA's `is_causal=True` aligns the mask top-left, which is wrong once queries sit after a cached prefix;
  `model.attention` builds an explicit position-based mask instead (skipped for single-token decode).
- Current perf (bf16, M4 Pro, `bench/engine.py`): decode 127 / 919 / 2082 tok/s at B = 1 / 8 / 32
  (100-token context), 103 tok/s at B=1 with 4k context; prefill 1k 213 ms, 8k 3.07 s. HTTP
  (`bench/load.py`): 132 / 440 / 1299 / 2025 tok/s at concurrency 1 / 4 / 16 / 32. Ceiling at B=1 is
  ~229 tok/s (1.19 GB weights/step at 273 GB/s). Pre-optimization (e3dea4b): 69.5 / 362 / 576.
- Benchmarking: always warm up MPS; compare before/after back to back in one session (runs ~5% apart
  across sessions); exclude multi-step (chunked) prefill from decode timing; test long contexts too
  (the first attention kernel looked fine at 100 tokens and was 5x slower than SDPA at 16k).
- MPS facts measured here: `torch.tensor(list, device="mps")` costs ~140 us (it syncs); build on CPU
  and `.to(mps, non_blocking=True)`, but the GPU reads the source *later*, so keep it alive and
  unmodified until the step's sync (runner._Upload). `torch.topk` costs ~110 us per row of 152k.
  One matmul with N=151,936 and >8 rows hits a slow kernel (7.7 ms at 32 rows vs 2.2 ms sliced).
  Each small op is ~3-5 us of launch; `F.rms_norm` is one kernel vs eight for the longhand version.
- `torch.compile` on MPS (torch 2.14) works but was no net win (decode +16% only at B=8, prefill
  slower, startup 2x), so it is not used.
- `torch.mps.compile_shader`: Python args bind to `[[buffer(i)]]` in call order, ints arrive as
  `long`, tensor views with a storage offset bind correctly, `threads=`/`group_size=` take 1-3D tuples.
- `tensor.view(-1, ...)` of a pool layer + `index_copy_` / `index_select` scatter/gather slots correctly on MPS.
- The KV pool is zero-initialised on purpose: the PyTorch decode path masks padded keys, but a
  masked key still multiplies its value by 0, and 0 * NaN (uninitialised memory) is NaN.
- Metal kernels (kernels.py) assume head_dim 128 and a GQA ratio in (1, 2, 4, 5, 8); otherwise
  the runner falls back to the PyTorch path. tests/test_kernels.py checks kernels vs PyTorch.
  The int8 matmul needs K % 256 == 0 (true for every Qwen3 size) and falls back otherwise.
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
  typing, and never wait on the child without a timeout. Keep reading the pty while waiting for
  exit: macOS blocks a process's exit until its unread terminal output drains.
- Qwen3-8B (untied lm_head) on this machine: bf16 15.7 tok/s at batch 1 (KV cut to 1.5 GiB),
  int8 29.0, int8 + 0.6B draft k=2 43.8. A 1.7B draft is accepted more (75-85%) but is no faster.
- Speculative decoding (4B target, 0.6B draft, batch 1, chat text): 28 -> 45 tok/s bf16, 64 tok/s with
  int8 for both; k=2-3 best; loses from batch 4 up (verification rows hit MPS's M>8 matmul cost),
  hence max_speculative_batch=2. Greedy spec == target alone is tested (tiny models + 1.7B/0.6B fp32).
- int8 (`quant.py`): perplexity within ~1% of bf16 (0.6B/1.7B/4B). Batch-1 decode 1.4x (0.6B), 1.75x (4B);
  batch 32 only 0.6-0.75x of bf16 (scalar kernel; would need simdgroup_matrix). Template the kernel on
  M or the accumulators spill (it was 2x slower). `torch._weight_int8pack_mm` on MPS was only fast for
  one row of a big matrix. Never dequantize through fp32 (`q.float() * scale`) for big models: the
  temporaries hit the memory limit (4B batch 32 went to 525 ms/step); use kernels.dequantize_int8.
- `torch.mps.recommended_max_memory()` is 17.8 GiB on this 24 GB machine. Exceeding it (weights + KV)
  pages GPU memory: every step slows ~2x with no error. cli.fit_kv_budget shrinks the KV cache.
- The loader streams tensors from memory-mapped shards (safe_open) and quantizes each projection as it
  arrives: CPU and GPU share memory, so reading whole shards first held the checkpoint twice. Then
  `torch.mps.empty_cache()`: the MPS allocator keeps freed buffers cached (4B int8 load: 16.1 GiB of GPU
  memory with the old loader, 7.0 GiB now, for 4.5 GiB of weights).
- A Hub cache entry can be partial (e.g. only config.json fetched by another tool); `snapshot_download
  (local_files_only=True)` still returns it. loader.missing_files checks before trusting it.
- Python reads `coding[:=]` in a file's first two lines as an encoding declaration: a header comment like
  "# Speculative decoding: a ..." is a SyntaxError ("unknown encoding: a"). Ruff doesn't catch it.
- Benchmarks must count tokens, not steps (a speculative step yields several).
- Qwen3-0.6B sometimes loops forever while thinking on trick prompts; that is the model, not the engine.
- Importing `transformers` prints a harmless torch.distributed "Redirects" warning on macOS; the engine never imports it.

## Conventions
- Commit messages: under 10 words, tagged (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `delete:`). No co-author lines.
- Never use em dashes in code, comments or docs.
