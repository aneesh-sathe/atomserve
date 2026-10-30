# atomserve

A small LLM inference engine for Apple silicon, written from scratch in plain PyTorch (MPS).
`atomserve <model>` loads a model, opens a streaming chat in the terminal, and serves an
OpenAI-compatible HTTP API from the same process, ollama-style.

It is built to be read. The core techniques of production servers such as vLLM and SGLang are
all here:

- paged KV cache
- continuous batching with preemption
- chunked prefill
- prefix caching
- batched sampling
- speculative decoding with a draft model
- int8 weight quantization
- hand-written Metal kernels for the hot paths

There are no frameworks hiding them: about 2,900 lines of code across 18 files (including seven
Metal kernels), one concept per file. Every file opens with a comment explaining its concept;
every function documents its tensor shapes and the non-obvious "why".

- **Models:** Qwen3 (`Qwen3ForCausalLM`). Qwen3-0.6B, 1.7B and 4B have been verified against
  Hugging Face `transformers`; larger Qwen3 checkpoints share the architecture. Other
  architectures are rejected at load time.
- **Scope:** chat and text completion. No tool calling, no multimodal input, one model per process.

## Quickstart

Requires macOS on Apple silicon and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run atomserve Qwen/Qwen3-0.6B
```

The first run downloads the model (~1.5 GB) into the Hugging Face cache; later runs start
offline in about a second.

```
Loading Qwen/Qwen3-0.6B (bf16, mps) ...
Ready in 0.8s. KV cache: 37,440 tokens, context up to 37,440.
OpenAI-compatible API on http://127.0.0.1:11435/v1
Type a message, /help for commands, Ctrl-D to exit.
>>> What is 17*3?
Okay, the user is asking what 17 multiplied by 3 is ...   (reasoning, shown dimmed)

51
```

### REPL commands

| Command | Effect |
|---|---|
| `/think [on\|off]` | Toggle Qwen3's thinking mode (reasoning before the answer). Also switches the sampling defaults. |
| `/params k=v ...` | Set `temperature`, `top_p`, `top_k`, `presence_penalty`, `max_tokens`, `seed`. `/params reset` restores the defaults. |
| `/reset` | Forget the conversation. |
| `/stats` | Last reply's TTFT and decode speed, prefix-cache hits, running/waiting requests, KV usage. |
| `/exit`, `/bye`, Ctrl-D | Quit. |

Ctrl-C stops the current reply, not the program. Wrap a message in `"""` to write several lines.

### HTTP API

```sh
curl http://127.0.0.1:11435/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat. Streaming (SSE) or not. |
| `POST /v1/completions` | Raw-text continuation. Streaming (SSE) or not. |
| `GET /v1/models` | The model being served. |
| `GET /health` | 200 once the server is up. |

Supported request fields:
- `max_tokens` / `max_completion_tokens`
- `temperature`, `top_p`, `presence_penalty`, `seed`
- `stop` (string or list)
- `stream`, `stream_options.include_usage`
- vLLM-style extensions: `top_k`, `ignore_eos`, `chat_template_kwargs.enable_thinking`

Replies keep Qwen3's `<think>...</think>` block in `content`; clients such as Open WebUI render it.
Errors use OpenAI's `{"error": {...}}` shape. A client that disconnects mid-stream has its request
aborted and its KV blocks freed.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `11435` | HTTP API address. |
| `--no-repl` | off | Serve the API only, logging requests. |
| `--dtype` | `bf16` | Weights and KV cache dtype (`bf16` or `fp32`). |
| `--kv-mem-gb` | `4` | GiB reserved for the KV cache; sets the pool's block count and the longest context. |
| `--block-size` | `16` | Tokens per KV block. |
| `--max-num-seqs` | `32` | Most requests running at once. |
| `--max-batched-tokens` | `512` | Token budget of one forward pass (see chunked prefill). |
| `--no-prefix-caching` | off | Always recompute shared prompt prefixes. |
| `--quantize int8` | off | int8 weights: half the weight memory, faster decode for 1 to 8 concurrent requests. |
| `--draft-model` | none | Speculative decoding with this smaller Qwen3 (e.g. `Qwen/Qwen3-0.6B` for 4B). |
| `--num-speculative-tokens` | `3` | Tokens the draft proposes per step. |

The fastest single-user setup for a bigger model combines both:

```sh
uv run atomserve Qwen/Qwen3-4B --quantize int8 --draft-model Qwen/Qwen3-0.6B
```

If weights plus KV cache would exceed the GPU's working set
(`torch.mps.recommended_max_memory()`), the KV cache is shrunk to fit and a note is printed.
Past that limit macOS pages GPU memory and every step slows down severalfold, with no error.

## How it works

### Request lifecycle

```
  REPL / HTTP handler threads                    engine thread (owns the GPU)
  ───────────────────────────                    ────────────────────────────
  render chat template, tokenize
  engine.submit(ids, params)  ──── inbox ────▶   loop:
                                                   drain inbox  -> scheduler.add / abort
                                                   plan   = scheduler.schedule()      pure Python
                                                   tokens = runner.execute(plan)      one forward + sample
                                                   scheduler.update(plan, tokens)     advance, retire
  for token in handle:  ◀──── per-request ─────     push each new token to its handle
    incremental detokenize          outbox
    stop strings, print / SSE
```

The engine thread is the only one that touches the scheduler or the GPU. Clients talk to it
through thread-safe queues, so terminal and HTTP requests are batched together. The engine deals
only in token ids; text, chat templates and stop strings belong to the clients. There's no
asyncio: one engine thread, one thread per HTTP request.

### Module map

Dependencies flow one way, top to bottom.

| File | Concept |
|---|---|
| `cli.py` | Entrypoint. Sizes the KV pool from `--kv-mem-gb`, wires the parts together, warms up MPS. |
| `repl.py` | Terminal chat client: history, streaming, dimmed thinking, Ctrl-C abort, commands. |
| `server.py` | OpenAI-compatible HTTP client of the engine: SSE, stop strings, disconnect handling. |
| `tokenizer.py` | BPE tokenizer, the Jinja chat template (rendered directly, matching `transformers` exactly), incremental detokenizer. |
| `engine.py` | Engine thread and step loop, `RequestHandle` streaming, stats. |
| `scheduler.py` | Continuous batching policy: admission, token budget, preemption, chunked prefill, prefix reuse. Pure Python. |
| `block_allocator.py` | KV block free list, refcounts, content-hashed prefix cache with LRU eviction. Pure Python. |
| `sequence.py` | `Sequence` state machine and `SamplingParams`. |
| `runner.py` | Flattens a scheduler plan into GPU tensors (the `Batch`), runs the model, samples. |
| `sampler.py` | Batched sampling: presence penalty, temperature, top-k, top-p, seeded draws. |
| `model.py` | The Qwen3 transformer as plain functions over a flat batch: RMSNorm, RoPE, GQA, paged attention, SwiGLU. The readable reference; also the CPU path. |
| `kernels.py` | Metal kernels for the hot spots: fused q/k-norm + RoPE + KV write, paged decode attention, top-k, SiLU·mul, int8 matmul. |
| `speculative.py` | Speculative decoding: draft proposals, one-pass verification by the target, rejection sampling. |
| `quant.py` | int8 weight quantization: per-row scales, and the matmul that picks a kernel by batch size. |
| `kv_cache.py` | The paged KV pool tensors: slot scatter (write) and gather (read). |
| `loader.py` | Hub download (offline when cached) and the safetensors-to-`Weights` mapping. |
| `config.py` | `ModelConfig` (validated `config.json`) and `EngineConfig` (serving limits). |

### The flat batch

Each step packs every scheduled sequence's new tokens, prefill chunks and single decode tokens
alike, into one list of T tokens with no padding (as in vLLM v1 and nano-vllm):

```
input_ids      [T]     new tokens of all sequences, concatenated
positions      [T]     absolute position of each token (RoPE, causal mask)
slot_mapping   [T]     KV pool slot each token's K/V is written to
logits_idx     [B']    rows that need a next-token prediction
prefills               per prompt chunk: its rows and the pool slots of its whole context
decode_rows    [Bd]    rows of the sequences with exactly one new token
decode_block_tables    [Bd, max_blocks] their block tables, padded, plus their context lengths
```

Every matmul (QKV, output projection, MLP, lm_head) runs on all `[T, hidden]` rows at once
without knowing which sequence a row belongs to. Attention is the only operation that needs
sequence boundaries. Prompt chunks are attended one at a time; there are few per step and each
is a large piece of work. All decodes share one batched attention call, so a step's launch count
doesn't grow with the batch. The lm_head (151,936 × 1024) runs only on `logits_idx` rows: the
last token of each sequence that finished its known tokens this step.

All of a step's integers reach the GPU in one asynchronous copy from a single host buffer.
On MPS each separate small copy costs 30 to 140 µs, about as much as a whole layer at batch 1.

### Paged KV cache

The KV pool is one preallocated tensor per K and V:

```
[layers, num_blocks, block_size, kv_heads, head_dim]
```

For Qwen3-0.6B in bf16 a token costs 28 × 2 × 8 × 128 × 2 B = 112 KiB, so a 16-token block is
1.75 MiB and 4 GiB holds 37,440 tokens.

A sequence's block table maps logical blocks to physical ones, so position `p` lives at:

```
slot = block_table[p // block_size] * block_size + p % block_size
```

Any free block will do, so memory doesn't fragment and nothing is reserved for a sequence's
maximum length.

The pool starts zeroed rather than uninitialized. The plain-PyTorch decode path pads short
contexts with slots that may never have been written, then masks them out; a masked key gets
weight 0, but 0 × NaN is still NaN.

### Metal kernels

`model.py` keeps every operation in plain PyTorch as the readable reference (and the CPU path).
On the Apple GPU, the runner switches to `kernels.py`, compiled at startup with
`torch.mps.compile_shader`. Tests check each kernel against the PyTorch version in fp32, and
check that the whole engine produces identical tokens with kernels on and off.

- **`qk_norm_rope_store`:** one launch per layer, replacing about 16. Thread i of a head's
  64-thread group owns the RoPE pair (i, i + 64). It RMS-normalizes q and k heads (a simdgroup
  reduction), rotates them by position with precise trig, writes q out and writes k and v
  straight into their pool slots.
- **`paged_decode_attention`:** the PagedAttention "v2" scheme.
  - Pass 1 runs one threadgroup per (KV head, sequence, 512-key partition) and serves all query
    heads of the GQA group, so each K/V row is read once.
  - Lane l of a simdgroup scores key l against the queries; the simdgroup then updates an
    online softmax and accumulates V, with each lane owning 4 dims.
  - Pass 2 merges the partitions. With a single partition, pass 1 writes the output itself.
  - It reaches 200 to 234 GB/s on K/V and is 2 to 3× faster than gathering K/V and calling SDPA.
- **`topk_rows`:** the sampler's top-k over the 151,936-entry vocabulary.
  - A threshold first: tau is the k-th largest of the 1,024 threads' maxima. Those k maxima are
    distinct entries ≥ tau, so the row's whole top k is ≥ tau.
  - Then each thread keeps a sorted list of its entries ≥ tau, and simd-level merges combine
    the lists.
  - It is 4× faster than `torch.topk` on MPS at batch 1 and 22× faster at batch 32.
- **`silu_mul`:** SwiGLU's `silu(gate) * up` in one pass.
- **`int8_linear`:** the int8 weight-only matmul for 1 to 8 activation rows (see below).

bf16 matmuls and prefill attention stay with PyTorch/MPS, which already run them at 65 to 95%
of the hardware's limits.

The causal mask for prompt chunks is built from absolute positions. SDPA's `is_causal` aligns
the triangle to the top-left, which is wrong once queries follow a cached prefix.

### Scheduler

Every step, within `max_num_seqs` and the `max_num_batched_tokens` budget:

1. **Running sequences first, oldest first.** Each gets one token for a decode, or the next chunk
   of an unfinished prompt. If the pool has no block for one, the most recently admitted
   sequence is preempted: its blocks are freed and it goes back to the front of the queue, to be
   recomputed from its tokens later. Recomputing needs no swap space, and with prefix caching
   the sequence usually finds its own blocks still cached.
2. **Then admit waiting sequences**, first come first served, while budget, sequence slots and
   blocks allow. Nothing new is admitted in a step that had to preempt.

**Chunked prefill.** A prompt never has to fit the budget in one piece; it takes what's left this
step and continues next step. So a long prompt neither waits for an empty step, blocking
everyone queued behind it, nor stalls running streams with one giant step. The budget caps
every step, and so caps how long a running stream waits for its next token.

### Prefix caching

A full block's K/V depend only on its tokens and every token before it, so blocks are named by a
chained hash:

```
h_i = sha256(h_{i-1} || tokens of block i)
```

- **Admission:** a sequence walks its full blocks in order through the cache and starts with
  every hit already "computed". At least one token is always left to compute, because its logits
  are needed.
- **After each step:** newly completed blocks are registered. Only fully computed blocks are
  ever hashed or shared, and full blocks are never written again, so sharing needs no
  copy-on-write.
- **Refcounts and eviction:** blocks are reference-counted. A block whose last user finished
  keeps its hash and stays reusable until `allocate()` needs memory. Eviction takes the least
  recently freed block first, and a sequence's blocks are freed last block first, so the head of
  a shared prefix is evicted last.

A chat resends its whole history every turn, so in practice everything up to the latest message
is a cache hit.

### Speculative decoding

At batch 1 a step costs about the time to read the target's weights, whether it computes one
token or five. So a small draft model of the same family proposes k tokens, one cheap step at a
time, and the target checks all of them in one pass:

```
draft:   ... t -> d1 -> d2 -> d3               (k sequential draft steps)
target:  [t, d1, d2, d3] in one pass           -> p1 .. p4
accept:  d1 kept, d2 kept, d3 rejected         -> d1 d2 + a fix drawn from p3
```

- **Rejection sampling keeps the output exact.** Draft token d_i, drawn from the draft's
  distribution q_i, is kept with probability min(1, p_i(d_i) / q_i(d_i)). At the first
  rejection, the replacement comes from norm(max(0, p_i − q_i)); if every draft survives, a
  bonus token comes from p_{k+1}. The result has exactly the target's distribution: a bad
  draft only costs speed. Greedy is the case of one-hot p and q: keep drafts while they match
  the target's top token.
- **Rejected tokens need no undo.** Their K/V sit past the sequence's `num_computed_tokens`, so
  the next step overwrites them. The scheduler reserves the k extra KV slots ("lookahead
  slots") up front.
- **One block table, two pools.** Draft and target each have a KV pool, indexed by the same
  block tables, so allocation, preemption and the prefix cache are shared. A block enters the
  prefix cache only once both models have computed it. The draft runs prompt chunks with the
  target and can lag a little during plain decodes, which it catches up on in its next pass.
- **Verification uses the decode path.** A short multi-token row (a token plus its drafts, or
  any chunk of at most 8 tokens) is attended as that many single queries, each seeing the keys
  up to its own position, through the paged decode kernel. That is exactly the causal mask.
- **Only small batches speculate** (`max_speculative_batch`, default 2). A big batch already
  shares each weight read between many sequences, so drafting is pure overhead there.

### int8 weight quantization

`--quantize int8` stores every projection matrix as int8 with one scale per output row:
`scale = max|W[n, :]| / 127`, `W_q = round(W / scale)`. Activations, norms, attention and the
KV cache stay bf16. On natural text, perplexity stays within about 1% of bf16 and the top
token matches at 94 to 97% of positions (0.6B, 1.7B, 4B).

The matmul dequantizes on the fly and picks its path by the number of activation rows M:

- **M ≤ 8** (decode, verification, small batches): the `int8_linear` kernel. Each simdgroup
  computes a few output rows, streams 8 weights per lane per step, dequantizes in registers
  and reuses each activation load across its rows. M is a template parameter, so the
  accumulators stay in registers. This reads the weights once, at 1 byte each: 1.4× (0.6B) to
  1.9× (4B) faster than bf16 at batch 1.
- **M ≤ 16:** the same kernel on 8-row chunks.
- **Larger** (prefill): one pass dequantizes the matrix to bf16, then a regular matmul runs.
  Compute dominates there, so prefill is only about 5% slower than bf16.

The tied lm_head keeps its bf16 matrix, which the embedding needs anyway, for large batches,
plus an int8 copy for small ones. int8 is a small-batch mode: at batch 32, decode runs at 0.6
to 0.75× of bf16, because this simple kernel doesn't use the GPU's matrix units.

### Sampling, templates, streaming

- **Sampling** is vectorized across the batch with per-row settings.
  - The pipeline is presence penalty, then top-k on the raw logits (one top-k over the widest k
    in the batch, masked per row), then temperature, softmax and top-p.
  - Temperature comes after top-k because dividing by a positive temperature doesn't change
    the ranking, so only k values per row are rescaled. A greedy row is simply its top 1.
  - The final draw is an exponential race, `argmax(p / E)` with `E ~ Exp(1)`. It needs no
    renormalization, and a seeded request draws its own noise from its own generator,
    independent of its batch neighbours.
- **Chat template:** `tokenizer_config.json`'s Jinja template is rendered with jinja2 using
  transformers' settings, so you see exactly what the model is fed. `enable_thinking=False`
  pre-fills an empty `<think></think>` block. Sampling defaults follow the Qwen3 model card:
  - thinking: temperature 0.6, top_p 0.95, top_k 20
  - non-thinking: temperature 0.7, top_p 0.8, top_k 20
- **Detokenization:** an emoji or a CJK character is often 2 to 4 byte-level BPE tokens. The
  incremental detokenizer decodes a short window of recent tokens and emits only new, complete
  text (the prefix/read-offset technique), so a half-finished character (U+FFFD) is never
  printed.

## Performance

Measured on a MacBook Pro with an M4 Pro (24 GB, 273 GB/s) running Qwen3-0.6B in bf16.

Decode at batch 1 runs at about 130 tok/s. The limit is about 229 tok/s, since every step must
read all 1.19 GB of weights. The matmuls themselves already run at 75 to 95% of memory
bandwidth; the rest of each 7.7 ms step is launch overhead, small kernels and sampling. Batching
is nearly free throughput: the weights are read once per step, however many sequences share it.

**Engine** (`bench/engine.py`: in-process, 128 output tokens, default sampling):

| | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| Decode, 100-token context | 127 tok/s | 919 tok/s | 2,082 tok/s |
| Decode, 4,000-token context | 103 tok/s | 274 tok/s | |

| Prefill (time to first token) | 1,000 tokens | 8,000 tokens |
|---|---|---|
| | 213 ms | 3,072 ms |

Prefill is compute-bound: matmuls and attention both run at about 5 TFLOPS, roughly 65% of peak.

**Bigger models, quantization and speculative decoding** (`bench/engine.py --chat`, batch 1,
128 tokens of a real chat reply):

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| bf16 | 133 tok/s | 28 tok/s |
| int8 | 184 tok/s | 50 tok/s |
| bf16 + 0.6B draft (k=3) | | 45 tok/s (58% of drafts accepted) |
| int8 + int8 0.6B draft (k=3) | | 64 tok/s |

At batch 4, int8 is 1.16× (0.6B) and 1.38× (4B) faster than bf16; at batch 8 they are even.
Speculation pays at batch 1 (1.6×) and 2 (1.1×), which is why it switches off above 2.

**HTTP** (`bench/load.py`: streaming chat, 128 output tokens per request):

| Concurrency | 1 | 4 | 16 | 32 |
|---|---|---|---|---|
| Aggregate decode tok/s | 132 | 440 | 1,299 | 2,025 |
| TTFT p50 (ms) | 12 | 33 | 93 | 108 |

**Optimization phase.** The same benchmark, run back to back before and after:

| | Before | After | Speedup |
|---|---|---|---|
| Decode, batch 1 | 69.5 tok/s | 127 tok/s | 1.8× |
| Decode, batch 8 | 362 tok/s | 919 tok/s | 2.5× |
| Decode, batch 32 | 576 tok/s | 2,082 tok/s | 3.6× |
| Decode, batch 1, 4,000-token context | 51 tok/s | 103 tok/s | 2.0× |
| Prefill, 1,000 tokens | 270 ms | 213 ms | 1.3× |

What it took, in order. The first profile showed decode spending most of each step on about
1,700 tiny kernels, not on the matmuls.

1. **Fused RMSNorm and stacked projections.** `F.rms_norm` (one kernel instead of eight) and one
   matmul each for q/k/v and gate/up.
2. **One batched attention call for all decodes,** instead of a per-sequence loop.
3. **Sliced LM head.** MPS picks a slow kernel for the 151,936-wide matmul once there are more
   than a few rows: 7.7 ms at 32 rows, versus 2.2 ms when sliced into 32k-row pieces.
4. **Metal kernels:** fused q/k-norm + RoPE + KV write, then paged decode attention.
5. **Top-k kernel and leaner sampler.** `torch.topk` took 3.6 ms per step at batch 32.
6. **One asynchronous upload per step.** `torch.tensor(list, device="mps")` costs about 140 µs
   each.
7. **Fused residual adds** (`addmm`) and the SiLU·mul kernel.

Tried and dropped: `torch.compile`, which won at batch 8 but not at batch 1 or 32, slowed
prefill, and doubled startup time.

**Effect of each serving technique** (measured when it was added):

| Technique | Without | With |
|---|---|---|
| KV cache (decode at a 1,000-token context) | 3.6 tok/s | 74 tok/s |
| Chunked prefill (8 streams, then an 8k-token prompt arrives): TTFT p99 of queued requests | ~4.9 s | ~95 ms |
| Prefix caching (chat turn 2 on a ~4k-token document, 30 output tokens) | 1,944 ms | 616 ms |

With chunked prefill, the longest gap between tokens of a running stream while that 8k prompt
prefills is about 300 ms at the default 512-token budget, versus about 1,150 ms at 2,048.

**Known limit:** identical greedy requests can produce different text in bf16 when they're
batched differently, because different matmul shapes round differently. In fp32 they are
identical, which the invariance tests check.

```sh
uv run python bench/engine.py --profile                   # per-phase decode step breakdown
uv run python bench/engine.py --batch 1,8 --context 4000  # long-context decode
uv run atomserve Qwen/Qwen3-0.6B --no-repl &
uv run python bench/load.py --concurrency 1,4,16,32
uv run python bench/load.py --concurrency 8 --long-prompt-tokens 8000
```

## Development

```sh
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run pytest                                         # fast tests, about 1 s (CPU, plus kernels on MPS)
uv run pytest -m slow                                 # real model on MPS, about 20 s
```

The tests prove behavior; they don't just exercise code paths:

- **Golden:**
  - logits match Hugging Face `transformers` (fp32, max diff ~1e-4)
  - 32 greedy tokens match `transformers.generate`
  - the chat template matches `apply_chat_template` exactly
- **Invariance** (outputs must not change; checked in fp32):
  - cached vs. recomputed from scratch
  - batched vs. alone
  - scattered vs. contiguous blocks
  - chunked vs. whole prefill
  - prefix-cached vs. cold
  - preempted vs. never preempted
- **Metal kernels:** each one matches the PyTorch math in fp32 (long positions, multi-partition
  contexts, several GQA ratios, ties for top-k, every int8 row count), and the engine's tokens
  are identical with kernels on and off, quantized or not.
- **Speculative decoding:**
  - Greedy output equals the target alone, token for token. This holds with a deliberately
    bad draft, with chunked prefill, preemption, prefix caching and speculation pausing and
    resuming. It also holds for real models: Qwen3-1.7B verifying 0.6B drafts in fp32.
  - Rejection sampling reproduces the target's distribution, checked statistically with a draft
    that nearly reverses it.
- **Quantization:** rounding error is bounded per row, and the real model's perplexity stays
  within 3% of bf16.
- **Scheduler, without a GPU:** a fake runner simulates the KV pool slot by slot and predicts the
  next token from what it reads back. Sharing a block wrongly, reusing a stale or partial
  prefix, or forgetting to recompute after preemption all change the outputs and fail the
  tests. Deliberately broken versions of these rules are confirmed to fail.
- **End to end:**
  - the real `atomserve` binary driven through a pseudo-terminal: multi-turn chat, Ctrl-C, Ctrl-D
  - a live HTTP server tested with raw sockets (SSE wire format, client disconnect) and the
    official `openai` client

`transformers` and `openai` are development dependencies only; the engine never imports them.

## Not yet

- **An int8 matmul on the GPU's matrix units** (`simdgroup_matrix`), so int8 also wins at large
  batches.
- **int4 weights** (group-wise scales), for models that don't fit in bf16 or int8.
- **Overlapping step N+1's scheduling with step N's GPU work.** The GPU idles only ~0.3 ms per
  step now, so this is a small win.
