# The Qwen3 transformer, written as plain functions over plain tensors.
#
# Input is a *flat batch*: the new tokens of several sequences concatenated
# into one list of T tokens, with no batch dimension and no padding (see
# `Batch`). Every matmul runs on all T rows at once without caring which
# sequence a row belongs to; attention is the only step that has to know
# where each sequence starts and ends. Output is logits only for the rows we
# sample from: the lm_head is a 151,936 x 1024 matmul, far too expensive to
# run on every prompt token.
#
# One decoder layer, pre-norm style:
#     x = x + attention(rmsnorm(x))
#     x = x + mlp(rmsnorm(x))
#
# Weights keep the Hugging Face layout [out_features, in_features], so a linear
# layer is `F.linear(x, W)` == `x @ W.T`.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from atomserve import kernels
from atomserve.config import ModelConfig
from atomserve.kv_cache import KVPool
from atomserve.quant import Int8Weight, linear

Linear = torch.Tensor | Int8Weight  # a projection matrix, plain or quantized


@dataclass
class LayerWeights:
    """Parameters of one decoder layer. Shapes shown for Qwen3-0.6B."""

    input_norm: torch.Tensor  # [1024]                 RMSNorm before attention
    qkv_proj: Linear  # [(16+8+8)*128, 1024]           hidden -> all q, k and v heads
    q_norm: torch.Tensor  # [128]                      RMSNorm applied to each query head
    k_norm: torch.Tensor  # [128]                      RMSNorm applied to each key head
    o_proj: Linear  # [1024, 16*128]                   concatenated heads -> hidden
    post_attn_norm: torch.Tensor  # [1024]             RMSNorm before the MLP
    gate_up_proj: Linear  # [2*3072, 1024]             SwiGLU gate and value branches
    down_proj: Linear  # [1024, 3072]                  back to hidden size

    # The checkpoint stores q/k/v (and gate/up) as separate matrices; the loader
    # stacks them so each is one matmul. Same math, fewer kernel launches: at
    # decode, every launch has a fixed cost comparable to the matmul itself.
    # With --quantize int8 the four projections are Int8Weight (quant.py).


@dataclass
class Weights:
    """All parameters of the model, already on the target device and dtype."""

    embed: torch.Tensor  # [vocab, hidden]  token id -> vector
    layers: list[LayerWeights]
    final_norm: torch.Tensor  # [hidden]
    lm_head: torch.Tensor  # [vocab, hidden]  same tensor as `embed` when tied
    lm_head_int8: Int8Weight | None = None  # quantized copy for small batches (--quantize)

    @property
    def nbytes(self) -> int:
        """Memory of all distinct parameter tensors (a tied lm_head counts once)."""
        tensors = {id(t): t for t in (self.embed, self.final_norm, self.lm_head)}
        quantized = [self.lm_head_int8] if self.lm_head_int8 else []
        for layer in self.layers:
            for t in vars(layer).values():
                if isinstance(t, Int8Weight):
                    quantized.append(t)
                else:
                    tensors[id(t)] = t
        plain = sum(t.numel() * t.element_size() for t in tensors.values())
        return plain + sum(q.nbytes for q in quantized)


@dataclass
class PrefillSpan:
    """One sequence with several new tokens (a prompt or prompt chunk) in a Batch."""

    rows: slice  # its rows in the flat batch
    context_slots: torch.Tensor  # [context_len] pool slot of each position 0..context_len-1


@dataclass
class Batch:
    """One forward pass worth of tokens from B sequences, flattened. Built by the runner.

    input_ids:    [T]  new tokens of all sequences, concatenated
    positions:    [T]  absolute position of each token (drives RoPE and the causal mask)
    slot_mapping: [T]  pool slot each token's K/V is written to
    logits_idx:   [B'] rows that need a next-token prediction (a sequence's last row,
                  once its prompt is fully processed)

    For attention, the sequences are split by how many new tokens they bring:
    prefills:     sequences with several (prompt chunks), attended one at a time;
                  few per step, and each is a large piece of work on its own
    decode_rows:  [Bd] rows of the sequences with exactly one (decodes), or None

    All decodes share one attention call, so a step's launch cost doesn't grow
    with the batch size. That call takes its keys in one of two forms:
    use_kernels=True (Metal kernels, kernels.py): read in place from the pool
      decode_block_tables: [Bd, max_blocks]  each decode's block table, padded
      decode_lens:         [Bd]              its context length
      decode_max_len:      host int          the longest context
    use_kernels=False (plain PyTorch): gathered into a padded tensor first
      decode_slots: [Bd, L] pool slots of each decode's context, padded to the longest
      decode_mask:  [Bd, L] True where decode_slots is a real key; None if no padding
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    logits_idx: torch.Tensor
    prefills: list[PrefillSpan]
    use_kernels: bool = False
    decode_rows: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    decode_lens: torch.Tensor | None = None
    decode_max_len: int = 0
    decode_slots: torch.Tensor | None = None
    decode_mask: torch.Tensor | None = None
    host_buffer: torch.Tensor | None = None  # source of the async upload; see runner._Upload


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square normalisation over the last dimension.

    x:      [..., D]
    weight: [D]   learned per-channel scale
    returns [..., D] in x's dtype

    Computes  weight * x / sqrt(mean(x^2) + eps): rescale each vector to unit RMS,
    then apply the learned scale. Unlike LayerNorm there is no mean subtraction
    and no bias. PyTorch's fused kernel does this in one launch (accumulating in
    fp32) where the longhand version takes eight; in bf16 it can differ from the
    reference by one rounding step, since the reference rounds before the weight
    multiply as well as after.
    """
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def rope_inv_freq(head_dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """RoPE frequency of each of the head_dim/2 coordinate pairs: theta^(-2i/head_dim).

    returns [head_dim/2] fp32
    """
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    return 1.0 / (theta**exponents)


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation angles for Rotary Position Embedding (RoPE).

    positions: [T]  absolute position of each token (int)
    returns (cos, sin), each [T, head_dim] in `dtype`

    Each head vector is treated as head_dim/2 pairs of coordinates; pair i is
    rotated by angle `position * theta^(-2i/head_dim)`. Low pairs spin fast (fine
    position), high pairs spin slowly (coarse position). The angle table is
    duplicated to full head_dim because `apply_rope` pairs dimension i with
    dimension i + head_dim/2 (the "rotate half" layout used by Qwen/Llama).
    Computed in fp32: large positions times small frequencies need the precision.
    """
    inv_freq = rope_inv_freq(head_dim, theta, positions.device)
    angles = positions.float()[:, None] * inv_freq[None, :]  # [T, head_dim/2]
    angles = torch.cat([angles, angles], dim=-1)  # [T, head_dim]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate query or key heads by their token's position.

    x:        [T, H, head_dim]
    cos, sin: [T, head_dim]   from `rope_cos_sin`
    returns   [T, H, head_dim]

    For each pair (a, b) = (x[i], x[i + head_dim/2]) this computes the 2-D rotation
    (a cos - b sin, b cos + a sin). Because q and k are both rotated by their own
    positions, the dot product q_m . k_n depends only on the distance m - n.
    """
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_positions: torch.Tensor
) -> torch.Tensor:
    """Causal scaled dot-product attention of new queries over all keys of one sequence.

    q:           [T, num_heads, head_dim]     queries for the T new tokens
    k, v:        [S, num_kv_heads, head_dim]  keys/values for positions 0..S-1 (S >= T)
    q_positions: [T]                          absolute position of each query
    returns      [T, num_heads, head_dim]

    A query at position p may attend to keys at positions 0..p. For a fresh
    sequence S == T and this is the familiar lower-triangular mask; with cached
    tokens the new queries sit at the *end* of the sequence and see the whole
    prefix. A single decode query (T == 1, p == S-1) sees everything, so no mask
    is built. SDPA's own `is_causal=True` would be wrong here: it aligns the
    triangle to the top-left, as if the queries started at position 0.

    With grouped-query attention (GQA) several query heads share one key/value
    head (16 query heads over 8 kv heads for 0.6B); `enable_gqa` does that sharing
    without copying K/V. The softmax scale is the standard 1/sqrt(head_dim).
    """
    S = k.shape[0]
    mask = None
    if q.shape[0] > 1:
        key_positions = torch.arange(S, device=q.device)
        mask = key_positions[None, :] <= q_positions[:, None]  # [T, S], True = may attend
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1),  # SDPA wants [heads, T, head_dim]
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=mask,
        enable_gqa=True,
    )
    return out.transpose(0, 1)


def paged_attention(q: torch.Tensor, pool: KVPool, layer_idx: int, batch: Batch) -> torch.Tensor:
    """Attention for every sequence of a flat batch, reading K/V from the paged pool.

    q:       [T, num_heads, head_dim]  queries for all new tokens of the batch
    returns  [T, num_heads, head_dim]

    The one place the model looks at sequence boundaries: each sequence's
    queries may only see its own keys. Prompt chunks gather their K/V out of
    the blocks and use SDPA; decodes go through one batched call.
    """
    if batch.decode_rows is not None:
        q_dec = q if not batch.prefills else q.index_select(0, batch.decode_rows)
        k_slots, v_slots = pool.layer_slots(layer_idx)
        if batch.use_kernels:
            decoded = kernels.paged_decode_attention(
                q_dec, k_slots, v_slots, batch.decode_block_tables, batch.decode_lens,
                pool.block_size, k_slots.shape[1], batch.decode_max_len,
            )  # fmt: skip
        else:
            Bd, L = batch.decode_slots.shape
            k, v = pool.read(layer_idx, batch.decode_slots.view(-1))
            k, v = k.view(Bd, L, *k.shape[1:]), v.view(Bd, L, *v.shape[1:])
            decoded = decode_attention(q_dec, k, v, batch.decode_mask)
        if not batch.prefills:  # a pure decode step: every row, in order
            return decoded
    out = torch.empty_like(q)
    for span in batch.prefills:
        k, v = pool.read(layer_idx, span.context_slots)  # [context_len, num_kv_heads, head_dim]
        out[span.rows] = attention(q[span.rows], k, v, batch.positions[span.rows])
    if batch.decode_rows is not None:
        out.index_copy_(0, batch.decode_rows, decoded)
    return out


def decode_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask: torch.Tensor
) -> torch.Tensor:
    """One new query per sequence, for a batch of sequences, in a single SDPA call.

    q:        [Bd, num_heads, head_dim]        each sequence's newest query
    k, v:     [Bd, L, num_kv_heads, head_dim]  each sequence's keys/values, padded to L
    key_mask: [Bd, L] or None                  False for padding; None if there is none
    returns   [Bd, num_heads, head_dim]

    A decode query is the last position, so it may see its whole context: the
    only masking needed is for padding.
    """
    out = F.scaled_dot_product_attention(
        q[:, :, None, :],  # [Bd, heads, 1, head_dim]
        k.transpose(1, 2),  # [Bd, kv_heads, L, head_dim]
        v.transpose(1, 2),
        attn_mask=None if key_mask is None else key_mask[:, None, None, :],
        enable_gqa=True,
    )
    return out[:, :, 0, :]


def attention_block(
    x: torch.Tensor,
    residual: torch.Tensor,
    layer: LayerWeights,
    layer_idx: int,
    rope: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    config: ModelConfig,
    batch: Batch,
    pool: KVPool,
) -> torch.Tensor:
    """The attention half of a decoder layer.

    x:        [T, hidden]  the input after the layer's first RMSNorm
    residual: [T, hidden]  the residual stream, before that norm
    rope:     (cos, sin) tables for the PyTorch path, or the frequencies for the kernel
    returns   [T, hidden]  residual + attention output (the add is fused into o_proj)

    Project to per-head q/k/v, RMSNorm each q and k head (Qwen3's addition over
    Llama: keeps attention logits well-scaled), rotate q and k by position,
    store the new k/v in the pool, attend, then mix the heads back to the
    hidden size with o_proj. Writing before attending means each new token sees
    its own key, and past tokens are never re-projected.
    """
    T, H, KV = x.shape[0], config.num_heads, config.num_kv_heads
    qkv = linear(x, layer.qkv_proj, use_kernels=batch.use_kernels).view(T, H + 2 * KV, -1)

    if batch.use_kernels:  # everything up to the pool write, fused (kernels.py)
        k_slots, v_slots = pool.layer_slots(layer_idx)
        q = kernels.qk_norm_rope_store(
            qkv, layer.q_norm, layer.k_norm, batch.positions, batch.slot_mapping, rope,
            k_slots, v_slots, H, KV, config.rms_norm_eps,
        )  # fmt: skip
    else:
        q, k, v = qkv.split([H, KV, KV], dim=1)
        q = rms_norm(q, layer.q_norm, config.rms_norm_eps)
        k = rms_norm(k, layer.k_norm, config.rms_norm_eps)
        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)
        pool.write(layer_idx, k, v, batch.slot_mapping)

    out = paged_attention(q, pool, layer_idx, batch)  # [T, num_heads, head_dim]
    return linear(out.reshape(T, H * config.head_dim), layer.o_proj, residual, batch.use_kernels)


def mlp_block(
    x: torch.Tensor, residual: torch.Tensor, layer: LayerWeights, use_kernels: bool = False
) -> torch.Tensor:
    """SwiGLU feed-forward network: residual + down( silu(gate(x)) * up(x) ).

    x:        [T, hidden]  the input after the layer's second RMSNorm
    residual: [T, hidden]  the residual stream, before that norm
    returns   [T, hidden]

    The gate branch, passed through SiLU, decides per channel how much of the up
    branch to let through. Applied to every token independently. The residual
    add is fused into the down projection: addmm(residual, a, W^T) is one kernel
    where `residual + a @ W^T` is two, and at decode a launch costs about as
    much as the add itself. With use_kernels, silu(gate) * up is one Metal kernel.
    """
    gate_up = linear(x, layer.gate_up_proj, use_kernels=use_kernels)  # [T, 2 * intermediate]
    if use_kernels:
        hidden = kernels.silu_mul(gate_up)
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
    return linear(hidden, layer.down_proj, residual, use_kernels)


def forward(weights: Weights, config: ModelConfig, batch: Batch, pool: KVPool) -> torch.Tensor:
    """Run the decoder stack over a flat batch; return next-token logits.

    returns [len(batch.logits_idx), vocab]  fp32 logits, one row per sampling sequence

    Side effect: the K/V of every token in the batch are written to `pool`, so
    the next call only has to pass tokens not yet seen. That makes the same
    function serve prefill (a whole prompt), decode (one token) and any mix of
    the two across sequences.
    """
    x = weights.embed[batch.input_ids]  # [T, hidden]
    if batch.use_kernels:  # the kernel computes the angles itself
        rope = rope_inv_freq(config.head_dim, config.rope_theta, x.device)
    else:
        rope = rope_cos_sin(batch.positions, config.head_dim, config.rope_theta, x.dtype)
    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, config.rms_norm_eps)
        x = attention_block(h, x, layer, i, rope, config, batch, pool)  # x + attention
        h = rms_norm(x, layer.post_attn_norm, config.rms_norm_eps)
        x = mlp_block(h, x, layer, batch.use_kernels)  # x + mlp
    x = x[batch.logits_idx]  # drop rows nobody samples from before the expensive part
    hidden = rms_norm(x, weights.final_norm, config.rms_norm_eps)
    return compute_logits(weights, hidden, batch.use_kernels)


def compute_logits(
    weights: Weights, hidden: torch.Tensor, use_kernels: bool = False
) -> torch.Tensor:
    """Project hidden states to next-token scores over the vocabulary.

    hidden: [N, hidden]  only the rows we want predictions for
    returns [N, vocab]   fp32 logits

    Returned in fp32 so sampling (softmax, top-p cumulative sums) is precise.

    The vocabulary is projected in slices of LM_HEAD_SLICE rows: MPS picks a
    slow kernel for one very wide matmul once there are more than a few rows
    (32 rows: 7.7 ms whole vs 2.2 ms sliced, on an M4 Pro), and slicing costs
    nothing for a single row. With a quantized copy, a few rows use the int8 kernel
    instead (half the bytes); more rows use the bf16 matrix, which a tied model
    keeps anyway for the embedding lookup.
    """
    small = hidden.shape[0] <= kernels.INT8_MAX_ROWS
    if weights.lm_head_int8 is not None and small:
        return linear(hidden, weights.lm_head_int8, use_kernels=use_kernels).float()
    slices = weights.lm_head.split(LM_HEAD_SLICE)
    return torch.cat([F.linear(hidden, w) for w in slices], dim=-1).float()


LM_HEAD_SLICE = 32768
