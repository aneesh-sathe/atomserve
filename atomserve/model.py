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

from atomserve.config import ModelConfig
from atomserve.kv_cache import KVPool


@dataclass
class LayerWeights:
    """Parameters of one decoder layer. Shapes shown for Qwen3-0.6B."""

    input_norm: torch.Tensor  # [1024]         RMSNorm before attention
    q_proj: torch.Tensor  # [16*128, 1024]     hidden -> all query heads
    k_proj: torch.Tensor  # [8*128, 1024]      hidden -> all key heads
    v_proj: torch.Tensor  # [8*128, 1024]      hidden -> all value heads
    q_norm: torch.Tensor  # [128]              RMSNorm applied to each query head
    k_norm: torch.Tensor  # [128]              RMSNorm applied to each key head
    o_proj: torch.Tensor  # [1024, 16*128]     concatenated heads -> hidden
    post_attn_norm: torch.Tensor  # [1024]     RMSNorm before the MLP
    gate_proj: torch.Tensor  # [3072, 1024]    SwiGLU gate branch
    up_proj: torch.Tensor  # [3072, 1024]      SwiGLU value branch
    down_proj: torch.Tensor  # [1024, 3072]    back to hidden size


@dataclass
class Weights:
    """All parameters of the model, already on the target device and dtype."""

    embed: torch.Tensor  # [vocab, hidden]  token id -> vector
    layers: list[LayerWeights]
    final_norm: torch.Tensor  # [hidden]
    lm_head: torch.Tensor  # [vocab, hidden]  same tensor as `embed` when tied


@dataclass
class Batch:
    """One forward pass worth of tokens from B sequences, flattened. Built by the runner.

    input_ids:     [T]  new tokens of all sequences, concatenated
    positions:     [T]  absolute position of each token (drives RoPE and the causal mask)
    slot_mapping:  [T]  pool slot each token's K/V is written to
    cu_seqlens:    B+1 host ints; sequence i's tokens are rows cu_seqlens[i]:cu_seqlens[i+1]
    context_slots: [sum of context lengths]  pool slots of every token each sequence
                   attends to (cached prefix + new tokens), sequences concatenated
    cu_ctxlens:    B+1 host ints; sequence i's context is context_slots[cu_ctxlens[i]:...]
    logits_idx:    [B']  rows that need a next-token prediction (a sequence's last row,
                   once its prompt is fully processed)

    The offsets are host ints because attention loops over sequences in Python
    and slices with them; reading them off the GPU would force a sync per layer.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    cu_seqlens: list[int]
    context_slots: torch.Tensor
    cu_ctxlens: list[int]
    logits_idx: torch.Tensor


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square normalisation over the last dimension.

    x:      [..., D]
    weight: [D]   learned per-channel scale
    returns [..., D] in x's dtype

    Rescales each vector to unit RMS, then applies the learned scale. Unlike
    LayerNorm there is no mean subtraction and no bias. The reduction runs in
    fp32 because summing squares in bf16 loses precision; the result is cast back
    *before* multiplying by the weight, matching the reference implementation.
    """
    input_dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x32.to(input_dtype)


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
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    )
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
    queries may only see its own keys. This version loops over sequences,
    gathering each one's K/V out of its blocks into a contiguous tensor; correct
    and obvious, and the first thing the optimization phase replaces.
    """
    out = torch.empty_like(q)
    for i in range(len(batch.cu_seqlens) - 1):
        rows = slice(batch.cu_seqlens[i], batch.cu_seqlens[i + 1])
        slots = batch.context_slots[batch.cu_ctxlens[i] : batch.cu_ctxlens[i + 1]]
        k, v = pool.read(layer_idx, slots)  # [context_len, num_kv_heads, head_dim]
        out[rows] = attention(q[rows], k, v, batch.positions[rows])
    return out


def attention_block(
    x: torch.Tensor,
    layer: LayerWeights,
    layer_idx: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
    batch: Batch,
    pool: KVPool,
) -> torch.Tensor:
    """The attention half of a decoder layer (after its input RMSNorm).

    x: [T, hidden]  ->  returns [T, hidden]

    Project to per-head q/k/v, RMSNorm each q and k head (Qwen3's addition over
    Llama: keeps attention logits well-scaled), rotate q and k by position,
    store the new k/v in the pool, attend, then mix the heads back to the
    hidden size with o_proj. Writing before attending means each new token sees
    its own key, and past tokens are never re-projected.
    """
    T = x.shape[0]
    q = F.linear(x, layer.q_proj).view(T, config.num_heads, config.head_dim)
    k = F.linear(x, layer.k_proj).view(T, config.num_kv_heads, config.head_dim)
    v = F.linear(x, layer.v_proj).view(T, config.num_kv_heads, config.head_dim)

    q = rms_norm(q, layer.q_norm, config.rms_norm_eps)
    k = rms_norm(k, layer.k_norm, config.rms_norm_eps)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    pool.write(layer_idx, k, v, batch.slot_mapping)
    out = paged_attention(q, pool, layer_idx, batch)  # [T, num_heads, head_dim]
    return F.linear(out.reshape(T, config.num_heads * config.head_dim), layer.o_proj)


def mlp_block(x: torch.Tensor, layer: LayerWeights) -> torch.Tensor:
    """SwiGLU feed-forward network: down( silu(gate(x)) * up(x) ).

    x: [T, hidden]  ->  returns [T, hidden]

    The gate branch, passed through SiLU, decides per channel how much of the up
    branch to let through. Applied to every token independently.
    """
    return F.linear(
        F.silu(F.linear(x, layer.gate_proj)) * F.linear(x, layer.up_proj), layer.down_proj
    )


def forward(weights: Weights, config: ModelConfig, batch: Batch, pool: KVPool) -> torch.Tensor:
    """Run the decoder stack over a flat batch; return next-token logits.

    returns [len(batch.logits_idx), vocab]  fp32 logits, one row per sampling sequence

    Side effect: the K/V of every token in the batch are written to `pool`, so
    the next call only has to pass tokens not yet seen. That makes the same
    function serve prefill (a whole prompt), decode (one token) and any mix of
    the two across sequences.
    """
    x = weights.embed[batch.input_ids]  # [T, hidden]
    cos, sin = rope_cos_sin(batch.positions, config.head_dim, config.rope_theta, x.dtype)
    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, config.rms_norm_eps)
        x = x + attention_block(h, layer, i, cos, sin, config, batch, pool)
        x = x + mlp_block(rms_norm(x, layer.post_attn_norm, config.rms_norm_eps), layer)
    x = x[batch.logits_idx]  # drop rows nobody samples from before the expensive part
    return compute_logits(weights, rms_norm(x, weights.final_norm, config.rms_norm_eps))


def compute_logits(weights: Weights, hidden: torch.Tensor) -> torch.Tensor:
    """Project hidden states to next-token scores over the vocabulary.

    hidden: [N, hidden]  only the rows we want predictions for
    returns [N, vocab]   fp32 logits

    Returned in fp32 so sampling (softmax, top-p cumulative sums) is precise.
    """
    return F.linear(hidden, weights.lm_head).float()
