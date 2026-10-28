# The Qwen3 transformer, written as plain functions over plain tensors.
#
# Input is a *flat* list of T tokens (no batch dimension) and their positions.
# Output is one hidden vector per token; `compute_logits` turns the rows we
# actually sample from into vocabulary scores. Keeping those two steps apart
# matters: the lm_head is a 151,936 x 1024 matmul, so we only run it on rows
# that need a next-token prediction.
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


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention of one sequence over itself, causally masked.

    q: [T, num_heads, head_dim]
    k: [T, num_kv_heads, head_dim]
    v: [T, num_kv_heads, head_dim]
    returns [T, num_heads, head_dim]

    Token t may only attend to tokens 0..t. With grouped-query attention (GQA)
    several query heads share one key/value head (16 query heads over 8 kv heads
    for 0.6B); `enable_gqa` does that sharing without copying K/V. The softmax
    scale is the standard 1/sqrt(head_dim).
    """
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1),  # SDPA wants [heads, T, head_dim]
        k.transpose(0, 1),
        v.transpose(0, 1),
        is_causal=True,
        enable_gqa=True,
    )
    return out.transpose(0, 1)


def attention_block(
    x: torch.Tensor,
    layer: LayerWeights,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
) -> torch.Tensor:
    """The attention half of a decoder layer (after its input RMSNorm).

    x: [T, hidden]  ->  returns [T, hidden]

    Project to per-head q/k/v, RMSNorm each q and k head (Qwen3's addition over
    Llama: keeps attention logits well-scaled), rotate q and k by position,
    attend, then mix the heads back to the hidden size with o_proj.
    """
    T = x.shape[0]
    q = F.linear(x, layer.q_proj).view(T, config.num_heads, config.head_dim)
    k = F.linear(x, layer.k_proj).view(T, config.num_kv_heads, config.head_dim)
    v = F.linear(x, layer.v_proj).view(T, config.num_kv_heads, config.head_dim)

    q = rms_norm(q, layer.q_norm, config.rms_norm_eps)
    k = rms_norm(k, layer.k_norm, config.rms_norm_eps)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    out = causal_attention(q, k, v)  # [T, num_heads, head_dim]
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


def forward(
    weights: Weights, config: ModelConfig, input_ids: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Run the whole decoder stack over one sequence, with no cache.

    input_ids: [T]  token ids
    positions: [T]  absolute position of each token (0..T-1 for a fresh sequence)
    returns    [T, hidden]  final-normed hidden state per token

    Every call recomputes attention over the full sequence, so generating N
    tokens costs O(N^2) work. The KV cache (next milestone) removes that.
    """
    x = weights.embed[input_ids]  # [T, hidden]
    cos, sin = rope_cos_sin(positions, config.head_dim, config.rope_theta, x.dtype)
    for layer in weights.layers:
        x = x + attention_block(
            rms_norm(x, layer.input_norm, config.rms_norm_eps), layer, cos, sin, config
        )
        x = x + mlp_block(rms_norm(x, layer.post_attn_norm, config.rms_norm_eps), layer)
    return rms_norm(x, weights.final_norm, config.rms_norm_eps)


def compute_logits(weights: Weights, hidden: torch.Tensor) -> torch.Tensor:
    """Project hidden states to next-token scores over the vocabulary.

    hidden: [N, hidden]  only the rows we want predictions for
    returns [N, vocab]   fp32 logits

    Returned in fp32 so sampling (softmax, top-p cumulative sums) is precise.
    """
    return F.linear(hidden, weights.lm_head).float()
