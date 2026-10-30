# Weight-only int8 quantization: each big matrix stored as int8 plus one scale per row.
#
# At decode, a step's time is mostly reading weights (1.19 GB per step for
# Qwen3-0.6B in bf16, 8 GB for Qwen3-4B), so storing them in half the bytes
# makes small-batch decode up to ~2x faster. Activations, norms, attention and
# the KV cache stay in bf16; only the projection matrices are quantized.
#
# Per output row n:  scale[n] = max_k |W[n, k]| / 127,  W_q[n, k] = round(W[n, k] / scale[n])
#
# so each row uses the whole int8 range and its rounding error is at most half a
# step of its own scale. Symmetric, per-row (per-output-channel) int8 is close
# to lossless for LLM weights; tests check the logits stay close to bf16.
#
# The matmul dequantizes on the fly, choosing by the number of activation rows M:
#   M <= 8   the Metal kernel (kernels.int8_linear): decode, speculative
#            verification, small batches. Weights read once, at 1 byte each.
#   M <= 16  the same kernel on 8-row chunks: weights re-read per chunk, still
#            no dequantized copy. Around bf16 speed.
#   bigger   (prefill) dequantize the matrix to bf16 and use a regular matmul;
#            compute dominates there, so the extra pass costs only a few percent.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from atomserve import kernels

CHUNKED_MAX_ROWS = 16  # up to here, the int8 kernel in 8-row chunks beats dequantizing


@dataclass
class Int8Weight:
    """A [N, K] weight matrix as int8 values and one fp32 scale per output row."""

    q: torch.Tensor  # [N, K] int8
    scale: torch.Tensor  # [N] fp32
    dtype: torch.dtype  # the dtype it replaces (activations and outputs stay in it)

    def dequantize(self) -> torch.Tensor:
        """Back to a regular [N, K] matrix in `dtype` (an approximation of the original)."""
        return (self.q.to(torch.float32) * self.scale[:, None]).to(self.dtype)

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.scale.numel() * self.scale.element_size()


def quantize_int8(w: torch.Tensor) -> Int8Weight:
    """Symmetric per-row int8 quantization of a [N, K] weight matrix."""
    w32 = w.float()
    scale = w32.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = (w32 / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return Int8Weight(q, scale, w.dtype)


def linear(
    x: torch.Tensor,
    w: torch.Tensor | Int8Weight,
    residual: torch.Tensor | None = None,
    use_kernels: bool = False,
) -> torch.Tensor:
    """x @ w^T, plus `residual` if given, for a regular or an int8 weight.

    x: [M, K]  w: [N, K]  residual: [M, N]  ->  [M, N]

    The residual add is fused into the matmul either way (addmm, or the kernel's
    epilogue): at decode a separate add costs about as much as the add itself.
    """
    if isinstance(w, Int8Weight):
        rows, step = x.shape[0], kernels.INT8_MAX_ROWS
        if use_kernels and w.q.shape[1] % 256 == 0 and rows <= CHUNKED_MAX_ROWS:
            if rows <= step:
                return kernels.int8_linear(x, w.q, w.scale, residual)
            chunks = range(0, rows, step)
            residuals = [None if residual is None else residual[i : i + step] for i in chunks]
            return torch.cat([
                kernels.int8_linear(x[i : i + step], w.q, w.scale, r)
                for i, r in zip(chunks, residuals, strict=True)
            ])  # fmt: skip
        w = kernels.dequantize_int8(w.q, w.scale, w.dtype) if use_kernels else w.dequantize()
    if residual is None:
        return F.linear(x, w)
    return torch.addmm(residual, x, w.t())
