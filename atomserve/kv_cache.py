# The paged KV cache: remembered keys and values of every token of every sequence.
#
# At every layer, attention needs the keys and values of *all* previous tokens.
# Those never change once computed (a token's K/V depend only on itself and
# the tokens before it), so we store them instead of recomputing them. That
# turns generation from O(N^2) work into O(N): the prompt is processed once
# ("prefill"), then each new token costs one small forward pass ("decode").
#
# Storage is one preallocated pool shared by all sequences, split into blocks
# of `block_size` tokens (pages, in OS terms). A sequence's tokens live in the
# blocks listed in its block table, in order, so token at position p sits at
#
#     slot = block_table[p // block_size] * block_size + p % block_size
#
# in the pool viewed as a flat array of slots. Which blocks are free is tracked
# separately by block_allocator.py; this file only holds the tensors.
#
# With quantized=True the pool stores int8 instead of the model's dtype, plus
# one fp32 scale per (token, kv head): each 128-value head vector v becomes
#
#     scale = max|v| / 127,   v_q = round(v / scale)
#
# so 128 + 4 bytes instead of 256. About twice the tokens fit in the same
# memory, and decode attention, which reads every cached key and value at every
# step, moves half the bytes. The error stays local: a key's rounding depends
# only on that one key, never on other tokens.
#
# Keys need one more step. A few of their 128 channels are consistently much
# larger than the rest (outlier channels), so max|k| / 127 is a coarse step for
# everything else: Qwen3-0.6B's perplexity rose 5% with plain int8 keys. So keys
# are stored rotated by a Hadamard matrix H (orthonormal, entries +-1/sqrt(128)),
# which mixes every channel into all 128 and leaves no outliers to waste the
# range on (as in QuaRot, Ashkboos et al. 2024). Queries get the same rotation,
# so attention scores are unchanged: (Hq) . (Hk) = q . k. With that, an int8
# cache is within noise of bf16. Values have no such outliers and are stored as is.

from dataclasses import dataclass

import torch

from atomserve import kernels
from atomserve.config import ModelConfig


def quantize_heads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization of each head vector (the last dimension).

    x: [T, num_kv_heads, head_dim]
    returns (int8 [T, num_kv_heads, head_dim], fp32 scale [T, num_kv_heads])

    Multiplies by 127 / max|v| rather than dividing by the scale, exactly as the
    Metal kernel does, so both paths store the same integers. An all-zero vector
    gets scale 0 (and stays zero) instead of dividing by zero.
    """
    x32 = x.float()
    amax = x32.abs().amax(dim=-1)
    inv = torch.where(amax > 0, 127.0 / amax, 0.0)
    q = (x32 * inv[..., None]).round().clamp(-127, 127).to(torch.int8)
    return q, amax / 127.0


def hadamard(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The orthonormal n x n Hadamard matrix (n a power of 2), in Sylvester order.

    returns [n, n] with H[i, j] = (-1)^popcount(i & j) / sqrt(n); symmetric, so H = H^T = H^-1.
    """
    if n & (n - 1):
        raise ValueError(f"a Hadamard rotation needs a power-of-2 size, got {n}")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)])
    return (h / n**0.5).to(device, dtype)


@dataclass
class LayerSlots:
    """One layer's pool as flat slot arrays: views, so writes through them land in the pool.

    k, v:             [num_blocks * block_size, num_kv_heads, head_dim]  model dtype or int8
    k_scale, v_scale: [num_blocks * block_size, num_kv_heads] fp32, or None if not quantized
    """

    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None


class KVPool:
    """Preallocated key/value storage for all sequences, all layers.

    k, v:             [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    k_scale, v_scale: [num_layers, num_blocks, block_size, num_kv_heads], if quantized
    key_rotation:     [head_dim, head_dim] fp32 Hadamard matrix, if quantized: keys are
                      written already rotated by it (the model rotates q and k alike)

    Memory per token = num_layers * 2 * num_kv_heads * head_dim * dtype bytes,
    which for Qwen3-0.6B in bf16 is 28 * 2 * 8 * 128 * 2 = 112 KiB, so one
    16-token block costs 1.75 MiB. Quantized: 28 * 2 * 8 * (128 + 4) = 57.75 KiB.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
        quantized: bool = False,
    ):
        """dtype: the model's activation dtype, which reads return; quantized: store int8."""
        shape = (config.num_layers, num_blocks, block_size, config.num_kv_heads, config.head_dim)
        # Zeroed, not left uninitialised: batched decode attention pads short contexts
        # with slots that may never have been written, and masks them out. A masked key
        # gets weight 0, but 0 * NaN is still NaN, so every slot must hold a finite value.
        storage = torch.int8 if quantized else dtype
        self.k = torch.zeros(shape, device=device, dtype=storage)
        self.v = torch.zeros(shape, device=device, dtype=storage)
        self.k_scale = self.v_scale = self.key_rotation = None
        if quantized:
            self.k_scale = torch.zeros(shape[:-1], device=device, dtype=torch.float32)
            self.v_scale = torch.zeros(shape[:-1], device=device, dtype=torch.float32)
            self.key_rotation = hadamard(config.head_dim, device)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.quantized = quantized

    @staticmethod
    def bytes_per_block(
        config: ModelConfig, block_size: int, dtype: torch.dtype, quantized: bool = False
    ) -> int:
        """Memory one block takes across all layers, K and V: what sizes the pool."""
        per_head = config.head_dim + 4 if quantized else config.head_dim * dtype.itemsize
        return config.num_layers * 2 * config.num_kv_heads * per_head * block_size

    def layer_slots(self, layer_idx: int) -> LayerSlots:
        """One layer's pool as flat slot arrays (views, not copies)."""

        def flat(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t[layer_idx].flatten(0, 1)

        return LayerSlots(*(flat(t) for t in (self.k, self.v, self.k_scale, self.v_scale)))

    def write(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Store this step's new keys/values for one layer.

        k, v:         [T, num_kv_heads, head_dim]  one row per token in the flat batch
        slot_mapping: [T]                          physical slot of each token

        Every token writes exactly one slot, so this is a single scatter no matter
        how many sequences are in the batch or whether they are prefilling or decoding.
        """
        slots = self.layer_slots(layer_idx)
        if not self.quantized:
            slots.k.index_copy_(0, slot_mapping, k)
            slots.v.index_copy_(0, slot_mapping, v)
            return
        for x, values, scales in ((k, slots.k, slots.k_scale), (v, slots.v, slots.v_scale)):
            q, scale = quantize_heads(x)
            values.index_copy_(0, slot_mapping, q)
            scales.index_copy_(0, slot_mapping, scale)

    def read(
        self, layer_idx: int, slots: torch.Tensor, use_kernels: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one sequence's keys/values for one layer, in position order.

        slots:   [S]  physical slot of positions 0..S-1 of the sequence
        returns  (k, v), each [S, num_kv_heads, head_dim] in the model dtype, a
                 contiguous copy (dequantized, if the pool is int8; with
                 use_kernels, by one Metal kernel per tensor)

        The copy is the price of a plain-PyTorch paged attention; the decode
        kernel reads the blocks in place instead.
        """
        layer = self.layer_slots(layer_idx)
        if not self.quantized:
            return layer.k.index_select(0, slots), layer.v.index_select(0, slots)
        if use_kernels:
            return (
                kernels.gather_dequantize_kv(layer.k, layer.k_scale, slots, self.dtype),
                kernels.gather_dequantize_kv(layer.v, layer.v_scale, slots, self.dtype),
            )
        return tuple(
            (values.index_select(0, slots).float() * scales.index_select(0, slots)[..., None]).to(
                self.dtype
            )
            for values, scales in ((layer.k, layer.k_scale), (layer.v, layer.v_scale))
        )
