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

import torch

from atomserve.config import ModelConfig


class KVPool:
    """Preallocated key/value storage for all sequences, all layers.

    k, v: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]

    Memory per token = num_layers * 2 * num_kv_heads * head_dim * dtype bytes,
    which for Qwen3-0.6B in bf16 is 28 * 2 * 8 * 128 * 2 = 112 KiB, so one
    16-token block costs 1.75 MiB.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (config.num_layers, num_blocks, block_size, config.num_kv_heads, config.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.num_blocks = num_blocks
        self.block_size = block_size

    @staticmethod
    def bytes_per_block(config: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
        """Memory one block takes across all layers, K and V: what sizes the pool."""
        per_token = config.num_layers * 2 * config.num_kv_heads * config.head_dim
        return per_token * block_size * dtype.itemsize

    def _slots(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's pool as flat slot arrays, each [num_blocks * block_size, kv_heads, D].

        A view, not a copy: writes through it land in the pool.
        """
        k, v = self.k[layer_idx], self.v[layer_idx]
        return k.view(-1, *k.shape[2:]), v.view(-1, *v.shape[2:])

    def write(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Store this step's new keys/values for one layer.

        k, v:         [T, num_kv_heads, head_dim]  one row per token in the flat batch
        slot_mapping: [T]                          physical slot of each token

        Every token writes exactly one slot, so this is a single scatter no matter
        how many sequences are in the batch or whether they are prefilling or decoding.
        """
        k_slots, v_slots = self._slots(layer_idx)
        k_slots.index_copy_(0, slot_mapping, k)
        v_slots.index_copy_(0, slot_mapping, v)

    def read(self, layer_idx: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one sequence's keys/values for one layer, in position order.

        slots:   [S]  physical slot of positions 0..S-1 of the sequence
        returns  (k, v), each [S, num_kv_heads, head_dim], a contiguous copy

        The copy is the price of a plain-PyTorch paged attention; a custom kernel
        would read the blocks in place instead.
        """
        k_slots, v_slots = self._slots(layer_idx)
        return k_slots.index_select(0, slots), v_slots.index_select(0, slots)
