# The KV cache: remembered keys and values of tokens the model has already seen.
#
# At every layer, attention needs the keys and values of *all* previous tokens.
# Those never change once computed (a token's K/V depend only on itself and
# the tokens before it), so we store them instead of recomputing them. That
# turns generation from O(N^2) work into O(N): the prompt is processed once
# ("prefill"), then each new token costs one small forward pass ("decode").
#
# This version serves a single sequence and stores it contiguously. It becomes
# a paged pool shared by many sequences in M4.

import torch

from atomserve.config import ModelConfig


class KVCache:
    """Preallocated per-layer key/value storage for one sequence.

    k, v: [num_layers, max_len, num_kv_heads, head_dim]

    Memory per token = num_layers * 2 * num_kv_heads * head_dim * dtype bytes,
    which for Qwen3-0.6B in bf16 is 28 * 2 * 8 * 128 * 2 = 112 KiB.
    """

    def __init__(self, config: ModelConfig, max_len: int, device: torch.device, dtype: torch.dtype):
        shape = (config.num_layers, max_len, config.num_kv_heads, config.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0  # tokens whose K/V are stored (same for every layer)

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store one layer's K/V for the new tokens; return that layer's full history.

        k, v:    [T, num_kv_heads, head_dim]   the T tokens of the current forward pass
        returns  (k_all, v_all), each [length + T, num_kv_heads, head_dim]

        `length` is not advanced here because every layer writes to the same slots;
        `advance` is called once after the last layer.
        """
        end = self.length + k.shape[0]
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} tokens, capacity {self.max_len}")
        self.k[layer_idx, self.length : end] = k
        self.v[layer_idx, self.length : end] = v
        return self.k[layer_idx, :end], self.v[layer_idx, :end]

    def advance(self, num_tokens: int) -> None:
        """Mark `num_tokens` more tokens as stored, after all layers have appended them."""
        self.length += num_tokens
