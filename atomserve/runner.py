# The model runner: turns "run these tokens of these sequences" into one GPU forward pass.
#
# It is the bridge between the two halves of the engine. Above it, plain
# Python objects decide *what* to run (sequences, block tables, token counts).
# Below it, the model is a pure function over tensors. The runner flattens the
# former into the latter: the `Batch` of input ids, positions, KV slot
# addresses and per-sequence offsets, built on the host and copied to the GPU
# once per step.
#
# The runner never changes a sequence. Advancing `num_computed_tokens` and
# appending sampled tokens is the caller's job, so the same plan can be
# inspected, tested or retried.

import torch

from atomserve.config import ModelConfig
from atomserve.kv_cache import KVPool
from atomserve.model import Batch, Weights, forward
from atomserve.sequence import Sequence


def slot_of(block_table: list[int], position: int, block_size: int) -> int:
    """Physical pool slot holding the K/V of the token at `position` of a sequence."""
    return block_table[position // block_size] * block_size + position % block_size


class ModelRunner:
    """Owns the weights and the KV pool; executes one step for a list of sequences."""

    def __init__(self, weights: Weights, config: ModelConfig, pool: KVPool):
        self.weights = weights
        self.config = config
        self.pool = pool
        self.device = weights.embed.device

    def prepare(self, seqs: list[Sequence], num_new_tokens: list[int]) -> Batch:
        """Flatten one step's work into a Batch.

        seqs[i] contributes its next `num_new_tokens[i]` uncomputed tokens:
        the whole prompt (prefill), a chunk of it, or the single newest token
        (decode). Its block table must already cover them. A sequence gets a
        logits row only if this step reaches the end of its known tokens; a
        prompt chunk that stops short has nothing to predict yet.
        """
        bs = self.pool.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        context_slots: list[int] = []
        cu_seqlens, cu_ctxlens = [0], [0]
        logits_idx: list[int] = []

        for seq, n in zip(seqs, num_new_tokens, strict=True):
            start, end = seq.num_computed_tokens, seq.num_computed_tokens + n
            if n < 1 or end > len(seq.token_ids):
                raise ValueError(f"cannot run tokens {start}:{end} of {len(seq.token_ids)}")
            if end > len(seq.block_table) * bs:
                raise ValueError(f"block table holds {len(seq.block_table) * bs} slots, need {end}")

            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            slots = [slot_of(seq.block_table, p, bs) for p in range(end)]
            slot_mapping.extend(slots[start:])
            context_slots.extend(slots)
            cu_seqlens.append(cu_seqlens[-1] + n)
            cu_ctxlens.append(cu_ctxlens[-1] + end)
            if end == len(seq.token_ids):
                logits_idx.append(cu_seqlens[-1] - 1)

        def to_device(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=self.device)

        return Batch(
            input_ids=to_device(input_ids),
            positions=to_device(positions),
            slot_mapping=to_device(slot_mapping),
            cu_seqlens=cu_seqlens,
            context_slots=to_device(context_slots),
            cu_ctxlens=cu_ctxlens,
            logits_idx=to_device(logits_idx),
        )

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], num_new_tokens: list[int]) -> torch.Tensor:
        """Run one step and return next-token logits, [B', vocab] fp32.

        Row j belongs to the j-th sequence (in `seqs` order) that reaches the end
        of its known tokens this step; see `prepare`. Writes the new tokens' K/V
        into the pool as a side effect.
        """
        batch = self.prepare(seqs, num_new_tokens)
        return forward(self.weights, self.config, batch, self.pool)
