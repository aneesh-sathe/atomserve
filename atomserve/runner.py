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
# appending sampled tokens is the scheduler's job, so the same plan can be
# inspected, tested or retried.

import torch

from atomserve.block_allocator import blocks_for_tokens
from atomserve.config import ModelConfig
from atomserve.kv_cache import KVPool
from atomserve.model import Batch, PrefillSpan, Weights, forward
from atomserve.sampler import Sampler
from atomserve.sequence import Sequence


def slot_of(block_table: list[int], position: int, block_size: int) -> int:
    """Physical pool slot holding the K/V of the token at `position` of a sequence."""
    return block_table[position // block_size] * block_size + position % block_size


class ModelRunner:
    """Owns the weights, KV pool and sampler; executes one step for a list of sequences."""

    def __init__(self, weights: Weights, config: ModelConfig, pool: KVPool):
        self.weights = weights
        self.config = config
        self.pool = pool
        self.device = weights.embed.device
        self.sampler = Sampler(self.device)

    def prepare(self, seqs: list[Sequence], num_new_tokens: list[int]) -> Batch:
        """Flatten one step's work into a Batch.

        seqs[i] contributes its next `num_new_tokens[i]` uncomputed tokens:
        the whole prompt (prefill), a chunk of it, or the single newest token
        (decode). Its block table must already cover them. A sequence gets a
        logits row only if this step reaches the end of its known tokens; a
        prompt chunk that stops short has nothing to predict yet.

        Context slots for attention are expanded from block tables on the GPU,
        so host work grows with the number of blocks, not of tokens.
        """
        bs = self.pool.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        logits_idx: list[int] = []
        prefills: list[PrefillSpan] = []
        decode_rows: list[int] = []
        decode_tables: list[list[int]] = []
        decode_lens: list[int] = []
        row = 0

        for seq, n in zip(seqs, num_new_tokens, strict=True):
            start, end = seq.num_computed_tokens, seq.num_computed_tokens + n
            if n < 1 or end > len(seq.token_ids):
                raise ValueError(f"cannot run tokens {start}:{end} of {len(seq.token_ids)}")
            if end > len(seq.block_table) * bs:
                raise ValueError(f"block table holds {len(seq.block_table) * bs} slots, need {end}")

            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            slot_mapping.extend(slot_of(seq.block_table, p, bs) for p in range(start, end))
            table = seq.block_table[: blocks_for_tokens(end, bs)]
            if n == 1:
                decode_rows.append(row)
                decode_tables.append(table)
                decode_lens.append(end)
            else:
                slots = self._expand_slots(self._to_device([table]), end)[0]
                prefills.append(PrefillSpan(slice(row, row + n), slots))
            row += n
            if seq.reaches_end(n):
                logits_idx.append(row - 1)

        batch = Batch(
            input_ids=self._to_device(input_ids),
            positions=self._to_device(positions),
            slot_mapping=self._to_device(slot_mapping),
            logits_idx=self._to_device(logits_idx),
            prefills=prefills,
        )
        if decode_rows:
            width = max(len(t) for t in decode_tables)
            tables = self._to_device([t + [0] * (width - len(t)) for t in decode_tables])
            longest = max(decode_lens)
            batch.decode_rows = self._to_device(decode_rows)
            batch.decode_slots = self._expand_slots(tables, longest)
            if min(decode_lens) < longest:
                key_positions = torch.arange(longest, device=self.device)
                batch.decode_mask = key_positions[None, :] < self._to_device(decode_lens)[:, None]
        return batch

    def _to_device(self, values: list) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _expand_slots(self, block_tables: torch.Tensor, length: int) -> torch.Tensor:
        """Pool slot of positions 0..length-1 for each row of block tables.

        block_tables: [N, num_blocks]  ->  returns [N, length]
        (the same formula as `slot_of`, vectorised; padded table entries yield
        harmless in-range slots that the caller masks out)
        """
        bs = self.pool.block_size
        p = torch.arange(length, device=self.device)
        return block_tables[:, p // bs] * bs + p % bs

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], num_new_tokens: list[int]) -> torch.Tensor:
        """Run one step and return next-token logits, [B', vocab] fp32.

        Row j belongs to the j-th sequence (in `seqs` order) that reaches the end
        of its known tokens this step; see `prepare`. Writes the new tokens' K/V
        into the pool as a side effect.
        """
        batch = self.prepare(seqs, num_new_tokens)
        return forward(self.weights, self.config, batch, self.pool)

    @torch.inference_mode()
    def execute(self, seqs: list[Sequence], num_new_tokens: list[int]) -> list[int]:
        """Run one step and sample: one new token per sequence that reaches its end.

        The engine's entry point; `run` alone is for callers that want logits.
        """
        logits = self.run(seqs, num_new_tokens)
        sampling = [s for s, n in zip(seqs, num_new_tokens, strict=True) if s.reaches_end(n)]
        return self.sampler.sample(logits, sampling)
