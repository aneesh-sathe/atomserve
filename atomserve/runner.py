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

from atomserve import kernels
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

    def __init__(
        self, weights: Weights, config: ModelConfig, pool: KVPool, use_kernels: bool | None = None
    ):
        """use_kernels: run the Metal kernels in kernels.py (default: when available)."""
        self.weights = weights
        self.config = config
        self.pool = pool
        self.device = weights.embed.device
        if use_kernels is None:
            use_kernels = kernels.available(self.device, weights.embed.dtype, config)
        self.use_kernels = use_kernels
        self.sampler = Sampler(self.device, use_kernels=use_kernels)

    def prepare(self, seqs: list[Sequence], num_new_tokens: list[int]) -> Batch:
        """Flatten one step's work into a Batch.

        seqs[i] contributes its next `num_new_tokens[i]` uncomputed tokens:
        the whole prompt (prefill), a chunk of it, or the single newest token
        (decode). Its block table must already cover them. A sequence gets a
        logits row only if this step reaches the end of its known tokens; a
        prompt chunk that stops short has nothing to predict yet.

        Context slots for attention are expanded from block tables on the GPU,
        so host work grows with the number of blocks, not of tokens. All of the
        step's integers go to the GPU in a single copy (see `_Upload`).
        """
        bs = self.pool.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        logits_idx: list[int] = []
        prefill_spans: list[tuple[slice, list[int], int]] = []  # rows, block table, context
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
                prefill_spans.append((slice(row, row + n), table, end))
            row += n
            if seq.reaches_end(n):
                logits_idx.append(row - 1)

        upload = _Upload()
        fields = [upload.add(v) for v in (input_ids, positions, slot_mapping, logits_idx)]
        prefill_tables = [upload.add(table) for _, table, _ in prefill_spans]
        width = max((len(t) for t in decode_tables), default=0)
        tables = upload.add([b for t in decode_tables for b in t + [0] * (width - len(t))])
        rows, lens = upload.add(decode_rows), upload.add(decode_lens)
        on_device = upload.send(self.device)

        batch = Batch(
            *(on_device[f] for f in fields),
            prefills=[
                PrefillSpan(rows, self._expand_slots(on_device[t][None], end)[0])
                for (rows, _, end), t in zip(prefill_spans, prefill_tables, strict=True)
            ],
            use_kernels=self.use_kernels,
            host_buffer=upload.host,
        )
        if decode_rows:
            decode_tables_t = on_device[tables].view(len(decode_rows), width)
            batch.decode_rows = on_device[rows]
            if self.use_kernels:
                batch.decode_block_tables = decode_tables_t
                batch.decode_lens = on_device[lens]
                batch.decode_max_len = max(decode_lens)
            else:
                longest = max(decode_lens)
                batch.decode_slots = self._expand_slots(decode_tables_t, longest)
                if min(decode_lens) < longest:
                    key_positions = torch.arange(longest, device=self.device)
                    batch.decode_mask = key_positions[None, :] < on_device[lens][:, None]
        return batch

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
        logits = forward(self.weights, self.config, batch, self.pool)
        if logits.is_mps:
            torch.mps.synchronize()  # the batch's host buffer may be freed once we return
        return logits

    @torch.inference_mode()
    def execute(self, seqs: list[Sequence], num_new_tokens: list[int]) -> list[int]:
        """Run one step and sample: one new token per sequence that reaches its end.

        The engine's entry point; `run` alone is for callers that want logits.
        `batch` stays referenced until sampling has synchronised with the GPU,
        which keeps its host buffer alive for the asynchronous upload.
        """
        batch = self.prepare(seqs, num_new_tokens)
        logits = forward(self.weights, self.config, batch, self.pool)
        sampling = [s for s, n in zip(seqs, num_new_tokens, strict=True) if s.reaches_end(n)]
        tokens = self.sampler.sample(logits, sampling)
        del batch  # only now: the GPU has consumed everything it read from the host buffer
        return tokens


class _Upload:
    """Packs one step's integer arrays into one host tensor, copied to the GPU in one go.

    Each separate small copy costs ~30-140 us on MPS, about as much as a whole
    decoder layer at batch 1, and a step needs up to a dozen of them. The copy
    is asynchronous (non_blocking), so the GPU reads the host tensor *later*:
    it must not be modified or freed until the step has synchronised. The Batch
    keeps a reference to it (`host_buffer`) for exactly that reason.
    """

    def __init__(self):
        self._values: list[int] = []
        self._parts: list[tuple[int, int]] = []
        self.host: torch.Tensor | None = None

    def add(self, values: list[int]) -> int:
        """Queue an array; returns its index into the list `send` returns."""
        start = len(self._values)
        self._values.extend(values)
        self._parts.append((start, len(self._values)))
        return len(self._parts) - 1

    def send(self, device: torch.device) -> list[torch.Tensor]:
        """Copy everything to `device`; returns one view per queued array, in order."""
        self.host = torch.tensor(self._values, dtype=torch.long)
        on_device = self.host.to(device, non_blocking=True)
        return [on_device[a:b] for a, b in self._parts]
