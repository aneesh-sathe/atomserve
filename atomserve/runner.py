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
#
# The batch is described as `Row`s (a slice of tokens at some positions, plus
# where their KV goes) rather than as sequences, so speculative decoding can
# also run tokens that aren't part of any sequence yet: the draft's guesses.

from dataclasses import dataclass

import torch

from atomserve import kernels
from atomserve.block_allocator import blocks_for_tokens
from atomserve.config import ModelConfig
from atomserve.kv_cache import KVPool
from atomserve.model import Batch, PrefillSpan, Weights, forward
from atomserve.sampler import Sampler
from atomserve.scheduler import SchedulerOutput
from atomserve.sequence import Sequence, StepResult

# Rows with at most this many new tokens (decodes, speculative verification,
# small prompt chunks) are attended as that many single-query rows, through
# the batched decode path; longer ones as a prompt chunk.
MAX_DECODE_QUERIES = 8


def slot_of(block_table: list[int], position: int, block_size: int) -> int:
    """Physical pool slot holding the K/V of the token at `position` of a sequence."""
    return block_table[position // block_size] * block_size + position % block_size


@dataclass
class Row:
    """One sequence's part of a forward pass.

    tokens:      the new tokens, at positions start, start+1, ...
    start:       position of tokens[0] (everything before it is already in the pool)
    block_table: the sequence's blocks; must cover every new position
    num_logits:  how many of the last rows need next-token logits: 1 normally,
                 0 for a prompt chunk that stops short, all of them to verify drafts
    """

    tokens: list[int]
    start: int
    block_table: list[int]
    num_logits: int = 1


def rows_for(seqs: list[Sequence], num_new_tokens: list[int]) -> list[Row]:
    """The usual rows: each sequence's next uncomputed tokens, logits only if it reaches its end."""
    rows = []
    for seq, n in zip(seqs, num_new_tokens, strict=True):
        start = seq.num_computed_tokens
        if n < 1 or start + n > seq.num_tokens:
            raise ValueError(f"cannot run tokens {start}:{start + n} of {seq.num_tokens}")
        tokens = seq.token_ids[start : start + n]
        rows.append(Row(tokens, start, seq.block_table, int(seq.reaches_end(n))))
    return rows


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

    def prepare(self, rows: list[Row]) -> Batch:
        """Flatten one step's work into a Batch.

        Each row contributes its new tokens: a whole prompt, a chunk of it, a
        single decode token, or a decode token plus drafts to verify. Context
        slots for attention are expanded from block tables on the GPU, so host
        work grows with the number of blocks, not of tokens. All of the step's
        integers go to the GPU in a single copy (see `_Upload`).
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
        flat = 0  # index of the row's first token in the flat batch

        for r in rows:
            n, end = len(r.tokens), r.start + len(r.tokens)
            if n < 1 or not 0 <= r.num_logits <= n:
                raise ValueError(f"bad row: {n} tokens, {r.num_logits} logits")
            if end > len(r.block_table) * bs:
                raise ValueError(f"block table holds {len(r.block_table) * bs} slots, need {end}")

            input_ids.extend(r.tokens)
            positions.extend(range(r.start, end))
            slot_mapping.extend(slot_of(r.block_table, p, bs) for p in range(r.start, end))
            table = r.block_table[: blocks_for_tokens(end, bs)]
            if n <= MAX_DECODE_QUERIES:  # each query on its own, seeing keys up to itself
                for j in range(n):
                    decode_rows.append(flat + j)
                    decode_tables.append(table)
                    decode_lens.append(r.start + j + 1)
            else:
                prefill_spans.append((slice(flat, flat + n), table, end))
            logits_idx.extend(range(flat + n - r.num_logits, flat + n))
            flat += n

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
    def forward(self, rows: list[Row]) -> tuple[torch.Tensor, Batch]:
        """Run one forward pass; returns the requested logits [sum(num_logits), vocab] fp32.

        Writes the rows' K/V into the pool. The Batch is returned too and must
        stay referenced until the caller has synchronised with the GPU (e.g. by
        reading sampled tokens): its host buffer feeds an asynchronous copy.
        """
        batch = self.prepare(rows)
        return forward(self.weights, self.config, batch, self.pool), batch

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], num_new_tokens: list[int]) -> torch.Tensor:
        """Run one step and return next-token logits, [B', vocab] fp32.

        Row j belongs to the j-th sequence (in `seqs` order) that reaches the end
        of its known tokens this step. For tests and tools; the engine uses `execute`.
        """
        logits, _ = self.forward(rows_for(seqs, num_new_tokens))
        if logits.is_mps:
            torch.mps.synchronize()  # the batch's host buffer may be freed once we return
        return logits

    @torch.inference_mode()
    def execute(self, out: SchedulerOutput) -> list[StepResult]:
        """Run one scheduled step and sample: the engine's entry point.

        A sequence that reaches the end of its known tokens gets one new token.
        """
        logits, batch = self.forward(rows_for(out.seqs, out.num_new_tokens))
        sampling = [
            s for s, n in zip(out.seqs, out.num_new_tokens, strict=True) if s.reaches_end(n)
        ]
        sampled = iter(self.sampler.sample(logits, sampling))
        del batch  # only now: sampling synchronised, so the GPU is done with the host buffer
        return [
            StepResult(
                tokens=[next(sampled)] if seq.reaches_end(n) else [],
                num_computed=seq.num_computed_tokens + n,
            )
            for seq, n in zip(out.seqs, out.num_new_tokens, strict=True)
        ]


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
