# The sampler: turns next-token logits into chosen tokens, for a whole batch at once.
#
# Every row of the batch can have its own settings (one request wants greedy,
# another temperature 0.7 with top-p 0.8), so each setting becomes a per-row
# tensor and the steps run as batched tensor ops instead of a Python loop:
#
#     logits -> presence penalty -> top-k -> / temperature -> softmax -> top-p -> draw
#
# Top-k runs first, on the raw logits: dividing by a positive temperature
# doesn't change which tokens are largest, and afterwards only k values per
# row are left to work on instead of 151,936. A greedy row is just its top 1.
#
# The draw is an "exponential race": argmax(p_i / E_i) with E_i ~ Exp(1)
# picks token i with probability exactly p_i / sum(p). It needs no
# renormalisation after masking, and lets a seeded request draw its noise
# from its own generator while unseeded rows share one.

from dataclasses import dataclass

import torch

from atomserve import kernels
from atomserve.sequence import Sequence


@dataclass
class Candidates:
    """Each row's sampling distribution, in sparse form.

    probs:   [B, W] fp32, the row's probabilities after top-k / temperature / top-p
             (zero for tokens filtered out; a greedy row is a single 1.0)
    indices: [B, W] the token ids those probabilities belong to, best first
    ks:      host ints, how many of the W columns row i may use
    """

    probs: torch.Tensor
    indices: torch.Tensor
    ks: list[int]

    def dense(self, vocab: int) -> torch.Tensor:
        """The same distributions as full [B, vocab] probability vectors."""
        out = torch.zeros(self.probs.shape[0], vocab, device=self.probs.device)
        return out.scatter_(1, self.indices, self.probs)


class Sampler:
    """Batched sampling, plus the per-request random generators of seeded requests."""

    def __init__(self, device: torch.device, use_kernels: bool = False):
        """use_kernels: take the top-k with the Metal kernel (kernels.topk_rows)."""
        self.device = device
        self.use_kernels = use_kernels
        self._generators: dict[int, torch.Generator] = {}
        # Host tensors still being copied to the GPU asynchronously; kept alive until
        # the caller has synchronised (see `synced`).
        self._in_flight: list[torch.Tensor] = []

    def release(self, seq_id: int) -> None:
        """Forget a finished request's generator."""
        self._generators.pop(seq_id, None)

    def generator(self, seq: Sequence) -> torch.Generator | None:
        """The request's own generator if it is seeded (created on first use), else None."""
        if seq.params.seed is None:
            return None
        if seq.id not in self._generators:
            g = torch.Generator(device=self.device)
            g.manual_seed(seq.params.seed)
            self._generators[seq.id] = g
        return self._generators[seq.id]

    def synced(self) -> None:
        """Call after reading results back from the GPU: frees the uploads' host buffers."""
        self._in_flight.clear()

    def top_k(self, logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row's k largest logits and their token ids, best first: ([B, k], [B, k])."""
        if self.use_kernels and k <= kernels.TOPK_MAX:
            return kernels.topk_rows(logits, k)
        return logits.topk(k, dim=-1)

    def candidates(self, logits: torch.Tensor, seqs: list[Sequence]) -> Candidates:
        """Turn logits into each row's sampling distribution (rows may repeat a sequence).

        logits: [B, vocab] fp32, row i sampled with seqs[i]'s settings
        """
        vocab = logits.shape[1]
        params = [s.params for s in seqs]
        logits = apply_presence_penalty(logits, seqs)
        # Candidates per row: its top_k (0 means all), and just 1 for a greedy row.
        ks = [
            1 if p.temperature == 0 else p.top_k if 0 < p.top_k < vocab else vocab for p in params
        ]
        width = max(ks)
        values, indices = self.top_k(logits, width)  # [B, width], sorted, best first
        if width == 1:  # the whole batch is greedy
            return Candidates(torch.ones_like(values), indices, ks)

        # Per-row settings, in one asynchronous host-to-device copy.
        settings = torch.tensor(
            [
                [p.temperature if p.temperature > 0 else 1.0 for p in params],
                [p.top_p for p in params],
                ks,
            ],
            dtype=torch.float32,
        )
        self._in_flight.append(settings)
        temperature, top_p, k = settings.to(logits.device, non_blocking=True)
        rank = torch.arange(width, device=logits.device)
        values = (values / temperature[:, None]).masked_fill(rank >= k[:, None], float("-inf"))
        probs = values.softmax(dim=-1)

        # Top-p (nucleus): drop a token once the mass of the tokens ranked above it
        # already reaches p. The best token always survives.
        mass_before = probs.cumsum(dim=-1) - probs
        return Candidates(probs.masked_fill(mass_before >= top_p[:, None], 0.0), indices, ks)

    def draw(self, cand: Candidates, seqs: list[Sequence]) -> torch.Tensor:
        """One token per row from its candidates, on the GPU: [B] token ids."""
        if cand.probs.shape[1] == 1:
            return cand.indices[:, 0]
        noise = torch.empty_like(cand.probs).exponential_()
        for i, seq in enumerate(seqs):
            if (g := self.generator(seq)) is not None:
                # Draw exactly k values so the stream doesn't depend on batch neighbours.
                noise[i, : cand.ks[i]].exponential_(generator=g)
        choice = (cand.probs / noise.clamp_min(1e-10)).argmax(dim=-1)  # clamp: no 0/0 = NaN
        return cand.indices.gather(1, choice[:, None]).squeeze(1)

    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """Choose one next token per row.

        logits: [B, vocab] fp32, row i belongs to seqs[i]
        returns B token ids (one host sync, here, per engine step)
        """
        if not seqs:  # a step that only advanced prompt chunks
            return []
        tokens = self.draw(self.candidates(logits, seqs), seqs).tolist()
        self.synced()
        return tokens


def apply_presence_penalty(logits: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
    """Subtract each row's presence_penalty from the logits of tokens it already generated.

    logits: [B, vocab]  ->  [B, vocab] (a new tensor if any penalty applies)

    All (row, token) pairs are gathered on the host and applied in one
    scatter, so the cost doesn't grow with the number of penalised rows.
    """
    rows: list[int] = []
    cols: list[int] = []
    amounts: list[float] = []
    for i, seq in enumerate(seqs):
        if seq.params.presence_penalty != 0:
            seen = set(seq.output_ids)
            rows.extend([i] * len(seen))
            cols.extend(seen)
            amounts.extend([-seq.params.presence_penalty] * len(seen))
    if not rows:
        return logits
    device = logits.device
    return logits.index_put(
        (torch.tensor(rows, device=device), torch.tensor(cols, device=device)),
        torch.tensor(amounts, device=device),
        accumulate=True,
    )
