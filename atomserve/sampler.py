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

import torch

from atomserve import kernels
from atomserve.sequence import Sequence


class Sampler:
    """Batched sampling, plus the per-request random generators of seeded requests."""

    def __init__(self, device: torch.device, use_kernels: bool = False):
        """use_kernels: take the top-k with the Metal kernel (kernels.topk_rows)."""
        self.device = device
        self.use_kernels = use_kernels
        self._generators: dict[int, torch.Generator] = {}

    def release(self, seq_id: int) -> None:
        """Forget a finished request's generator."""
        self._generators.pop(seq_id, None)

    def _generator(self, seq: Sequence) -> torch.Generator:
        """The request's own generator, created from its seed on first use."""
        if seq.id not in self._generators:
            g = torch.Generator(device=self.device)
            g.manual_seed(seq.params.seed)
            self._generators[seq.id] = g
        return self._generators[seq.id]

    def top_k(self, logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row's k largest logits and their token ids, best first: ([B, k], [B, k])."""
        if self.use_kernels and k <= kernels.TOPK_MAX:
            return kernels.topk_rows(logits, k)
        return logits.topk(k, dim=-1)

    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """Choose one next token per row.

        logits: [B, vocab] fp32, row i belongs to seqs[i]
        returns B token ids (one host sync, here, per engine step)
        """
        if not seqs:  # a step that only advanced prompt chunks
            return []
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
            return indices[:, 0].tolist()

        # Per-row settings, in one host-to-device copy.
        settings = torch.tensor(
            [
                [p.temperature if p.temperature > 0 else 1.0 for p in params],
                [p.top_p for p in params],
                ks,
                [p.temperature == 0 for p in params],
            ],
            device=logits.device,
            dtype=torch.float32,
        )
        temperature, top_p, k, is_greedy = settings
        rank = torch.arange(width, device=logits.device)
        values = (values / temperature[:, None]).masked_fill(rank >= k[:, None], float("-inf"))
        probs = values.softmax(dim=-1)

        # Top-p (nucleus): drop a token once the mass of the tokens ranked above it
        # already reaches p. The best token always survives.
        mass_before = probs.cumsum(dim=-1) - probs
        probs = probs.masked_fill(mass_before >= top_p[:, None], 0.0)

        noise = torch.empty_like(probs).exponential_()
        for i, seq in enumerate(seqs):
            if seq.params.seed is not None and seq.params.temperature > 0:
                # Draw exactly k values so the stream doesn't depend on batch neighbours.
                noise[i, : ks[i]].exponential_(generator=self._generator(seq))
        choice = (probs / noise.clamp_min(1e-10)).argmax(dim=-1)  # clamp: no 0/0 = NaN
        sampled = indices.gather(1, choice[:, None]).squeeze(1)
        return torch.where(is_greedy.bool(), indices[:, 0], sampled).tolist()


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
