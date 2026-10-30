# Speculative decoding, where a small draft model guesses and the big model checks.
#
# At batch 1, a decode step costs about the time to read all of the target's
# weights, whether it computes 1 token or 5. So let a cheap draft model (same
# tokenizer, e.g. Qwen3-0.6B for Qwen3-4B) propose k tokens one by one, then
# run the target once over all of them: k+1 positions in one pass, which
# costs about the same as 1. Each step therefore yields between 1 and k+1
# tokens for roughly (k draft steps + 1 target step) of time.
#
#     draft:   ... t  -> d1 -> d2 -> d3 -> d4           (k cheap, sequential steps)
#     target:  [t, d1, d2, d3, d4] in one pass          -> p1 .. p5
#     accept:  d1 ok, d2 ok, d3 rejected -> keep d1 d2, then sample a fix from p3
#
# Acceptance is rejection sampling (Leviathan et al. 2023; Chen et al. 2023):
# draft token d_i, drawn from the draft's distribution q_i, is kept with
# probability min(1, p_i(d_i) / q_i(d_i)). At the first rejection the
# replacement is drawn from norm(max(0, p_i - q_i)); if all k survive, a
# bonus token comes from p_{k+1}. The output then has exactly the target's
# distribution, whatever the draft proposes: a bad draft only costs speed.
# Greedy requests are the special case of one-hot p and q: keep drafts while
# they equal the target's argmax.
#
# Rejected tokens need no undo. Their K/V were written to slots past the
# sequence's num_computed_tokens, which the next step simply overwrites.
#
# Both models' KV pools are indexed by the same block tables. The draft also
# runs every prompt chunk, so a long prompt never turns into one big draft
# prefill later. Plain decodes (when the batch is too big to speculate) skip
# the draft; it records how far behind it is (draft_num_computed) and catches
# up in its next pass, as it also does after a fully accepted round, since its
# own last guess is never fed back to it.

import torch

from atomserve.runner import ModelRunner, Row, rows_for
from atomserve.scheduler import SchedulerOutput
from atomserve.sequence import Sequence, StepResult


class SpeculativeRunner:
    """Runs scheduled steps with a target model and a draft model sharing block tables.

    Drop-in for ModelRunner in the engine: `execute(out)` returns one StepResult per
    scheduled sequence. Sequences with num_lookahead > 0 speculate that many tokens.
    """

    def __init__(self, target: ModelRunner, draft: ModelRunner):
        if target.config.vocab_size != draft.config.vocab_size:
            raise ValueError("draft and target models must share a vocabulary")
        self.target = target
        self.draft = draft
        self.sampler = target.sampler  # one set of per-request generators for both models
        self.num_drafted = 0  # draft tokens proposed so far
        self.num_accepted = 0  # ... and how many of them the target kept

    @torch.inference_mode()
    def execute(self, out: SchedulerOutput) -> list[StepResult]:
        seqs, ns, ks = out.seqs, out.num_new_tokens, out.num_lookahead
        spec = [i for i, k in enumerate(ks) if k > 0]
        drafts, draft_probs = self._propose(seqs, ns, ks, spec)

        # Verify: speculating sequences run [newest token, d1..dk] with logits at every
        # position; the others run their usual rows.
        rows = rows_for(seqs, ns)
        for i, d in zip(spec, drafts, strict=True):
            seq = seqs[i]
            rows[i] = Row([seq.token_ids[-1], *d], seq.num_tokens - 1, seq.block_table, len(d) + 1)
        logits, batch = self.target.forward(rows)

        # Split the target's logits rows between plain sampling and verification.
        offsets, at = [], 0
        for r in rows:
            offsets.append(at)
            at += r.num_logits
        plain = [i for i in range(len(seqs)) if not ks[i] and rows[i].num_logits]
        plain_tokens = self.sampler.draw(
            self.sampler.candidates(logits[[offsets[i] for i in plain]], [seqs[i] for i in plain]),
            [seqs[i] for i in plain],
        ) if plain else None  # fmt: skip
        accepted, fixes = self._verify(seqs, spec, drafts, draft_probs, logits, offsets)

        # One sync for the whole step.
        plain_tokens = plain_tokens.tolist() if plain_tokens is not None else []
        accepted, fixes = accepted.tolist(), fixes.tolist()
        self.sampler.synced()
        del batch  # the GPU is done with its host buffer now

        results = []
        plain_iter, spec_iter = iter(plain_tokens), iter(zip(drafts, accepted, fixes, strict=True))
        for i, (seq, n, k) in enumerate(zip(seqs, ns, ks, strict=True)):
            old = seq.num_computed_tokens
            if k == 0:
                tokens = [next(plain_iter)] if i in plain else []
                draft_done = old + n if n > 1 else seq.draft_num_computed  # see _propose
                results.append(StepResult(tokens, old + n, draft_done))
                continue
            d, a, fix = next(spec_iter)
            self.num_drafted += k
            self.num_accepted += a
            # Target K/V are valid for the newest token and the a accepted drafts; the
            # draft wrote positions up to old + k - 1 (it never saw its own last guess).
            results.append(StepResult([*d[:a], fix], old + a + 1, min(old + k, old + a + 1)))
        return results

    def _propose(
        self, seqs: list[Sequence], ns: list[int], ks: list[int], spec: list[int]
    ) -> tuple[list[list[int]], list[torch.Tensor]]:
        """Run the draft: catch every sequence up, then guess ks[i] tokens for each speculator.

        returns drafts[j] (host token ids, len ks[spec[j]]) and, per draft round r,
        the [len(spec), vocab] distributions the round-r tokens were drawn from
        (rows of sequences that drafted fewer than r tokens are unused)
        """
        vocab = self.target.config.vocab_size
        spec_seqs = [seqs[i] for i in spec]
        drafts: list[list[int]] = [[] for _ in spec]
        probs: list[torch.Tensor] = []

        # Round 1: speculators and prompt chunks feed whatever the draft hasn't seen
        # yet (the chunk, the newest token, or a few more if the draft fell behind);
        # speculators also get logits for their first guess. Plain decodes skip it.
        rows = []
        for seq, n, k in zip(seqs, ns, ks, strict=True):
            if k > 0 or n > 1:
                end = seq.num_computed_tokens + n
                tokens = seq.token_ids[seq.draft_num_computed : end]
                rows.append(Row(tokens, seq.draft_num_computed, seq.block_table, int(k > 0)))
        if not rows:
            return drafts, probs
        logits, batch = self.draft.forward(rows)
        if not spec:
            if logits.is_mps:
                torch.mps.synchronize()  # nothing else syncs before `batch` is dropped
            return drafts, probs

        for r in range(1, max(ks) + 1):
            if r > 1:  # feed each active speculator its previous guess
                active = [j for j, i in enumerate(spec) if ks[i] >= r]
                rows = [
                    Row([drafts[j][-1]], spec_seqs[j].num_tokens + r - 2, spec_seqs[j].block_table)
                    for j in active
                ]
                logits, batch = self.draft.forward(rows)
            else:
                active = list(range(len(spec)))
            cand = self.sampler.candidates(logits, [spec_seqs[j] for j in active])
            tokens = self.sampler.draw(cand, [spec_seqs[j] for j in active]).tolist()
            self.sampler.synced()
            dense = torch.zeros(len(spec), vocab, device=logits.device)
            dense[active] = cand.dense(vocab)
            probs.append(dense)
            for j, t in zip(active, tokens, strict=True):
                drafts[j].append(t)
        del batch
        return drafts, probs

    def _verify(
        self,
        seqs: list[Sequence],
        spec: list[int],
        drafts: list[list[int]],
        draft_probs: list[torch.Tensor],
        logits: torch.Tensor,
        offsets: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rejection-sample every speculator's drafts against the target, all at once.

        returns (accepted [S]: drafts kept, fix [S]: the token that follows them),
        both still on the GPU
        """
        device = logits.device
        if not spec:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty
        S, K, vocab = len(spec), max(len(d) for d in drafts), logits.shape[1]
        k = torch.tensor([len(d) for d in drafts], device=device)

        # Target distributions at the K+1 verified positions of each speculator, [S, K+1, V]
        # (positions past a sequence's own k repeat its last row and are never used).
        rows = [
            offsets[i] + min(j, len(d))
            for i, d in zip(spec, drafts, strict=True)
            for j in range(K + 1)
        ]
        row_seqs = [seqs[i] for i in spec for _ in range(K + 1)]
        p = self.sampler.candidates(logits[rows], row_seqs).dense(vocab).view(S, K + 1, vocab)
        q = torch.stack(draft_probs, dim=1)  # [S, K, V]
        d = torch.tensor([ds + [0] * (K - len(ds)) for ds in drafts], device=device)  # [S, K]

        # Keep d_j with probability min(1, p_j(d_j) / q_j(d_j)); q_j(d_j) > 0 as d_j ~ q_j.
        p_d = p[:, :K].gather(2, d[..., None]).squeeze(2)
        q_d = q.gather(2, d[..., None]).squeeze(2)
        u = self._uniform(spec, seqs, (S, K), device)
        keep = (u * q_d < p_d) & (torch.arange(K, device=device) < k[:, None])
        accepted = keep.long().cumprod(dim=1).sum(dim=1)  # leading run of kept drafts

        # The next token: from norm(max(0, p_a - q_a)) after a rejection at a, or from
        # p_{k+1} (the bonus) when every draft was kept.
        residual = torch.cat([(p[:, :K] - q).clamp_min(0.0), p[:, K:]], dim=1)  # [S, K+1, V]
        rows_s = torch.arange(S, device=device)
        bonus = accepted == k
        dist = torch.where(bonus[:, None], p[rows_s, k], residual[rows_s, accepted])
        empty = dist.sum(dim=1, keepdim=True) <= 0  # p == q up to rounding: use p itself
        dist = torch.where(empty, p[rows_s, accepted], dist)
        fixes = self._draw_dense(dist, spec, seqs)
        return accepted, fixes

    def _uniform(self, spec, seqs, shape, device) -> torch.Tensor:
        """U(0, 1) draws per speculator, from its own generator when it is seeded."""
        u = torch.rand(shape, device=device)
        for j, i in enumerate(spec):
            if (g := self.sampler.generator(seqs[i])) is not None:
                u[j] = torch.rand(shape[1], device=device, generator=g)
        return u

    def _draw_dense(self, dist: torch.Tensor, spec, seqs) -> torch.Tensor:
        """One token per row of unnormalised [S, V] weights (exponential race)."""
        noise = torch.empty_like(dist).exponential_()
        for j, i in enumerate(spec):
            if (g := self.sampler.generator(seqs[i])) is not None:
                noise[j].exponential_(generator=g)
        return (dist / noise.clamp_min(1e-10)).argmax(dim=1)
