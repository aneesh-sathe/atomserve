# A sequence: one request's tokens, its sampling settings and where its KV cache lives.
#
# This is the unit the scheduler juggles. Its life:
#
#     WAITING --admit--> RUNNING --EOS / max_tokens / abort--> FINISHED
#        ^                  |
#        +----preempt-------+   (blocks freed; K/V recomputed when readmitted)
#
# The runner only reads sequences; the scheduler is the only thing that
# changes them, always on the engine thread. Plain Python, no torch, so the
# scheduler can be tested without a GPU.

from dataclasses import dataclass, field
from enum import Enum


class SequenceStatus(Enum):
    WAITING = "waiting"  # queued; holds no KV blocks
    RUNNING = "running"  # admitted; owns blocks and is scheduled every step
    FINISHED = "finished"  # done; blocks returned


@dataclass(frozen=True)
class SamplingParams:
    """How to pick each next token, and when to stop. One per request.

    temperature:      0 means greedy (always the most likely token)
    top_k:            sample only among the k most likely tokens; 0 disables
    top_p:            then only among the smallest set whose probability mass reaches p
    presence_penalty: subtracted from the logit of every token already generated,
                      discouraging repetition (OpenAI semantics)
    max_tokens:       stop after this many generated tokens ("length")
    seed:             fixed seed for reproducible sampling; None uses the shared RNG
    ignore_eos:       keep generating past end-of-sequence (benchmarks want fixed lengths)
    """

    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    presence_penalty: float = 0.0
    max_tokens: int = 4096
    seed: int | None = None
    ignore_eos: bool = False


# Qwen3's recommended sampling for each mode (model card); used as request defaults.
THINKING_PARAMS = SamplingParams(temperature=0.6, top_p=0.95, top_k=20)
NON_THINKING_PARAMS = SamplingParams(temperature=0.7, top_p=0.8, top_k=20)


@dataclass(eq=False)
class Sequence:
    """Mutable state of one request inside the engine.

    token_ids:           prompt followed by every token generated so far
    num_prompt_tokens:   length of the prompt part of token_ids
    num_computed_tokens: leading tokens whose K/V are in the pool; the rest
                         still have to go through the model
    block_table:         physical KV block of each logical block, in order
                         (logical block i holds positions i*block_size ...)
    block_hashes:        content hash of each *full, computed* block, for prefix caching
    num_cached_tokens:   prompt tokens whose K/V were found in the prefix cache on
                         admission, instead of being computed (for stats)
    """

    id: int
    token_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    num_prompt_tokens: int = -1  # set from token_ids in __post_init__
    num_computed_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    block_hashes: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: str | None = None  # "stop" (EOS), "length" or "abort"

    def __post_init__(self) -> None:
        if self.num_prompt_tokens < 0:
            self.num_prompt_tokens = len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def output_ids(self) -> list[int]:
        """Generated tokens only."""
        return self.token_ids[self.num_prompt_tokens :]

    def reaches_end(self, num_new_tokens: int) -> bool:
        """True if computing `num_new_tokens` more tokens covers every known token.

        Only then is there a next token to predict. A prompt chunk that stops
        short (chunked prefill) produces K/V but no sample. The scheduler and the
        runner both rely on this one definition.
        """
        return self.num_computed_tokens + num_new_tokens == len(self.token_ids)
