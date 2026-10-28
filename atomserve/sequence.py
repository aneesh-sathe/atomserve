# A sequence: one request's tokens and where its KV cache lives.
#
# The runner reads this state to build a batch; whoever drives generation
# (generate.py for now, the scheduler from M5) is the only one that changes it.
# Status, sampling parameters and stop conditions join it in M5.

from dataclasses import dataclass, field


@dataclass
class Sequence:
    """Mutable state of one sequence being generated.

    token_ids:           prompt followed by every token generated so far
    num_computed_tokens: leading tokens whose K/V are already in the pool; the
                         rest still have to go through the model
    block_table:         physical KV block of each logical block, in order
                         (logical block i holds positions i*block_size ...)
    """

    token_ids: list[int]
    num_computed_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
