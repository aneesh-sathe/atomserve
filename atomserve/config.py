# Model configuration: the handful of numbers that define a Qwen3 transformer.
#
# Read from the checkpoint's config.json. We support exactly one architecture
# (Qwen3ForCausalLM), so anything that would change the math (biases, rope
# scaling, sliding windows) is rejected up front instead of silently ignored.

import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_ARCHITECTURE = "Qwen3ForCausalLM"


@dataclass(frozen=True)
class ModelConfig:
    """Shape and numerics of a Qwen3 decoder-only transformer."""

    vocab_size: int  # rows in the embedding / lm_head (151,936 for Qwen3)
    hidden_size: int  # width of the residual stream (1024 for 0.6B)
    intermediate_size: int  # width inside the SwiGLU MLP (3072 for 0.6B)
    num_layers: int  # number of identical decoder layers (28 for 0.6B)
    num_heads: int  # query heads per layer (16 for 0.6B)
    num_kv_heads: int  # key/value heads per layer; < num_heads means GQA (8 for 0.6B)
    head_dim: int  # width of each attention head (128)
    rms_norm_eps: float  # epsilon inside every RMSNorm
    rope_theta: float  # RoPE base frequency (1,000,000)
    tie_word_embeddings: bool  # True: lm_head reuses the embedding matrix
    max_position_embeddings: int  # longest context the model was trained for

    @classmethod
    def from_dict(cls, raw: dict) -> ModelConfig:
        """Build a ModelConfig from a parsed config.json, failing fast on unsupported models."""
        architectures = raw.get("architectures") or []
        if architectures != [SUPPORTED_ARCHITECTURE]:
            raise ValueError(
                f"atomserve only supports {SUPPORTED_ARCHITECTURE}, got {architectures}"
            )
        if raw.get("attention_bias"):
            raise ValueError("attention_bias=True is not supported")
        if raw.get("rope_scaling"):
            raise ValueError("rope_scaling is not supported")
        if raw.get("use_sliding_window"):
            raise ValueError("sliding-window attention is not supported")
        if raw.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act={raw['hidden_act']} is not supported")

        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_layers=raw["num_hidden_layers"],
            num_heads=raw["num_attention_heads"],
            num_kv_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            max_position_embeddings=raw["max_position_embeddings"],
        )

    @classmethod
    def load(cls, model_dir: Path) -> ModelConfig:
        """Read and validate `<model_dir>/config.json`."""
        return cls.from_dict(json.loads((model_dir / "config.json").read_text()))


@dataclass(frozen=True)
class EngineConfig:
    """Serving limits: how much KV memory exists and how much work one step may do."""

    block_size: int = 16  # tokens per KV block
    num_blocks: int = 1024  # size of the KV pool, in blocks
    max_num_seqs: int = 32  # most sequences running (holding KV blocks) at once
    max_num_batched_tokens: int = 512  # token budget of one forward pass (bounds a step)
    max_model_len: int = 16384  # longest sequence (prompt + output) accepted

    def __post_init__(self) -> None:
        if self.max_model_len > self.num_blocks * self.block_size:
            # Otherwise a lone sequence could outgrow the pool with nobody left to preempt.
            raise ValueError(
                f"max_model_len {self.max_model_len} exceeds KV pool capacity "
                f"{self.num_blocks * self.block_size} tokens"
            )
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("max_num_batched_tokens must be >= max_num_seqs (one decode each)")
