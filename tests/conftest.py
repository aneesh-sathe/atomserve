# Shared fixtures.
#
# The tiny model is random weights at toy size on CPU: it runs the real code
# paths (paged pool, runner, flat batch) in milliseconds, so invariants about
# the engine can be fast tests. The real-model fixtures are session-scoped so
# the checkpoint is read once per run.

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from atomserve.block_allocator import blocks_for_tokens
from atomserve.config import EngineConfig, ModelConfig
from atomserve.engine import create_engine
from atomserve.kv_cache import KVPool
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import LayerWeights, Weights
from atomserve.runner import ModelRunner, Row
from atomserve.sequence import SamplingParams

MODEL_ID = "Qwen/Qwen3-0.6B"

TINY_CONFIG = ModelConfig(
    vocab_size=50,
    hidden_size=32,
    intermediate_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
    tie_word_embeddings=True,
    max_position_embeddings=256,
)


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    return TINY_CONFIG


def random_weights(c: ModelConfig, device: str = "cpu", seed: int = 0) -> Weights:
    """Seeded random fp32 weights of the right shapes for config `c`."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * 0.2).to(device)

    def norm(dim: int) -> torch.Tensor:
        return 1.0 + rand(dim)

    q_dim, kv_dim = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim
    layers = [
        LayerWeights(
            input_norm=norm(c.hidden_size),
            qkv_proj=rand(q_dim + 2 * kv_dim, c.hidden_size),
            q_norm=norm(c.head_dim),
            k_norm=norm(c.head_dim),
            o_proj=rand(c.hidden_size, q_dim),
            post_attn_norm=norm(c.hidden_size),
            gate_up_proj=rand(2 * c.intermediate_size, c.hidden_size),
            down_proj=rand(c.hidden_size, c.intermediate_size),
        )
        for _ in range(c.num_layers)
    ]
    embed = rand(c.vocab_size, c.hidden_size)
    return Weights(embed=embed, layers=layers, final_norm=norm(c.hidden_size), lm_head=embed)


@pytest.fixture(scope="session")
def tiny_weights() -> Weights:
    """Random fp32 weights for TINY_CONFIG on CPU (seeded, so every run is the same)."""
    return random_weights(TINY_CONFIG)


@pytest.fixture(scope="session")
def full_logits() -> Callable[[Weights, ModelConfig, list[int]], torch.Tensor]:
    """f(weights, config, ids) -> logits at every position [len(ids), vocab], on CPU.

    Runs `ids` as one fresh prefill in a throwaway pool, asking for a logits row
    per token instead of just the last one: the uncached reference that cached,
    paged and batched runs are compared against.
    """

    def run(weights: Weights, config: ModelConfig, ids: list[int]) -> torch.Tensor:
        block_size = 16
        num_blocks = blocks_for_tokens(len(ids), block_size)
        device, dtype = weights.embed.device, weights.embed.dtype
        runner = ModelRunner(weights, config, KVPool(config, num_blocks, block_size, device, dtype))
        row = Row(list(ids), 0, list(range(num_blocks)), num_logits=len(ids))
        logits, _ = runner.forward([row])
        return logits.cpu()

    return run


@pytest.fixture(scope="session")
def greedy() -> Callable[..., list[list[int]]]:
    """f(weights, config, prompts, max_tokens, **engine_kwargs) -> output ids per prompt.

    Greedy generation through the real engine (scheduler + paged pool + runner),
    all prompts batched together. EOS is ignored so every output has exactly
    `max_tokens` tokens and runs can be compared token for token.
    """

    def run(weights, config, prompts, max_tokens, **engine_kwargs) -> list[list[int]]:
        engine_config = EngineConfig(
            **{"num_blocks": 256, "max_model_len": 4096, "block_size": 16, **engine_kwargs}
        )
        engine = create_engine(weights, config, engine_config, frozenset())
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        return engine.generate(prompts, params)

    return run


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Local directory of the test model (downloaded on first use)."""
    return resolve_model_dir(MODEL_ID)


@pytest.fixture(scope="session")
def config(model_dir) -> ModelConfig:
    return ModelConfig.load(model_dir)


@pytest.fixture(scope="session")
def weights_fp32(model_dir, config) -> Weights:
    """Our weights in fp32 on the GPU: used to compare against the reference exactly."""
    return load_weights(model_dir, config, default_device(), torch.float32)


@pytest.fixture(scope="session")
def weights_bf16(model_dir, config) -> Weights:
    """Our weights in bf16 on the GPU: the dtype atomserve actually serves in."""
    return load_weights(model_dir, config, default_device(), torch.bfloat16)
