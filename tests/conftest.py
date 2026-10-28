# Shared fixtures for the slow tests that need real model weights.
# Session-scoped so the checkpoint is read once per test run.

from pathlib import Path

import pytest
import torch

from atomserve.config import ModelConfig
from atomserve.loader import default_device, load_weights, resolve_model_dir
from atomserve.model import Weights

MODEL_ID = "Qwen/Qwen3-0.6B"


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
