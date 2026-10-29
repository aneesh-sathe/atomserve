# Tests for the command-line surface (argument parsing only; no model is loaded).

import pytest

from atomserve import __version__
from atomserve.cli import build_parser, main


def test_help_exits_cleanly(capsys):
    """`atomserve --help` prints usage and exits with status 0."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: atomserve" in capsys.readouterr().out


def test_version_flag(capsys):
    """`atomserve --version` reports the package version."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_model_is_required():
    """Running without a model id is a usage error (argparse exits with status 2)."""
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_parses_model_id():
    """The positional argument is captured verbatim as the model id."""
    assert build_parser().parse_args(["Qwen/Qwen3-0.6B"]).model == "Qwen/Qwen3-0.6B"


def test_kv_pool_is_sized_from_the_memory_budget():
    """4 GiB of bf16 KV at 1.75 MiB per 16-token block is 2340 blocks; the context
    is capped by both the pool and the model's trained length."""
    from atomserve.cli import engine_config_for
    from atomserve.config import ModelConfig

    qwen3_06b = ModelConfig(151936, 1024, 3072, 28, 16, 8, 128, 1e-6, 1e6, True, 40960)
    config = engine_config_for(build_parser().parse_args(["m", "--kv-mem-gb", "4"]), qwen3_06b)
    assert config.num_blocks == 2340
    assert config.max_model_len == 2340 * 16
    small = engine_config_for(build_parser().parse_args(["m", "--kv-mem-gb", "1"]), qwen3_06b)
    assert small.max_model_len == small.num_blocks * 16 < 40960
