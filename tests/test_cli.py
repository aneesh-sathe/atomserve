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
