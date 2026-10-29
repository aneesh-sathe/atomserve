# REPL tests: the reply formatting rules (fast), and a real terminal session that
# drives the `atomserve` binary through a pseudo-terminal like a person would (slow).

import contextlib
import io
import os
import pty
import select
import shutil
import subprocess
import sys
import time

import pytest

from atomserve.repl import ReplyPrinter


def printed(deltas: list[str]) -> str:
    out = io.StringIO()
    printer = ReplyPrinter(out, color=True)
    for delta in deltas:
        printer.write(delta)
    printer.close()
    return out.getvalue().replace("\033[2m", "<dim>").replace("\033[0m", "</dim>")


def test_thinking_is_dimmed_and_tags_hidden():
    deltas = ["<think>", "\n", "Let me", " think.", "\n", "</think>", "\n\n", "Answer", "!"]
    assert printed(deltas) == "<dim>Let me think.</dim>\n\nAnswer!</dim>\n"


def test_whitespace_inside_a_reply_is_kept_and_trailing_dropped():
    assert printed(["Line one.", "\n\n", "Line", " two.", "\n"]) == "Line one.\n\nLine two.</dim>\n"


def test_tags_inside_one_delta_are_split():
    assert printed(["<think>hmm</think>ok"]) == "<dim>hmm</dim>\n\nok</dim>\n"


def test_empty_think_block_prints_nothing_for_it():
    assert printed(["<think>", "\n\n", "</think>", "\n\n", "Hi"]) == "<dim></dim>Hi</dim>\n"


# Runs in the child: make the pty our controlling terminal (so Ctrl-C is a real
# SIGINT from the tty driver), then become the program under test.
_CLAIM_TTY = (
    "import fcntl, os, sys, termios; fcntl.ioctl(0, termios.TIOCSCTTY, 0); "
    "os.execv(sys.argv[1], sys.argv[1:])"
)


class Terminal:
    """A child process attached to a pseudo-terminal (its controlling tty).

    Spawned with subprocess rather than pty.fork(): forking a multi-threaded
    process (pytest holds engine threads by now) can deadlock the child.
    """

    def __init__(self, *argv: str):
        self.fd, child_end = pty.openpty()
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _CLAIM_TTY, *argv],
            stdin=child_end,
            stdout=child_end,
            stderr=child_end,
            start_new_session=True,  # setsid: a session of its own, ready to own the tty
        )
        os.close(child_end)
        self.output = b""

    def read_until(self, marker: bytes, timeout: float) -> str:
        """Read output until `marker` appears after what was already read."""
        start, deadline = len(self.output), time.time() + timeout
        while marker not in self.output[start:] and time.time() < deadline:
            if select.select([self.fd], [], [], 0.1)[0]:
                try:
                    chunk = os.read(self.fd, 65536)
                except OSError:
                    break
                self.output += chunk
        assert marker in self.output[start:], self.output[start:].decode(errors="replace")
        return self.output[start:].decode(errors="replace")

    def send(self, text: str) -> None:
        time.sleep(0.3)  # let readline finish setting up the terminal first
        os.write(self.fd, text.encode())

    def wait(self, timeout: float = 20) -> int:
        """Exit code of the child; kills it (and fails) if it does not exit in time."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (code := self.proc.poll()) is not None:
                return code
            if select.select([self.fd], [], [], 0.1)[0]:
                with contextlib.suppress(OSError):
                    self.output += os.read(self.fd, 65536)
        self.proc.kill()
        self.proc.wait()
        raise AssertionError("did not exit: " + self.output[-300:].decode(errors="replace"))


@pytest.mark.slow
def test_terminal_chat_session(model_dir):
    """Two turns without thinking (the second relies on the first), a thinking reply
    stopped with Ctrl-C, a command, and Ctrl-D to quit."""
    term = Terminal(shutil.which("atomserve"), str(model_dir))
    term.read_until(b">>> ", 120)
    term.send("/think off\r")
    term.read_until(b">>> ", 10)
    term.send("My name is Ada. Reply with just OK.\r")
    term.read_until(b">>> ", 60)
    term.send("What is my name? One word.\r")
    assert "Ada" in term.read_until(b">>> ", 60)

    term.send("/think on\r")
    term.read_until(b">>> ", 10)
    term.send("Write a long essay about the history of Rome.\r")
    term.read_until(b"\033[2m", 30)  # dimmed thinking has started
    time.sleep(1)
    term.send("\x03")
    assert "[stopped]" in term.read_until(b">>> ", 30)

    term.send("/stats\r")
    assert "time to first token" in term.read_until(b">>> ", 10)
    term.send("\x04")
    assert term.wait() == 0
