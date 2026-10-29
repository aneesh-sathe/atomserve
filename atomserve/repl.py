# The chat REPL: an interactive terminal client of the engine.
#
# Each turn appends the user's message to the conversation, renders the
# *whole* conversation through the chat template, submits it as one prompt and
# streams the reply back as it is generated. The model is stateless between
# requests, so the history is resent every turn; prefix caching is what makes
# that cheap, since every earlier turn's KV blocks are reused.
#
# Ctrl-C while the model is replying aborts that request, not the program.

import sys
from collections.abc import Callable
from dataclasses import replace
from typing import TextIO

from atomserve.engine import Engine, RequestHandle
from atomserve.sequence import NON_THINKING_PARAMS, THINKING_PARAMS, SamplingParams
from atomserve.tokenizer import ChatTokenizer, IncrementalDetokenizer

TUNABLE = {"temperature": float, "top_p": float, "top_k": int, "presence_penalty": float}
TUNABLE |= {"max_tokens": int, "seed": int}

HELP = """\
Commands:
  /think [on|off]   toggle thinking mode (the model reasons before answering)
  /params [k=v ..]  show or set sampling: temperature top_p top_k presence_penalty
                    max_tokens seed ("/params reset" restores the defaults)
  /reset            forget the conversation
  /stats            timing of the last reply and engine state
  /exit, /bye       quit (or Ctrl-D)
Start and end a message with \"\"\" to write several lines."""


class ReplyPrinter:
    """Writes streamed reply text to the terminal, showing the <think> part dimmed.

    The tags themselves are hidden, as is the whitespace around them, so a
    reply prints as: dimmed reasoning, a blank line, the answer. Whitespace at
    the end of a delta is held back until something else follows, because it
    may turn out to be the whitespace just before a tag.
    """

    def __init__(self, out: TextIO, color: bool):
        self.out = out
        self.dim, self.reset = ("\033[2m", "\033[0m") if color else ("", "")
        self.thinking = False
        self.printed_any = False  # anything shown since the last tag
        self.pending_ws = ""  # trailing whitespace not yet shown

    def write(self, text: str) -> None:
        """Show one delta of reply text, handling any think tags in it."""
        for part in _split_tags(text):
            if part == "<think>":
                self.out.write(self.dim)
                self.thinking, self.printed_any, self.pending_ws = True, False, ""
            elif part == "</think>":
                self.out.write(self.reset + ("\n\n" if self.printed_any else ""))
                self.thinking, self.printed_any, self.pending_ws = False, False, ""
            else:
                self._write_text(part)
        self.out.flush()

    def _write_text(self, text: str) -> None:
        if not self.printed_any:
            text = text.lstrip()  # whitespace right after a tag or at the start
        body = text.rstrip()
        if body:
            self.out.write(self.pending_ws + body)
            self.printed_any = True
            self.pending_ws = text[len(body) :]
        elif self.printed_any:
            self.pending_ws += text

    def close(self, note: str = "") -> None:
        """End the reply: leave dim mode, drop trailing whitespace, end the line."""
        self.out.write(self.reset + (f"\n{self.dim}{note}{self.reset}" if note else "") + "\n")
        self.out.flush()


def _split_tags(text: str) -> list[str]:
    """Split text into plain parts and standalone <think> / </think> tags."""
    parts, rest = [], text
    while rest:
        i = min((j for j in (rest.find("<think>"), rest.find("</think>")) if j >= 0), default=-1)
        if i < 0:
            parts.append(rest)
            break
        tag = "<think>" if rest.startswith("<think>", i) else "</think>"
        parts.extend([rest[:i], tag] if i else [tag])
        rest = rest[i + len(tag) :]
    return parts


class Repl:
    """Conversation state plus the command loop."""

    def __init__(
        self,
        engine: Engine,
        tokenizer: ChatTokenizer,
        input_fn: Callable[[str], str] = input,
        out: TextIO = sys.stdout,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.input = input_fn
        self.out = out
        self.color = out.isatty()
        self.messages: list[dict] = []
        self.thinking = True
        self.overrides: dict = {}
        self.last: RequestHandle | None = None

    def params(self, num_prompt_tokens: int) -> SamplingParams:
        """Mode defaults plus user overrides; by default a reply may fill the context."""
        base = THINKING_PARAMS if self.thinking else NON_THINKING_PARAMS
        room = self.engine.scheduler.config.max_model_len - num_prompt_tokens
        return replace(base, **{"max_tokens": room, **self.overrides})

    def run(self) -> None:
        """Read-eval-print until /exit or Ctrl-D."""
        while True:
            try:
                line = self._read_message()
            except EOFError:
                self.out.write("\n")
                return
            except KeyboardInterrupt:
                self.out.write("\nUse Ctrl-D or /bye to exit.\n")
                continue
            if not line.strip():
                continue
            if line.startswith("/"):
                if self.command(line.strip()) == "exit":
                    return
                continue
            self.chat(line)

    def _read_message(self) -> str:
        """One line, or several between a pair of triple quotes."""
        line = self.input(">>> ")
        if not line.startswith('"""'):
            return line
        lines, rest = [], line[3:]
        while not rest.endswith('"""'):
            lines.append(rest)
            rest = self.input("... ")
        lines.append(rest[:-3])
        return "\n".join(lines).strip("\n")

    def chat(self, text: str) -> None:
        """Send one user message and stream the reply."""
        messages = [*self.messages, {"role": "user", "content": text}]
        prompt = self.tokenizer.encode_chat(messages, enable_thinking=self.thinking)
        try:
            handle = self.engine.submit(prompt, self.params(len(prompt)))
        except ValueError as error:
            self.out.write(f"error: {error}\n")
            return
        self.last = handle
        printer = ReplyPrinter(self.out, self.color)
        detok = IncrementalDetokenizer(self.tokenizer)
        reply = []
        try:
            for token in handle:
                delta = detok.push(token)
                reply.append(delta)
                printer.write(delta)
        except KeyboardInterrupt:
            handle.abort()
            for _ in handle:  # drain until the engine confirms the abort
                pass
            printer.close("[stopped]")
            return  # a cut-off reply is not added to the conversation
        reply.append(detok.flush())
        printer.write(reply[-1])
        printer.close(
            "[reply cut off at the length limit]" if handle.finish_reason == "length" else ""
        )
        self.messages = [*messages, {"role": "assistant", "content": "".join(reply)}]

    def command(self, line: str) -> str | None:
        """Run a slash command. Returns "exit" to quit."""
        name, *args = line.split()
        if name in ("/exit", "/bye", "/quit"):
            return "exit"
        if name in ("/help", "/?"):
            self.out.write(HELP + "\n")
        elif name == "/reset":
            self.messages = []
            self.out.write("Conversation cleared.\n")
        elif name == "/think":
            if args and args[0] not in ("on", "off"):
                self.out.write("usage: /think [on|off]\n")
                return None
            self.thinking = (args[0] == "on") if args else not self.thinking
            self.out.write(f"Thinking {'on' if self.thinking else 'off'}.\n")
        elif name == "/params":
            self._params_command(args)
        elif name == "/stats":
            self.out.write(self.stats_text() + "\n")
        else:
            self.out.write(f"Unknown command {name}. Type /help for commands.\n")
        return None

    def _params_command(self, args: list[str]) -> None:
        if args == ["reset"]:
            self.overrides = {}
        for arg in args if args != ["reset"] else []:
            key, _, value = arg.partition("=")
            if key not in TUNABLE or not value:
                self.out.write(f"usage: /params key=value; keys: {' '.join(TUNABLE)}\n")
                return
            try:
                self.overrides[key] = TUNABLE[key](value)
            except ValueError:
                self.out.write(f"{key} needs a {TUNABLE[key].__name__}, got {value!r}\n")
                return
        p = self.params(0)
        max_tokens = p.max_tokens if "max_tokens" in self.overrides else "rest of context"
        self.out.write(
            f"temperature={p.temperature} top_p={p.top_p} top_k={p.top_k} "
            f"presence_penalty={p.presence_penalty} max_tokens={max_tokens} seed={p.seed}\n"
        )

    def stats_text(self) -> str:
        """Last reply's latency and size, then what the engine is doing."""
        lines = []
        if (h := self.last) is not None and h.finished_at is not None:
            ttft = f"{h.ttft * 1000:.0f} ms" if h.ttft is not None else "-"
            rate = f"{h.decode_tokens_per_s:.1f} tok/s" if h.decode_tokens_per_s else "-"
            lines.append(
                f"last reply: {h.num_output_tokens} tokens, time to first token {ttft}, "
                f"decode {rate}"
            )
            lines.append(
                f"  prompt:   {h.num_prompt_tokens} tokens, {h.num_cached_tokens} from prefix cache"
            )
        s = self.engine.stats()
        lines.append(
            f"engine: {s['running']} running, {s['waiting']} waiting, {s['steps']} steps, "
            f"{s['generated_tokens']} tokens generated"
        )
        lines.append(
            f"  KV cache: {s['kv_blocks_used']}/{s['kv_blocks_total']} blocks ({s['kv_usage']:.1%})"
        )
        return "\n".join(lines)
