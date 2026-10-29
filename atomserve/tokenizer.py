# Text <-> tokens, and the chat template that turns a conversation into a prompt.
#
# The model never sees "messages"; it sees one token string in Qwen's ChatML
# format, rendered by the Jinja template that ships in tokenizer_config.json:
#
#     <|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n
#
# and it answers by continuing that text until it emits <|im_end|>. Rendering
# the template ourselves with jinja2 (instead of through transformers) keeps
# exactly what the model is fed in plain sight.
#
# Qwen3 "thinks" by default: it first writes its reasoning between <think> and
# </think>, then the answer. `enable_thinking=False` makes the template
# pre-fill an empty think block so the model answers directly. <think> and
# </think> are ordinary tokens, so they survive decoding and clients can
# recognise (and e.g. dim) the reasoning part.

import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


class ChatTokenizer:
    """The model's BPE tokenizer, chat template and end-of-sequence tokens."""

    def __init__(self, model_dir: Path):
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text())
        # Same Jinja settings as transformers, so the rendered text is identical.
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self._template = env.from_string(tokenizer_config["chat_template"])

        # generation_config.json lists every token that ends a reply.
        generation_config = json.loads((model_dir / "generation_config.json").read_text())
        eos = generation_config["eos_token_id"]
        self.eos_token_ids: frozenset[int] = frozenset(eos if isinstance(eos, list) else [eos])

    def encode(self, text: str) -> list[int]:
        """Token ids of `text`. Special tokens written in the text (<|im_start|>) are parsed."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        """Text of `ids`, dropping special tokens such as <|im_end|>."""
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def render_chat(self, messages: list[dict], enable_thinking: bool = True) -> str:
        """The prompt text for a conversation, ending where the assistant's reply starts.

        messages: [{"role": "system" | "user" | "assistant", "content": str}, ...]

        Earlier assistant turns may include their <think> block; the template
        strips it, as the model was trained to only see final answers in history.
        """
        return self._template.render(
            messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking
        )

    def encode_chat(self, messages: list[dict], enable_thinking: bool = True) -> list[int]:
        """`render_chat` then `encode`: the prompt token ids for a conversation."""
        return self.encode(self.render_chat(messages, enable_thinking))


class IncrementalDetokenizer:
    """Turns a stream of token ids into a stream of text, never emitting broken characters.

    Byte-level BPE splits text into bytes, not characters: an emoji or a CJK
    character is often 2-4 tokens, and decoding only some of them yields the
    replacement character U+FFFD. So each push decodes a short window of
    recent tokens and emits only text that is new *and* complete:

        ids:   ... [prefix_offset ......... read_offset ......] new
                    \\___ already emitted __/ \\_ not yet emitted _/

    Decoding a window (rather than the new token alone) also keeps any
    tokenizer whose output depends on the neighbouring token correct.
    Cost per token is O(window), not O(sequence length).
    """

    def __init__(self, tokenizer: ChatTokenizer):
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._prefix_offset = 0  # start of the decode window
        self._read_offset = 0  # end of what has been emitted

    def push(self, token_id: int) -> str:
        """Add one token; return the newly completed text ("" if nothing complete yet)."""
        self._ids.append(token_id)
        emitted = self._tokenizer.decode(self._ids[self._prefix_offset : self._read_offset])
        window = self._tokenizer.decode(self._ids[self._prefix_offset :])
        if len(window) > len(emitted) and not window.endswith("\ufffd"):
            self._prefix_offset, self._read_offset = self._read_offset, len(self._ids)
            return window[len(emitted) :]
        return ""

    def flush(self) -> str:
        """Whatever is still held back at the end of the stream, broken or not."""
        emitted = self._tokenizer.decode(self._ids[self._prefix_offset : self._read_offset])
        window = self._tokenizer.decode(self._ids[self._prefix_offset :])
        self._prefix_offset = self._read_offset = len(self._ids)
        return window[len(emitted) :]
