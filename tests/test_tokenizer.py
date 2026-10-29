# Tokenizer tests against the real Qwen3 tokenizer files. Our jinja2 rendering must
# match transformers' apply_chat_template exactly, and streamed text must never
# contain a broken character.

import pytest
from transformers import AutoTokenizer

from atomserve.tokenizer import ChatTokenizer, IncrementalDetokenizer

pytestmark = pytest.mark.slow

CONVERSATIONS = [
    [{"role": "user", "content": "Hi!"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi!"}],
    [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "<think>\nThey greet me.\n</think>\n\nHello!"},
        {"role": "user", "content": "What is 2+2?"},
    ],
]


@pytest.fixture(scope="module")
def tokenizers(model_dir):
    return ChatTokenizer(model_dir), AutoTokenizer.from_pretrained(model_dir)


@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("messages", CONVERSATIONS)
def test_chat_template_matches_transformers(tokenizers, messages, enable_thinking):
    ours, hf = tokenizers
    kwargs = {"add_generation_prompt": True, "enable_thinking": enable_thinking}
    text = hf.apply_chat_template(messages, tokenize=False, **kwargs)
    assert ours.render_chat(messages, enable_thinking) == text
    assert ours.encode_chat(messages, enable_thinking) == hf(text).input_ids


def test_thinking_off_prefills_an_empty_think_block(tokenizers):
    ours, _ = tokenizers
    prompt = ours.render_chat(CONVERSATIONS[0], enable_thinking=False)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_eos_tokens_come_from_generation_config(tokenizers):
    ours, _ = tokenizers
    assert ours.eos_token_ids == {151645, 151643}  # <|im_end|>, <|endoftext|>


@pytest.mark.parametrize(
    "text", ["Hi 👋🏽 世界 - naïve café 🇮🇳!", "数学很有趣。😀😀", "plain ascii"]
)
def test_streamed_text_is_never_broken(tokenizers, text):
    """Token-by-token deltas join to the full text, and no delta holds a U+FFFD."""
    ours, _ = tokenizers
    detok = IncrementalDetokenizer(ours)
    deltas = [detok.push(t) for t in ours.encode(text)] + [detok.flush()]
    assert "".join(deltas) == text
    assert not any("\ufffd" in d for d in deltas)


def test_special_tokens_are_not_streamed(tokenizers):
    ours, _ = tokenizers
    detok = IncrementalDetokenizer(ours)
    ids = [*ours.encode("Hello"), 151645]  # <|im_end|>
    assert "".join(detok.push(t) for t in ids) + detok.flush() == "Hello"
