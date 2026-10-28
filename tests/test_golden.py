# Golden tests: atomserve's model must reproduce Hugging Face transformers' Qwen3.
#
# The reference runs in fp32 on CPU with eager attention, the most literal
# implementation available. Ours runs in fp32 on MPS, so any difference beyond
# floating-point noise means a bug in our math, not a dtype effect.

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atomserve.generate import generate_greedy

pytestmark = pytest.mark.slow

RAW_PROMPT = "The capital of France is"
CHAT_MESSAGES = [{"role": "user", "content": "Explain what a KV cache is in one sentence."}]


@pytest.fixture(scope="module")
def hf_model(model_dir):
    return AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def prompts(model_dir) -> dict[str, list[int]]:
    """A raw-text prompt and a chat-templated prompt, as token ids."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    chat = tok.apply_chat_template(CHAT_MESSAGES, add_generation_prompt=True, tokenize=False)
    return {"raw": tok(RAW_PROMPT).input_ids, "chat": tok(chat).input_ids}


def hf_logits(hf_model, ids: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        return hf_model(torch.tensor([ids])).logits[0].float()


@pytest.mark.parametrize("kind", ["raw", "chat"])
def test_fp32_logits_match_reference(kind, prompts, weights_fp32, config, hf_model, full_logits):
    """Every position's logits agree with the reference to within fp32 noise."""
    ids = prompts[kind]
    ours, ref = full_logits(weights_fp32, config, ids), hf_logits(hf_model, ids)
    assert ours.shape == ref.shape
    assert (ours - ref).abs().max() < 1e-3  # observed ~1.3e-4 on logits of magnitude ~37
    assert torch.equal(ours.argmax(-1), ref.argmax(-1))


@pytest.mark.parametrize("kind", ["raw", "chat"])
def test_greedy_generation_matches_reference(kind, prompts, weights_fp32, config, hf_model):
    """32 greedy tokens are identical to transformers' own greedy generate()."""
    ids = prompts[kind]
    ours = generate_greedy(weights_fp32, config, ids, max_new_tokens=32)
    with torch.inference_mode():
        ref = hf_model.generate(
            torch.tensor([ids]), max_new_tokens=32, min_new_tokens=32, do_sample=False
        )[0, len(ids) :].tolist()
    assert ours == ref


def test_bf16_logits_close_to_reference(prompts, weights_bf16, config, hf_model, full_logits):
    """In bf16 (the serving dtype) logits stay close and the top prediction agrees."""
    ids = prompts["chat"]
    ours, ref = full_logits(weights_bf16, config, ids), hf_logits(hf_model, ids)
    cosine = torch.nn.functional.cosine_similarity(ours, ref, dim=-1)
    assert cosine.min() > 0.99
    assert ours[-1].argmax() == ref[-1].argmax()
