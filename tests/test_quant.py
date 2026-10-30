# int8 weight quantization tests: the rounding error is bounded per row, the
# quantized model computes what its dequantized weights say, and on the real
# model quality stays close to bf16.

import pytest
import torch

from atomserve.config import EngineConfig
from atomserve.engine import create_engine
from atomserve.loader import quantize_weights
from atomserve.quant import Int8Weight, linear, quantize_int8
from atomserve.sequence import SamplingParams
from tests.conftest import TINY_CONFIG, random_weights


def test_rounding_error_is_at_most_half_a_step_per_row():
    w = torch.randn(64, 256) * torch.logspace(-3, 1, 64)[:, None]  # rows of very different scale
    q = quantize_int8(w)
    assert q.q.dtype == torch.int8 and q.q.abs().max() <= 127
    error = (q.dequantize() - w).abs().amax(dim=1)
    assert (error <= q.scale / 2 + 1e-7).all()  # relative to each row's own scale


def test_linear_with_an_int8_weight_uses_the_dequantized_matrix():
    w = quantize_int8(torch.randn(48, 32))
    x, residual = torch.randn(5, 32), torch.randn(5, 48)
    assert torch.allclose(linear(x, w), x @ w.dequantize().T, atol=1e-5)
    assert torch.allclose(linear(x, w, residual), residual + x @ w.dequantize().T, atol=1e-5)


def test_quantized_engine_runs_and_is_mostly_unchanged():
    """The tiny random model, quantized, still produces mostly the same greedy tokens."""
    weights = random_weights(TINY_CONFIG)
    quantized = random_weights(TINY_CONFIG)
    quantize_weights(quantized)
    assert isinstance(quantized.layers[0].qkv_proj, Int8Weight)
    assert quantized.nbytes < weights.nbytes
    config = EngineConfig(block_size=4, num_blocks=64, max_model_len=64)
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]
    plain = create_engine(weights, TINY_CONFIG, config, frozenset()).generate(prompts, params)
    q = create_engine(quantized, TINY_CONFIG, config, frozenset()).generate(prompts, params)
    same = sum(a == b for p, r in zip(plain, q, strict=True) for a, b in zip(p, r, strict=True))
    assert same >= 12  # of 16


@pytest.mark.slow
def test_int8_keeps_real_model_quality(model_dir, config, weights_bf16, full_logits):
    """Perplexity on natural text within 3% of bf16, and the same top-1 token almost always."""
    from atomserve.loader import load_weights
    from atomserve.tokenizer import ChatTokenizer

    text = (
        "The history of the printing press begins in the fifteenth century, when Johannes "
        "Gutenberg combined a screw press with movable metal type, oil-based ink and a mould "
        "for casting letters. Before that, books in Europe were copied by hand, a slow and "
        "costly process that kept them rare. Within a few decades, printing shops had spread "
        "to more than two hundred cities, and millions of volumes were in circulation."
    )
    ids = ChatTokenizer(model_dir).encode(text)
    int8 = load_weights(model_dir, config, weights_bf16.embed.device, torch.bfloat16, "int8")
    ref, quant = full_logits(weights_bf16, config, ids), full_logits(int8, config, ids)
    target = torch.tensor(ids[1:])

    def perplexity(logits):
        return torch.nn.functional.cross_entropy(logits[:-1], target).exp().item()

    assert perplexity(quant) == pytest.approx(perplexity(ref), rel=0.03)
    assert (quant.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9


def test_untied_int8_lm_head_alone_serves_small_and_big_batches():
    """An untied model keeps only the int8 lm_head; few rows and many rows (dequantized in
    slices) both compute hidden @ dequantize(lm_head)^T."""
    import dataclasses

    from atomserve.model import compute_logits

    weights = random_weights(TINY_CONFIG)
    weights = dataclasses.replace(weights, lm_head=torch.randn(TINY_CONFIG.vocab_size, 32))
    quantize_weights(weights, tied=False)
    assert weights.lm_head is None
    head = weights.lm_head_int8.dequantize()
    for rows in (3, 40):
        hidden = torch.randn(rows, 32)
        assert torch.allclose(compute_logits(weights, hidden), hidden @ head.T, atol=1e-5)
