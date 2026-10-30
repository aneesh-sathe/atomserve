# Getting a model from the Hugging Face Hub (or disk) onto the GPU.
#
# Two jobs: (1) resolve a model id to a local directory, downloading only the
# files we use, and (2) read the safetensors checkpoint and arrange its tensors
# into the `Weights` structure that model.py computes with.

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from atomserve.config import ModelConfig
from atomserve.model import LayerWeights, Linear, Weights
from atomserve.quant import quantize_int8

# Config, tokenizer and weight files. Skips README, images, etc.
_NEEDED_FILES = ["*.json", "*.safetensors", "*.txt"]


def default_device() -> torch.device:
    """The Apple GPU (MPS) when available, else CPU (only useful for tests)."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def missing_files(model_dir: Path) -> list[str]:
    """Files atomserve reads that `model_dir` lacks (empty when it is complete).

    Checked because a Hub cache entry can be partial: another tool may have
    fetched only config.json, and the offline lookup would still return it.
    """
    required = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
    index = model_dir / "model.safetensors.index.json"
    if index.exists():  # sharded checkpoint: every shard the index names
        required += sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif not any(model_dir.glob("*.safetensors")):
        required.append("model.safetensors")
    return [name for name in required if not (model_dir / name).exists()]


def resolve_model_dir(model: str) -> Path:
    """Map a local directory or a Hub id like "Qwen/Qwen3-0.6B" to a local directory.

    Hub downloads land in the standard Hugging Face cache, so they are shared
    with other tools and only fetched once. A complete cached copy is used
    without touching the network, which keeps startup fast and lets atomserve
    run offline; an incomplete one is topped up.
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    try:
        cached = Path(snapshot_download(model, allow_patterns=_NEEDED_FILES, local_files_only=True))
        if not missing_files(cached):
            return cached
    except FileNotFoundError:  # not cached at all (LocalEntryNotFoundError subclasses this)
        pass
    return Path(snapshot_download(model, allow_patterns=_NEEDED_FILES))


def load_weights(
    model_dir: Path,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    quantize: str | None = None,
) -> Weights:
    """Read the `*.safetensors` shards and build `Weights` on `device` in `dtype`.

    Checkpoint names look like `model.layers.3.self_attn.q_proj.weight`; this is
    the only place that knows them. q/k/v and gate/up are stacked here into one
    matrix each.

    Tensors are read one at a time from memory-mapped shards, moved to the device
    and, with quantize="int8", quantized right away (see quant.py). On Apple
    silicon CPU and GPU share memory, so reading whole shards first would hold
    the checkpoint twice; streaming keeps the peak near the final model plus one
    matrix. That is what lets Qwen3-8B load as 8 GB of int8 on a 24 GB machine,
    without ever holding its 16 GB bf16 form.
    """
    if quantize not in (None, "int8"):
        raise ValueError(f"unknown quantization {quantize!r}")
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")

    with ExitStack() as stack:
        files = [stack.enter_context(safe_open(s, framework="pt", device="cpu")) for s in shards]
        where = {name: f for f in files for name in f.keys()}  # noqa: SIM118 (safe_open isn't a dict)
        unused = set(where)

        def read(name: str) -> torch.Tensor:
            unused.discard(name)
            return where[name].get_tensor(name)

        def take(name: str) -> torch.Tensor:
            return read(name).to(device=device, dtype=dtype)

        def matrix(*names: str) -> Linear:
            """A projection: several checkpoint matrices stacked along their output rows."""
            w = torch.cat([read(n) for n in names]).to(device=device, dtype=dtype)
            return quantize_int8(w) if quantize else w

        layers = []
        for i in range(config.num_layers):
            p = f"model.layers.{i}."
            layers.append(
                LayerWeights(
                    input_norm=take(p + "input_layernorm.weight"),
                    qkv_proj=matrix(
                        p + "self_attn.q_proj.weight",
                        p + "self_attn.k_proj.weight",
                        p + "self_attn.v_proj.weight",
                    ),
                    q_norm=take(p + "self_attn.q_norm.weight"),
                    k_norm=take(p + "self_attn.k_norm.weight"),
                    o_proj=matrix(p + "self_attn.o_proj.weight"),
                    post_attn_norm=take(p + "post_attention_layernorm.weight"),
                    gate_up_proj=matrix(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"),
                    down_proj=matrix(p + "mlp.down_proj.weight"),
                )
            )
        weights = Weights(
            embed=take("model.embed_tokens.weight"),
            layers=layers,
            final_norm=take("model.norm.weight"),
            lm_head=None,
        )
        # Tied models may still ship an lm_head copy; the config decides which is used.
        unused.discard("lm_head.weight")
        if config.tie_word_embeddings:
            weights.lm_head = weights.embed
        elif "lm_head.weight" in where:
            weights.lm_head = take("lm_head.weight")
        else:
            raise KeyError("lm_head.weight missing and tie_word_embeddings is False")
        if unused:
            raise KeyError(f"unexpected tensors in checkpoint: {sorted(unused)[:5]} ...")

    if quantize:
        quantize_lm_head(weights, tied=config.tie_word_embeddings)
    if device.type == "mps":
        # The allocator keeps freed buffers (the bf16 matrices that became int8) cached
        # for reuse; hand them back so the KV cache and other processes can have them.
        torch.mps.empty_cache()
    return weights


def quantize_lm_head(weights: Weights, tied: bool) -> None:
    """Give the lm_head an int8 copy for small batches (see model.compute_logits).

    A tied lm_head is the embedding too, so its bf16 matrix stays; an untied one
    (1.2 GB in Qwen3-8B) is replaced, and big batches dequantize it in slices.
    """
    weights.lm_head_int8 = quantize_int8(weights.lm_head)
    if not tied:
        weights.lm_head = None


def quantize_weights(weights: Weights, tied: bool = True) -> None:
    """Quantize an already loaded model in place (load_weights(quantize="int8") streams instead)."""
    for layer in weights.layers:
        for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            setattr(layer, name, quantize_int8(getattr(layer, name)))
    quantize_lm_head(weights, tied)
