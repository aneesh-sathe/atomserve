# Getting a model from the Hugging Face Hub (or disk) onto the GPU.
#
# Two jobs: (1) resolve a model id to a local directory, downloading only the
# files we use, and (2) read the safetensors checkpoint and arrange its tensors
# into the `Weights` structure that model.py computes with.

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from atomserve.config import ModelConfig
from atomserve.model import LayerWeights, Weights

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
    model_dir: Path, config: ModelConfig, device: torch.device, dtype: torch.dtype
) -> Weights:
    """Read every `*.safetensors` shard and build `Weights` on `device` in `dtype`.

    Checkpoint names look like `model.layers.3.self_attn.q_proj.weight`; this is
    the only place that knows them. q/k/v and gate/up are stacked here into one
    matrix each. Tensors are loaded on CPU, then cast and moved
    one at a time so peak memory stays near one model's worth.
    """
    raw: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        raw.update(load_file(shard, device="cpu"))
    if not raw:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")

    def take(name: str) -> torch.Tensor:
        return raw.pop(name).to(device=device, dtype=dtype)

    def stack(*names: str) -> torch.Tensor:
        """Concatenate projection matrices along their output rows (see LayerWeights)."""
        return torch.cat([raw.pop(n) for n in names]).to(device=device, dtype=dtype)

    layers = []
    for i in range(config.num_layers):
        p = f"model.layers.{i}."
        layers.append(
            LayerWeights(
                input_norm=take(p + "input_layernorm.weight"),
                qkv_proj=stack(
                    p + "self_attn.q_proj.weight",
                    p + "self_attn.k_proj.weight",
                    p + "self_attn.v_proj.weight",
                ),
                q_norm=take(p + "self_attn.q_norm.weight"),
                k_norm=take(p + "self_attn.k_norm.weight"),
                o_proj=take(p + "self_attn.o_proj.weight"),
                post_attn_norm=take(p + "post_attention_layernorm.weight"),
                gate_up_proj=stack(p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"),
                down_proj=take(p + "mlp.down_proj.weight"),
            )
        )

    embed = take("model.embed_tokens.weight")
    final_norm = take("model.norm.weight")
    # Tied models may still ship an lm_head copy; the config decides which is used.
    lm_head_raw = raw.pop("lm_head.weight", None)
    if config.tie_word_embeddings:
        lm_head = embed
    elif lm_head_raw is not None:
        lm_head = lm_head_raw.to(device=device, dtype=dtype)
    else:
        raise KeyError("lm_head.weight missing and tie_word_embeddings is False")

    if raw:
        raise KeyError(f"unexpected tensors in checkpoint: {sorted(raw)[:5]} ...")
    return Weights(embed=embed, layers=layers, final_norm=final_norm, lm_head=lm_head)
