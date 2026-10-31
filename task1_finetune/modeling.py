"""Model loading, adapter construction and the token-normalized SFT loss for Task 1."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.lora import inject_adapters, target_linears


def load_causal_lm(model_id_or_path: str, dtype: torch.dtype, device: str = "cuda"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
    return model.to(device)


def original_weights(model_id: str, names: list[str]) -> dict[str, torch.Tensor]:
    """Exact FP32 copies of selected ``<module>.weight`` tensors from the original checkpoint files."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    files = sorted(path.glob("*.safetensors"))
    wanted = {f"{n}.weight": n for n in names}
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                if key in wanted:
                    out[wanted[key]] = sf.get_tensor(key).float()
    missing = set(names) - set(out)
    if missing:
        raise KeyError(f"weights not found in checkpoint: {sorted(missing)[:5]}")
    return out


def checkpoint_tensor(path: str | Path, key: str) -> torch.Tensor:
    """Read one tensor (FP32) from a saved HF model directory (single or sharded safetensors)."""
    from safetensors import safe_open

    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        fname = json.loads(index.read_text())["weight_map"][key]
        files = [path / fname]
    else:
        files = sorted(path.glob("*.safetensors"))
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            if key in sf.keys():
                return sf.get_tensor(key).float()
    raise KeyError(key)


def build_model(cfg: dict, method_cfg: dict, device: str = "cuda"):
    """Return (model, info). Full FT: FP32 master weights. Adapters: frozen FP16 backbone + FP32 adapters."""
    kind = method_cfg["kind"]
    info: dict = {"kind": kind}
    if kind == "full":
        model = load_causal_lm(cfg["model_id"], torch.float32, device)
    else:
        model = load_causal_lm(cfg["model_id"], torch.float16, device)
        for p in model.parameters():
            p.requires_grad_(False)
        acfg = cfg["adapter"]
        ref = None
        if kind == "oplora":
            names = [n for n, _ in target_linears(model, acfg["targets"])]
            ref = original_weights(cfg["model_id"], names)
        info["adapter"] = inject_adapters(model, acfg["targets"], method_cfg["r"], method_cfg["alpha"],
                                          acfg["dropout"], seed=cfg["seed"],
                                          method="lora" if kind == "lora" else "oplora",
                                          protect_k=method_cfg.get("protect_k"), reference_weights=ref)
        del ref
    model.config.use_cache = False
    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["total_params_unique"] = sum(p.numel() for p in model.parameters())
    return model, info


def _ce_chunk(h: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, w)
    return F.cross_entropy(logits.float(), y, reduction="sum")


def sft_loss_sum(model, input_ids, attention_mask, labels, chunk_tokens: int) -> tuple[torch.Tensor, int]:
    """Summed next-token NLL over supervised positions (response + end-of-turn).

    The LM head and cross-entropy are evaluated only at supervised positions and in checkpointed chunks, so
    the full (batch x length x 151,936) logit tensor is never materialized. Mathematically identical to the
    standard shifted cross-entropy with prompt/padding positions masked.
    """
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = hidden[:, :-1]
    y = labels[:, 1:]
    mask = y != -100
    h_sel, y_sel = h[mask], y[mask]
    w = model.lm_head.weight
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, y_sel.numel(), chunk_tokens):
        total = total + checkpoint(_ce_chunk, h_sel[i:i + chunk_tokens], y_sel[i:i + chunk_tokens], w,
                                   use_reentrant=False)
    return total, int(y_sel.numel())
