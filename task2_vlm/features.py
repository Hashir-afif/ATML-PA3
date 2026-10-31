"""Cache SigLIP patch features for every selected image (run once, before any Qwen training).

    python -m task2_vlm.features --config configs/task2.yaml

``SiglipVisionModel(...).last_hidden_state`` is the encoder output after SigLIP's final ``post_layernorm``;
the attention-pooling head is not used. That gives the 196 final patch features (14 x 14 grid, row-major
spatial order) with the final layer normalization applied exactly once, which is checked numerically against
``post_layernorm(hidden_states[-1])``. Preprocessing is the deterministic SigLIP processor (bicubic resize to
224, rescale, normalize). Features are stored in FP16 as one memory-mapped array per split plus an index; the
vision encoder is then released so it never shares GPU memory with Qwen. Timing and memory are logged
separately from training.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image

from common.utils import environment_info, load_config, load_json, log, peak_memory, rel, reset_peak_memory, save_json
from task2_vlm.data_prep import image_path


def feature_files(cfg: dict, split: str):
    d = rel(cfg["data"]["feature_dir"])
    return d / f"siglip_{split}.npy", d / f"siglip_{split}_index.json"


class FeatureStore:
    """image (split, id) -> FP16 array (196, 768), memory-mapped."""

    def __init__(self, cfg: dict):
        self.arrays, self.index = {}, {}
        for split in ("train2014", "val2014"):
            arr, idx = feature_files(cfg, split)
            self.arrays[split] = np.load(arr, mmap_mode="r")
            self.index[split] = {int(k): v for k, v in load_json(idx).items()}

    def get(self, split: str, image_id: int) -> np.ndarray:
        return self.arrays[split][self.index[split][int(image_id)]]

    def batch(self, items: list[tuple[str, int]]) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.get(s, i) for s, i in items]))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import SiglipImageProcessor, SiglipVisionModel

    man = load_json(cfg["data"]["manifest"])
    proc = SiglipImageProcessor.from_pretrained(cfg["vision_id"])
    reset_peak_memory()
    t_load = time.perf_counter()
    model = SiglipVisionModel.from_pretrained(cfg["vision_id"], torch_dtype=torch.float32).cuda().eval()
    load_seconds = time.perf_counter() - t_load
    stats = {"vision_id": cfg["vision_id"], "dtype_stored": "float16", "encoder_dtype": "float32",
             "processor": proc.to_dict(), "splits": {}, "encoder_load_seconds": load_seconds}
    checked = False
    for split in ("train2014", "val2014"):
        arr_path, idx_path = feature_files(cfg, split)
        ids = man["images"][split]
        if arr_path.exists() and idx_path.exists() and len(load_json(idx_path)) == len(ids):
            log(f"{split}: features exist")
            continue
        arr_path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float16, shape=(len(ids), 196, 768))
        t0 = time.perf_counter()
        for s in range(0, len(ids), args.batch_size):
            chunk = ids[s:s + args.batch_size]
            ims = [Image.open(image_path(cfg, split, i)).convert("RGB") for i in chunk]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda()
            calls = []
            hook = None if checked else model.vision_model.post_layernorm.register_forward_hook(
                lambda mod, inp, out: calls.append(out.detach()))
            o = model(pixel_values=px)
            feats = o.last_hidden_state  # post_layernorm applied once
            assert feats.shape[1:] == (196, 768), feats.shape
            if not checked:
                hook.remove()
                # the final LayerNorm runs exactly once per forward and its output is what we store
                stats["post_layernorm_calls_per_forward"] = len(calls)
                stats["layernorm_once_max_abs_diff"] = float((calls[0] - feats).abs().max())
                assert len(calls) == 1 and stats["layernorm_once_max_abs_diff"] == 0.0
                checked = True
            out[s:s + len(chunk)] = feats.half().cpu().numpy()
            if (s // args.batch_size) % 20 == 0:
                log(f"{split}: {s + len(chunk)}/{len(ids)}")
        out.flush()
        del out
        save_json({str(i): k for k, i in enumerate(ids)}, idx_path)
        stats["splits"][split] = {"n_images": len(ids), "seconds": time.perf_counter() - t0,
                                  "bytes": arr_path.stat().st_size}
    stats["memory"] = peak_memory()
    stats["environment"] = environment_info()
    del model
    torch.cuda.empty_cache()
    p = rel(cfg["paths"]["results"]) / "feature_caching.json"
    if stats["splits"]:
        save_json(stats, p)
    log(f"feature caching done: {stats['splits']}")


if __name__ == "__main__":
    main()
