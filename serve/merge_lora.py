"""Merge a LoRA adapter into the BF16 base weights, one safetensors shard at a time.

W' = W + (alpha / r) * B @ A for every adapted linear layer, computed in fp32. Streaming
per shard keeps peak RAM at about one shard (~4 GB) instead of the whole 16 GB model, which
matters in WSL. The adapter was trained on the 4-bit base; merging into the original BF16
weights is the standard QLoRA deployment path.

    python serve/merge_lora.py --base Qwen/Qwen3-8B --adapter ~/distill-data/runs/qwen3-8b/adapter \
        --out ~/distill-data/models/qwen3-8b-distill
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    base = Path(snapshot_download(a.base))
    cfg = json.loads((a.adapter / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    lora = load_file(str(a.adapter / "adapter_model.safetensors"))
    # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight -> model.layers.0....weight
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        name, part = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
        pairs.setdefault(name + ".weight", {})[part.split(".")[0]] = v
    print(f"{len(pairs)} adapted matrices, r={r} alpha={alpha} scale={scale}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    merged = 0
    for shard in sorted(base.glob("*.safetensors")):
        tensors = load_file(str(shard))
        for k in tensors:
            if k in pairs:
                w = tensors[k].to(torch.float32)
                w += scale * (pairs[k]["B"].to(torch.float32) @ pairs[k]["A"].to(torch.float32))
                tensors[k] = w.to(tensors[k].dtype)
                merged += 1
        save_file(tensors, str(a.out / shard.name), metadata={"format": "pt"})
        print(f"{shard.name}: done", flush=True)
    assert merged == len(pairs), f"merged {merged} of {len(pairs)}: key names did not line up"
    for f in base.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy(f, a.out / f.name)
    (a.out / "distill_merge.json").write_text(json.dumps(
        {"base": a.base, "adapter": str(a.adapter), "r": r, "alpha": alpha, "scale": scale,
         "matrices": merged}, indent=1))
    print(f"DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
