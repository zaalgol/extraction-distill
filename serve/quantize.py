"""Quantize a model with llm-compressor for vLLM: FP8 (W8A8 dynamic) or AWQ (W4A16).

Qwen3-8B in BF16 (16.4 GB) does not fit the 16 GB GPU, so every served 8B model is
quantized. FP8 needs no calibration; AWQ calibrates on extraction prompts + teacher answers
from the training split (in-domain text, never the gold set).

    python serve/quantize.py --model ~/distill-data/models/qwen3-8b-distill --scheme awq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path.home() / "distill-data"


def calibration(tok, n: int, max_len: int) -> Dataset:
    recs = [json.loads(x) for x in (DATA / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    # same hash split as training: calibrate on training documents only
    recs = [r for r in recs if int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 1000 >= 100]
    random.Random(0).shuffle(recs)
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False)
             for r in recs[:n]]
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=max_len)
    return Dataset.from_dict({"input_ids": enc["input_ids"],
                              "attention_mask": enc["attention_mask"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scheme", choices=["fp8", "awq"], required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    out = a.out or Path(str(Path(a.model).expanduser()).rstrip("/") + f"-{a.scheme}")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto")
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.scheme == "fp8":
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        oneshot(model=model, recipe=recipe)
    else:
        recipe = AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])
        oneshot(model=model, recipe=recipe, dataset=calibration(tok, a.samples, a.max_len),
                max_seq_length=a.max_len, num_calibration_samples=a.samples)
    model.save_pretrained(str(out), save_compressed=True)
    tok.save_pretrained(str(out))
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
