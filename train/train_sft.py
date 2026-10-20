"""Step 4: QLoRA SFT of a Qwen3 student on the teacher's labels, with Unsloth.

Loss on the assistant JSON only. The prompt part is rendered exactly as at serving time
(chat template, add_generation_prompt, enable_thinking=False), so the student learns the
continuation it will actually be asked for, and everything up to it is masked with -100.
10% of documents (by a hash of the id, so the split is stable) are held out for validation
loss. Hyperparameters, the loss curves and the split go to results/train/<run>/.

    python train/train_sft.py --model unsloth/Qwen3-8B-unsloth-bnb-4bit --run qwen3-8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from unsloth import FastLanguageModel  # noqa: I001 - must be imported before transformers

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

REPO = Path(__file__).resolve().parent.parent
DATA = Path.home() / "distill-data"


def is_val(doc_id: str, frac: float) -> bool:
    return int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 1000 < frac * 1000


def encode(tok, rec: dict, max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(rec["messages"][:2], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    # vLLM renders the same template with enable_thinking=False; training on another
    # rendering would teach a continuation the server never asks for.
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), prompt[-80:]
    answer = rec["messages"][2]["content"] + "<|im_end|>\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(a_ids) > max_len:
        return None
    return {"input_ids": p_ids + a_ids, "attention_mask": [1] * (len(p_ids) + len(a_ids)),
            "labels": [-100] * len(p_ids) + a_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-8B-unsloth-bnb-4bit")
    ap.add_argument("--run", default="qwen3-8b")
    ap.add_argument("--data", type=Path, default=DATA / "sft.jsonl")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()

    out = DATA / "runs" / a.run
    res = REPO / "results" / "train" / a.run
    res.mkdir(parents=True, exist_ok=True)

    model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len,
                                                   load_in_4bit=True, dtype=torch.bfloat16)
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=a.seed)

    recs = [json.loads(x) for x in a.data.read_text(encoding="utf-8").splitlines() if x]
    train, val, too_long = [], [], 0
    for r in recs:
        e = encode(tok, r, a.max_len)
        if e is None:
            too_long += 1
            continue
        (val if is_val(r["id"], a.val_frac) else train).append(e)
    lens = sorted(len(e["input_ids"]) for e in train)
    print(f"train {len(train)}  val {len(val)}  dropped (> {a.max_len} tokens) {too_long}  "
          f"tokens p50 {lens[len(lens) // 2]} max {lens[-1]}", flush=True)
    (res / "split.json").write_text(json.dumps(
        {"val_ids": sorted(r["id"] for r in recs if is_val(r["id"], a.val_frac))}, indent=0))

    steps_per_epoch = math.ceil(len(train) / (a.batch * a.grad_accum))
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, lr_scheduler_type="cosine",
        warmup_steps=0.05,   # < 1 means a fraction of all steps (transformers 5)
        weight_decay=0.0, optim="adamw_8bit", bf16=True,
        logging_steps=2, eval_strategy="steps", eval_steps=max(1, steps_per_epoch // 3),
        save_strategy="epoch", save_total_limit=3,
        train_sampling_strategy="group_by_length",
        report_to="none", seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(train),
                      eval_dataset=Dataset.from_list(val),
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                                           label_pad_token_id=-100))
    t0 = time.time()
    trainer.train()
    final = trainer.evaluate()
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))

    (res / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (res / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "train_docs": len(train), "val_docs": len(val), "dropped_too_long": too_long,
        "steps": trainer.state.global_step, "train_minutes": round((time.time() - t0) / 60, 1),
        "final_eval_loss": final["eval_loss"],
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "lora_target_modules": "all linear (q,k,v,o,gate,up,down)", "lora_dropout": 0,
        "scheduler": "cosine, 5% warmup", "optimizer": "adamw_8bit", "precision": "bf16",
        "quantization": "4-bit NF4 base (QLoRA)"}, indent=1))
    print(f"DONE eval_loss {final['eval_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
