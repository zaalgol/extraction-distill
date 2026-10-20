"""Smoke test: 4-bit load + LoRA + 5 optimizer steps with Unsloth on the local GPU."""
import time

import torch
from unsloth import FastLanguageModel

t0 = time.time()
model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3-0.6B", max_seq_length=1024,
                                               load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
batch = tok(["Extraction smoke test. " * 60] * 2, return_tensors="pt").to("cuda")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train()
for step in range(5):
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"step {step} loss {loss.item():.3f}")
print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, {time.time() - t0:.0f}s")
print("TRAIN SMOKE OK")
