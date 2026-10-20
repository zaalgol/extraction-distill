#!/usr/bin/env bash
# Weights for the optional Qwen3-4B student (4-bit for training, BF16 for the merge).
cd /mnt/c/code/extraction-distill
. ~/venvs/distill-train/bin/activate
{ hf download unsloth/Qwen3-4B-unsloth-bnb-4bit && hf download Qwen/Qwen3-4B; } > logs/download_4b.log 2>&1
