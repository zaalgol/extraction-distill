#!/usr/bin/env bash
# The smaller student: Qwen3-4B fits the GPU in BF16, so it is served unquantized.
set -e
cd /mnt/c/code/extraction-distill
. ~/venvs/distill-train/bin/activate
python train/train_sft.py --model unsloth/Qwen3-4B-unsloth-bnb-4bit --run qwen3-4b --batch 4 --grad-accum 4
python serve/merge_lora.py --base Qwen/Qwen3-4B --adapter ~/distill-data/runs/qwen3-4b/adapter --out ~/distill-data/models/qwen3-4b-distill
echo JOB_DONE
