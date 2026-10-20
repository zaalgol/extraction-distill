#!/usr/bin/env bash
# Step 4-5 for the 8B student: QLoRA SFT -> merge into BF16 -> FP8 and AWQ checkpoints.
set -e
cd /mnt/c/code/extraction-distill
M=~/distill-data/models
. ~/venvs/distill-train/bin/activate
python train/train_sft.py --model unsloth/Qwen3-8B-unsloth-bnb-4bit --run qwen3-8b
python serve/merge_lora.py --base Qwen/Qwen3-8B --adapter ~/distill-data/runs/qwen3-8b/adapter --out $M/qwen3-8b-distill
deactivate
. ~/venvs/distill-quant/bin/activate
# FP8 is weight rounding only: CPU keeps the GPU free and the 16 GB model off it
CUDA_VISIBLE_DEVICES= python serve/quantize.py --model $M/qwen3-8b-distill --scheme fp8
echo JOB_DONE
