#!/usr/bin/env bash
# FP8 copy of the untuned Qwen3-8B, the "base" row of the eval and benchmark. CPU only
# (weight rounding, no calibration), so it can run while the GPU labels.
cd /mnt/c/code/extraction-distill
. ~/venvs/distill-quant/bin/activate
CUDA_VISIBLE_DEVICES= python serve/quantize.py --model Qwen/Qwen3-8B --scheme fp8 \
  --out ~/distill-data/models/qwen3-8b-base-fp8 > logs/quant_base_fp8.log 2>&1
