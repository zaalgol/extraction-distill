#!/usr/bin/env bash
# Pilot: the full train -> merge -> FP8 path on the first ~100 labels, to catch pipeline bugs
# before the real run. Its numbers are not reported.
set -e
cd /mnt/c/code/extraction-distill
cp ~/distill-data/sft.jsonl ~/distill-data/sft_pilot.jsonl
. ~/venvs/distill-train/bin/activate
python train/train_sft.py --run pilot-8b --data ~/distill-data/sft_pilot.jsonl --epochs 2 > logs/pilot_train.log 2>&1
python serve/merge_lora.py --adapter ~/distill-data/runs/pilot-8b/adapter --out ~/distill-data/models/pilot-8b >> logs/pilot_train.log 2>&1
deactivate
. ~/venvs/distill-quant/bin/activate
CUDA_VISIBLE_DEVICES= python serve/quantize.py --model ~/distill-data/models/pilot-8b --scheme fp8 >> logs/pilot_train.log 2>&1
echo PIPELINE_DONE >> logs/pilot_train.log
