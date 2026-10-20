#!/usr/bin/env bash
# AWQ W4A16 of the distilled 8B: calibration runs layer by layer on the GPU.
set -e
cd /mnt/c/code/extraction-distill
. ~/venvs/distill-quant/bin/activate
python serve/quantize.py --model ~/distill-data/models/qwen3-8b-distill --scheme awq
echo JOB_DONE
