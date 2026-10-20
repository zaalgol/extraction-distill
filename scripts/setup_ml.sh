#!/usr/bin/env bash
# Stage 2: Unsloth (training) and vLLM (serving) in separate venvs, so their pinned
# torch/transformers versions can't conflict. Each part runs a GPU smoke test. Re-runnable.
set -uo pipefail
P=/mnt/c/code/extraction-distill
LOG=$P/logs/setup_ml.log
exec > >(tee "$LOG") 2>&1
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "== [1/2] training venv: unsloth"
T="$HOME/venvs/distill-train"
. "$T/bin/activate"
time uv pip install unsloth --torch-backend=cu128 && uv pip list 2>/dev/null | grep -iE "^(torch|transformers|trl|peft|bitsandbytes|unsloth|unsloth.zoo|xformers|triton|datasets) " 
python "$P/scripts/smoke/train_smoke.py" || echo "TRAIN SMOKE FAILED"
deactivate

echo "== [2/2] serving venv: vllm"
S="$HOME/venvs/distill-serve"
# Recreated each run: installing into an existing venv keeps a stale torch build, and the
# vLLM wheel needs the CUDA 13 one (libcudart.so.13).
rm -rf "$S" && uv venv --python 3.12 "$S"
. "$S/bin/activate"
time uv pip install vllm --torch-backend=cu130 && uv pip list 2>/dev/null | grep -iE "^(torch|vllm|transformers|xgrammar|flashinfer.*) "
# Prebuilt FlashInfer kernels (incl. sm120): without them the sampler JIT-compiles and needs nvcc.
vllm download-kernels
python "$P/scripts/smoke/serve_smoke.py" || echo "SERVE SMOKE FAILED"
deactivate
echo "DONE"
