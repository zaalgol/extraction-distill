#!/usr/bin/env bash
# vLLM's OpenAI-compatible server on :8000, in the foreground (start it detached from Windows
# with scripts/wsl_job.ps1, see scripts/jobs/serve_*.sh; stop with `pkill -f "vllm serve"`).
#   serve/serve.sh MODEL_PATH SERVED_NAME [extra vllm args...]
# Thinking is off by default (osint's Ollama path sends think=false too); JSON-schema guided
# decoding comes from each request's response_format. One GPU job at a time.
set -euo pipefail
MODEL=$1; NAME=$2; shift 2
P=/mnt/c/code/extraction-distill
. "$HOME/venvs/distill-serve/bin/activate"
exec vllm serve "$MODEL" --served-model-name "$NAME" --host 0.0.0.0 --port 8000 \
  --max-model-len 12288 --gpu-memory-utilization 0.88 --max-num-seqs 32 \
  --default-chat-template-kwargs '{"enable_thinking": false}' "$@" \
  > "$P/logs/serve-$NAME.log" 2>&1
