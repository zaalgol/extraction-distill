#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server in the background and wait until it is healthy.
#   serve/serve.sh MODEL_PATH SERVED_NAME [extra vllm args...]
# Thinking is off by default (the extractor's Ollama path sends think=false too), and JSON
# schema guided decoding comes from the request's response_format. One GPU job at a time:
# stop it with `pkill -f "vllm serve"` before training or labeling.
set -euo pipefail
MODEL=$1; NAME=$2; shift 2
P=/mnt/c/code/extraction-distill
. "$HOME/venvs/distill-serve/bin/activate"
pkill -f "vllm serve" && sleep 5 || true
nohup vllm serve "$MODEL" --served-model-name "$NAME" --host 0.0.0.0 --port 8000 \
  --max-model-len 12288 --gpu-memory-utilization 0.88 --max-num-seqs 32 \
  --default-chat-template-kwargs '{"enable_thinking": false}' "$@" \
  > "$P/logs/serve-$NAME.log" 2>&1 &
echo "pid $!, log logs/serve-$NAME.log"
for _ in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { echo "ready"; exit 0; }
  kill -0 $! 2>/dev/null || { echo "vllm exited"; tail -30 "$P/logs/serve-$NAME.log"; exit 1; }
  sleep 5
done
echo "timeout"; exit 1
