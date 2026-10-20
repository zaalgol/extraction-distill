# extraction-distill

Distill the osint-agents entity/relation/event extractor (teacher: `qwen3.6:27b` via Ollama)
into a QLoRA-tuned small model served with vLLM, and measure quality on the osint-agents gold
set and speed. A public portfolio project: the results go into the README and a CV line, so
every number must be reproducible from this repo.

## Baseline (osint-agents `evals/extraction/RESULTS.md`, 52-doc gold set `323524d796ec`)

| Model | Entities F1 | Relations F1 (P / R) | Events F1 | s/doc |
|---|---|---|---|---|
| `qwen3.6:27b` (teacher) | 0.95 | 0.82 (0.73 / 0.94) | 0.85 | 31.3 |
| `huihui_ai/qwen3-abliterated:14b` (current fast model) | 0.88 | 0.55 (0.44 / 0.75) | 0.78 | 5.3 |

Goal: a student (Qwen3-8B, maybe also Qwen3-4B) with relation F1 well above 0.55, at least as
fast as the 14B. Relation precision is the main weakness to fix.

## Environment (verified 2026-10-20)

- Windows 11, RTX 5070 Ti 16 GB (`sm_120`, Blackwell), driver 610.88, 32 GB RAM.
- WSL Ubuntu 26.04: 20 CPUs, 15 GB RAM (WSL default; raise `memory=` in
  `%UserProfile%\.wslconfig` if training needs it), 950 GB free.
- torch 2.11.0+cu128 works on the GPU (bf16 matmul ~98 TFLOPS).
- venvs live in WSL, separate on purpose (Unsloth and vLLM pin different torch/transformers):
  - `~/venvs/distill-train`: torch, Unsloth
  - `~/venvs/distill-serve`: vLLM
- `scripts/setup_wsl.sh` and `scripts/setup_ml.sh` build them; logs go to `logs/`.
- Ollama 0.40.4 runs on **Windows** and hosts the teacher and the 14B. Checked 2026-10-20: WSL
  can't reach it (NAT networking; neither `localhost` nor the host IP 172.29.80.1 answers).
  Decision: run the Ollama steps (1–3) **on Windows** with osint-agents' uv env; WSL only for
  training and serving. The eval (Windows) reaches vLLM in WSL through WSL's default localhost
  forwarding (verify at step 5).
- **One GPU job at a time.** Ollama labeling, training and vLLM can't share 16 GB. Unload
  Ollama models (`keep_alive: 0`) before training or serving.
- osint-agents is at `C:\code\osint-agents` (`/mnt/c/code/osint-agents`), branch
  `eval-and-observability`. On Windows, Application Control blocks uv's .exe launchers:
  always `uv run python -m osint ...`.

## Pipeline

1. **Synthetic documents** (`~/distill-data/docs.jsonl`): ~1,000 fictional, multilingual
   news-style texts in the gold set's 9 languages (en, ar, ru, fr, es, he, fa, tr, de), with
   the gold set's mix: claims and attacks, arrests, finance and wallets, online
   infrastructure, meetings, denials, speculation, and ~15% off-topic near-misses. Written by
   the local 14B from sampled scenario specs (language, event types, stance, invented names).
   **Not by Claude**: Anthropic's terms restrict using Claude output to train models.
2. **Leakage guard**: no gold entity name and no 8-gram from a gold text may appear in the
   training data. A script and a test enforce it.
3. **Teacher labels**: run osint-agents' own `LLMExtractor` with role `agent_heavy`, so the
   prompt and output schema are exactly the production ones. Keep a document only if the JSON
   validates and every quote verifies (osint's `QuoteIndex`). Save the exact chat messages and
   the assistant JSON to `~/distill-data/sft.jsonl`. ~30 s/doc: an overnight run.
4. **QLoRA SFT** with Unsloth on Qwen3-8B (4-bit), loss on assistant tokens only, 10% held out
   for validation loss. Log hyperparameters and curves.
5. **Serve** with vLLM (OpenAI-compatible; merged LoRA or `--enable-lora`) with JSON-schema
   guided decoding.
6. **Evaluate** with osint-agents' eval against the served student through its
   OpenAI-compatible backend (add `--backend/--base-url` options to `osint eval extraction` if
   needed), and add the student's rows to osint's RESULTS.md.
7. **Benchmark serving**: base vs fine-tuned vs quantized (AWQ or similar); tokens/s and
   p50/p95 latency at concurrency 1, 4 and 16, plus VRAM. Qwen3-8B in BF16 does not fit in
   16 GB.
8. **README**: results table, method, and limitations (small gold set, synthetic training
   data, teacher bias).

## Conventions

- Python 3.12, ruff, pytest. Comments explain why, not what.
- Code lives here (`C:\code\extraction-distill`). Large data, checkpoints and model caches go
  in the WSL filesystem (`~/distill-data`, `~/.cache/huggingface`), not under `/mnt/c`
  (slow I/O). Commit only small artifacts (configs, results, small samples).
- Long GPU jobs run with `nohup` and write to `logs/`, so they survive a closed terminal.

## Status

- Done: environment check, both venvs built by `scripts/setup_wsl.sh` + `scripts/setup_ml.sh`.
  Smoke tests pass (2026-10-20):
  - Unsloth (`distill-train`, torch 2.11 cu128): 5 QLoRA steps, 1.1 GB peak VRAM.
  - vLLM 0.32.0 (`distill-serve`, torch 2.13 cu130): Qwen3-0.6B generates. Needed: a fresh
    venv (in-place install kept the cu128 torch), `vllm download-kernels` (FlashInfer sm120
    kernels; else it JIT-compiles and wants nvcc), and a `__main__` guard in scripts that
    create `LLM(...)` (WSL forces multiprocessing `spawn`).
- git initialized.
- Next: step 1 (synthetic documents, on Windows).
