# Status: 2026-10-21 01:27

Labeling (step 3) is running, about 16.5 h left. When it finishes, a driver
(`scripts/after_labeling.py`) runs steps 4–8 by itself. Nothing needs your input until
then.

| Step | State |
|---|---|
| 0. Environment | Done |
| 1. Synthetic documents | Done: 1,280 documents |
| 2. Leakage guard | Done: test passes on the documents |
| 3. Teacher labels | **Running**: 117 of 1,280 documents (17 of the 1,180 left when it resumed) |
| 4–8. Train, serve, eval, benchmark, report | Code written and checked in a pilot run; will start automatically |
| README | Draft written (method, how to reproduce); results and limitations not yet filled in |

## What I did

### Environment
- **Training venv (Unsloth):** the smoke test passes once `build-essential` was installed.
- **Serving venv (vLLM 0.32, torch cu130):** rebuilt from scratch. It needed:
  - prebuilt FlashInfer kernels (`vllm download-kernels`);
  - a `__main__` guard in scripts that start vLLM.
- **Quantization venv:** added a third one, `~/venvs/distill-quant`, with llm-compressor.
- **Ollama:** WSL can't reach Ollama, which runs on Windows. So the Ollama steps (1–3) run on
  Windows, and training and serving run in WSL. Windows reaches vLLM in WSL at
  `localhost:8000`.
- **`%UserProfile%\.wslconfig`:**
  - `memory=26GB`, `swap=16GB`: merging and quantizing load the 16.4 GB BF16 8B model into
    RAM.
  - `autoMemoryReclaim=gradual`.
- **Detached WSL jobs:** run through `scripts/wsl_job.ps1` and `scripts/jobs/*.sh`. A job
  started with `nohup` from a short `wsl.exe` call can die when WSL ends the session.
- **git:** repo initialized; all work is committed.

### Step 1: synthetic documents (`distill/specs.py`, `distill/gen_docs.py`)
- **Scenario specs** are sampled in code:
  - 9 languages;
  - 14 scenario types (claims, arrests, finance with wallets, infrastructure, meetings,
    denials, speculation, mixed stance, ~15% off-topic near-misses, …);
  - invented names and identifiers.
- **Writer:** `qwen3-abliterated:14b` on Ollama, at ~1.2 s per document.
- **Filters** reject leaked writer notes and text in the wrong script (the 14B sometimes puts
  CJK into Arabic, or Arabic letters into Hebrew).
- **Volume:**
  - 1,400 specs, then a 600-spec top-up for ar/fa/he → 1,280 kept, 720 rejected.
  - By language: en 380, fa 161, ar 155, es 118, de 115, fr 114, ru 97, tr 95, he 45.
  - Hebrew stays thin.
- **Output:** `~/distill-data/docs.jsonl` (WSL).

### Step 2: leakage guard (`distill/leakage.py`, `tests/test_leakage.py`)
- A training document may not contain:
  - any gold entity name or alias (whole words for Latin script, substrings for other
    scripts);
  - any 8-gram from a gold text.
- It runs during generation and again as a test over `docs.jsonl` and `sft.jsonl`.

### Step 3: teacher labels (`distill/label.py`)
- **Teacher:** osint-agents' own `LLMExtractor`, role `agent_heavy` (`qwen3.6:27b`). The
  prompt and schema are exactly the production ones.
- **Recording:** a recording LLM client saves the exact chat messages and the teacher's JSON.
- **A document is kept only if:**
  - the first answer validated (no retry);
  - every quote is found in the text (`QuoteIndex`).
- **Speed:** ~50 s per document. The 27B is 17.5 GB, so ~3.7 GB of it runs from system RAM.
  - Two parallel workers didn't help.
  - A smaller context didn't help.
- **Labels so far:**
  - 96% kept.
  - Denial and speculation documents get `denied`/`speculative` stances.
  - 16 of 18 off-topic documents are marked irrelevant.
- **Resumable.** The order is hashed, so stopping early still gives a balanced mix.
- **Retries:** `--retry-errors` re-labels documents whose call failed. The Ollama runner
  crashed once when it reloaded right after vLLM.

### Steps 4–8: code
| File | Does |
|---|---|
| `train/train_sft.py` | QLoRA with Unsloth (4-bit base, LoRA r=32 on all linear layers, 3 epochs, lr 2e-4 cosine). Loss on the assistant JSON only. The prompt is rendered exactly as vLLM renders it (asserted). 10% hash split for validation loss. |
| `serve/merge_lora.py` | Merges LoRA into the BF16 base one shard at a time (low RAM) |
| `serve/quantize.py` | FP8 W8A8 (no calibration) and AWQ W4A16 (calibrated on training examples) with llm-compressor |
| `serve/serve.sh` | vLLM OpenAI server, thinking off, JSON-schema guided decoding |
| `distill/run_eval.py` | Runs osint's unchanged `osint eval extraction` against vLLM and copies the run JSON into `results/eval/` |
| `bench/bench_serving.py` | Replays the 52 gold-set requests at concurrency 1/4/16: latency p50/p95, TTFT, tokens/s, GPU memory |
| `results/build_report.py` | Builds `results/RESULTS.md` and the loss curve from `results/` |
| `scripts/after_labeling.py` | The driver for steps 4–8 (see below) |

**osint-agents changes (uncommitted there, alongside your own uncommitted edits):**
- `osint eval extraction --backend/--base-url`;
- the OpenAI-compatible backend now sends `max_tokens` (the production 6000 cap).

Two osint tests fail with and without my changes (a filename with `:` on Windows, and a
tracing test).

### Pilot (labeling paused for ~1 h)
- **Setup:** the whole path ran on the first 97 labels: train → merge → FP8 → vLLM → osint
  eval → benchmark.
- **Bug found and fixed:** transformers 5 dropped `warmup_ratio` and `group_by_length`.
- **Pilot results (87 training docs, 12 steps):**
  - relations F1 0.65 (the 14B scores 0.55);
  - 3.2 s per document;
  - ~1,050 output tokens/s at concurrency 16;
  - 100% valid JSON.
  - Events F1 is 0.16 because event dates are mostly empty. The teacher's labels date about
    half their events, so this should be undertraining, not a pipeline bug. Untested.
- Kept in `results/pilot/`; these numbers won't be reported.
- **Done ahead of time:** the base Qwen3-8B FP8 checkpoint (`~/distill-data/models/qwen3-8b-base-fp8`).

## What happens next (automatic)
`scripts/after_labeling.py` runs as its own Windows process and logs to
`logs/after_labeling.log`. Once `logs/label.log` ends with `DONE`, it will:
1. Retry the failed labels, then run the leakage test. It stops if the test fails.
2. Unload Ollama. Train the 8B (~50 min), merge it, and make the FP8 version. Then AWQ, as a
   separate job (the only step the pilot didn't cover). Then train and merge the 4B.
3. For each of the 8B distilled FP8, 8B base FP8, 8B distilled AWQ and 4B distilled BF16:
   serve it, run osint's eval, and run the benchmark.
4. Rebuild `results/RESULTS.md`, then write `ALL DONE` to the log.

A stage whose output already exists is skipped, so rerunning the driver picks up where it
stopped.

## Remaining to finish (checklist)

Automatic (the driver):
- [ ] Labeling finishes: 1,280 documents (~16.5 h from 01:27).
- [ ] Retry the failed labels; the leakage test passes on `sft.jsonl`.
- [ ] Step 4: train the Qwen3-8B student; loss curves and settings saved to
      `results/train/qwen3-8b/`.
- [ ] Step 5: merge it, then make the FP8 and AWQ versions.
- [ ] Train and merge the Qwen3-4B student (optional extra).
- [ ] Step 6: osint eval for 8B distilled FP8, 8B base FP8, 8B distilled AWQ, 4B distilled.
- [ ] Step 7: serving benchmark for the same four models.
- [ ] Build `results/RESULTS.md`.

Manual (me, after `ALL DONE`):
- [ ] Check the results:
  - Relation F1 must be well above 0.55 and speed at least the 14B's (CLAUDE.md goal).
  - Events F1 should have recovered from the pilot's 0.16.
  - If not, find the cause before reporting. Likely knobs: more epochs or data, or
    rebalancing toward the failing cases.
- [ ] Rerun anything that failed (for example, if AWQ fails, fix it and rerun the driver; it
      skips finished stages).
- [ ] Step 6 (rest): add the student rows to osint-agents' `evals/extraction/RESULTS.md` with
      `osint eval report`.
- [ ] Step 8: finish the README:
  - results tables from `results/RESULTS.md` and the loss curve;
  - limitations: 52-document gold set, synthetic training data written by a 14B, the
    student inherits the teacher's biases, thin Hebrew, Ollama vs vLLM speed comparison;
  - a one-line CV summary.
- [ ] Update the Status section in `CLAUDE.md`, commit `results/` and the README.
- [ ] Hand-off: osint-agents changes (`--backend/--base-url`, `max_tokens`) are still
      uncommitted in that repo, next to your own edits; you decide whether to commit them.

## For you
- Delete the stray `C:\c` folder. I created it by mistake with a wrong path, and a safety
  check blocked me from deleting it. It only holds copies of two eval files that are also in
  `results/eval/`.
- Watching: `logs\label.log` (labeling) and `logs\after_labeling.log` (everything after).
  Say "continue" after `ALL DONE`, or if anything stops.
- Memory: the earlier low-RAM event came from running the FP8 quantization in WSL while
  Ollama held ~10 GB for the 27B. The driver now runs these steps one at a time.
