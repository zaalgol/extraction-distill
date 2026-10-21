# Status: 2026-10-21 13:45

**Done.** All 8 steps finished. The goal is met: relation F1 is well above 0.55 and the
student is faster than the 14B. Results are in the README and `results/RESULTS.md`, and
everything is pushed to https://github.com/zaalgol/extraction-distill.

| Model | Relations F1 (P / R) | Events F1 | Entities F1 | s / doc |
|---|---|---|---|---|
| Teacher `qwen3.6:27b` | 0.82 (0.73 / 0.94) | 0.85 | 0.95 | 31.3 |
| 14B (current fast model) | 0.55 (0.44 / 0.75) | 0.78 | 0.88 | 5.3 |
| Qwen3-8B, not fine-tuned (FP8) | 0.54 (0.46 / 0.66) | 0.73 | 0.92 | 7.9 |
| **Qwen3-8B distilled (FP8)** | **0.76 (0.64 / 0.94)** | 0.79 | 0.92 | **3.0** |
| **Qwen3-4B distilled (BF16)** | **0.74 (0.66 / 0.84)** | 0.76 | 0.89 | **2.5** |

The 8B student serves 1,040 output tokens/s, or 4.2 documents/s, at concurrency 16.

## What happened since the last status
- **Labeling stopped early** after 790 of 1,280 documents (the machine ran out of RAM: the
  WSL VM plus Ollama's 27B). By your decision, the 753 labels kept became the final training
  set: 665 train, 88 validation. The leakage test passes on them.
- **Driver:** `scripts/after_labeling.py` got two options:
  - `--no-wait`: start without `DONE` in `label.log`;
  - `--skip quant_awq`: skip a failed optional job.

  It now evaluates the two required models before the optional ones.
- **Bug 1, serving:** vLLM didn't start because `wsl.exe -e` passes `~` unexpanded.
  `serve/serve.sh` now expands it.
- **Bug 2, events F1 0.26:** the teacher, through Ollama, wrote event keys as
  `type, description, quote, time, …`, but the schema order is
  `type, description, time, …, quote`. vLLM's guided decoding emits keys in schema order,
  so after `quote` the student could no longer write `time`, and every event lost its date.
  - Confirmed with `bench/event_fields.py`: 0 of 14 events had a date with guided decoding,
    9 of 12 without.
  - Fix: `distill/schema_order.py` (same content, schema key order) and
    `tests/test_schema_order.py`. Both students were retrained, and events F1 went from 0.26
    to 0.79.
  - The first run is archived in `results/diagnostics/teacher-key-order/`.
- **AWQ:** calibration ran out of GPU memory. It's optional, so it was skipped, not debugged,
  as you asked.
- **osint-agents:** regenerated `evals/extraction/RESULTS.md` with `osint eval report`, so it
  now includes the 8B base, 8B distilled and 4B distilled rows. The pilot run and the first
  8B run were moved to `data/evals/superseded/` so they don't show up as models.

## Left for you
- **osint-agents is uncommitted.** That covers my changes (`--backend/--base-url`,
  `max_tokens` in the openai backend, the regenerated `RESULTS.md`) and your own earlier
  edits. You decide what to commit there.
- **Delete the stray `C:\c` folder.** It holds copies of two eval files that are also in
  `results/eval/`.
- **Optional follow-ups:**
  - label the remaining 490 documents, more Hebrew especially;
  - AWQ with fewer or shorter calibration samples;
  - a second seed to estimate variance.
