# extraction-distill

Distilling an OSINT entity / relation / event extractor from a 27B teacher into a
QLoRA-tuned **Qwen3-8B**, served with **vLLM**, on one 16 GB consumer GPU.

The extractor belongs to osint-agents, a separate OSINT investigation project: it reads a
news-style text in any of 9 languages and returns schema-constrained JSON with entities,
typed relations (with a stance: asserted / reported / denied / speculative), events and a
verbatim quote for every item. The production model, `qwen3.6:27b`, is accurate but takes
~31 s per document; the fast fallback, a 14B, loses a third of relation F1. The goal here:
a small student that keeps most of the teacher's relation quality at the 14B's speed or
better.

<!-- RESULTS -->

## Method

1. **Synthetic documents** (`distill/specs.py`, `distill/gen_docs.py`). Scenario specs are
   sampled in code: language (en, ar, ru, fr, es, he, fa, tr, de), scenario (claims and
   attacks, arrests, finance with crypto wallets, online infrastructure, meetings,
   leadership, affiliation, threats, denials, speculation, mixed stance, ~15% off-topic
   near-misses such as football "attacks" or hostile takeovers), length, register and
   invented names and identifiers. The local `qwen3-abliterated:14b` writes the text from
   the spec. Generated documents are filtered for script drift (the 14B sometimes slips
   CJK or Cyrillic letters into Arabic or Hebrew) and copied writer notes. No Claude output
   is used anywhere in the training data.
2. **Leakage guard** (`distill/leakage.py`, `tests/test_leakage.py`). No gold-set entity
   name or alias, and no 8-gram of any gold text, may appear in a training document or
   training example. The guard runs at generation time and again as a test over the final
   files.
3. **Teacher labels** (`distill/label.py`). osint-agents' own `LLMExtractor` (role
   `agent_heavy` = `qwen3.6:27b`) labels every document, so the system prompt, user prompt
   and JSON schema are byte-for-byte the production ones. A recording LLM client saves the
   exact chat messages and the teacher's JSON. A document is kept only if the first answer
   validated against the schema and every quote verifies against the text with osint's
   `QuoteIndex`.
4. **QLoRA SFT** (`train/train_sft.py`). Unsloth, Qwen3-8B in 4-bit NF4, LoRA on all linear
   layers. The loss covers the assistant JSON only; the prompt is rendered exactly as vLLM
   renders it at serving time (Qwen3 chat template, thinking off), and the trainer asserts
   that. 10% of documents (hash split) are held out for validation loss.
5. **Merge and quantize** (`serve/merge_lora.py`, `serve/quantize.py`). The adapter is
   merged into the BF16 weights shard by shard (peak RAM ≈ one shard), then quantized with
   llm-compressor: FP8 W8A8 (dynamic activations, no calibration) and AWQ W4A16
   (calibrated on 256 training examples). Qwen3-8B in BF16 (16.4 GB) does not fit the GPU,
   so every served 8B model is quantized, including the untuned baseline.
6. **Serve** (`serve/serve.sh`). vLLM's OpenAI-compatible server; osint sends the JSON
   schema as `response_format`, so decoding is schema-guided (xgrammar).
7. **Evaluate** with osint-agents' unchanged scorer (`osint eval extraction`, given new
   `--backend/--base-url` options) on its 52-document hand-annotated gold set
   (`distill/run_eval.py`).
8. **Benchmark** (`bench/bench_serving.py`). The 52 gold-set extraction requests, replayed
   at concurrency 1, 4 and 16 with streaming, temperature 0 and the production 6000-token
   cap.

## Reproduce

Hardware: RTX 5070 Ti 16 GB, 32 GB RAM, Windows 11 + WSL Ubuntu. Ollama runs on Windows,
so the steps that use it run there in osint-agents' environment; training and serving run
in WSL.

```bash
# WSL: venvs (training: Unsloth; serving: vLLM; quantization: llm-compressor)
bash scripts/setup_wsl.sh && bash scripts/setup_ml.sh

# Windows, from this repo (Ollama with qwen3-abliterated:14b and qwen3.6:27b)
uv run --project C:\code\osint-agents python -m distill.gen_docs --n 1400
uv run --project C:\code\osint-agents python -m distill.gen_docs --n 600 --seed 8 --start 10000 --langs ar fa he
uv run --project C:\code\osint-agents python -m distill.label          # ~18 h
uv run --project C:\code\osint-agents python -m pytest                 # leakage guard
uv run --project C:\code\osint-agents python -m distill.export_prompts # benchmark workload

# WSL: train, merge, quantize, serve (unload Ollama's models first: one GPU job at a time)
python train/train_sft.py --run qwen3-8b
python serve/merge_lora.py --adapter ~/distill-data/runs/qwen3-8b/adapter --out ~/distill-data/models/qwen3-8b-distill
python serve/quantize.py --model ~/distill-data/models/qwen3-8b-distill --scheme fp8
serve/serve.sh ~/distill-data/models/qwen3-8b-distill-fp8 qwen3-8b-distill-fp8
python bench/bench_serving.py --model qwen3-8b-distill-fp8 --label "Qwen3-8B distilled" --weights "FP8 W8A8"

# Windows: score with osint's eval, then rebuild the tables
uv run --project C:\code\osint-agents python -m distill.run_eval --model qwen3-8b-distill-fp8
python results/build_report.py
```

<!-- LIMITATIONS -->
