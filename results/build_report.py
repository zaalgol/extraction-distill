"""Build results/RESULTS.md (and the loss-curve plot) from the JSON files under results/.

Every number in the README comes from here: eval runs (results/eval, copied from osint's
`osint eval extraction`), serving benchmarks (results/bench) and training logs
(results/train). Runs on Windows or WSL; the plot needs matplotlib.

    python results/build_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from distill.paths import gold_path  # noqa: E402

# eval file prefix -> (label, serving stack). Order = table order.
MODELS = [
    ("qwen3.6-27b", "`qwen3.6:27b` (teacher)", "Ollama, Q4_K_M"),
    ("huihui_ai_qwen3-abliterated-14b", "`qwen3-abliterated:14b` (current fast model)",
     "Ollama, Q4_K_M"),
    ("qwen3-8b-base-fp8", "Qwen3-8B, not fine-tuned", "vLLM, FP8"),
    ("qwen3-8b-distill-fp8", "**Qwen3-8B distilled**", "vLLM, FP8"),
    ("qwen3-8b-distill-awq", "**Qwen3-8B distilled**", "vLLM, AWQ 4-bit"),
    ("qwen3-4b-distill", "**Qwen3-4B distilled**", "vLLM, BF16"),
]
STANCE = {"denial", "speculative"}


def latest(prefix: str, sha: str) -> dict | None:
    runs = []
    for p in (HERE / "eval").glob(f"extraction-{prefix}-20*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("gold_sha256") == sha:
            runs.append(d)
    return max(runs, key=lambda d: d["created_at"]) if runs else None


def f(prf: dict) -> str:
    return f"{prf['f1']:.2f} ({prf['precision']:.2f} / {prf['recall']:.2f})"


def quality(gold: list[dict], sha: str) -> list[str]:
    stance = [g["id"] for g in gold if STANCE & set(g.get("tags", []))]
    off = [g["id"] for g in gold if not g["relevant"]]
    rows = ["| Model | Serving | Entities F1 (P / R) | Relations F1 (P / R) | Events F1 (P / R) "
            "| Quotes verified | False relations from denials / speculation "
            "| Off-topic rejected | s / doc |",
            "|---|---|---|---|---|---|---|---|---|"]
    for prefix, label, stack in MODELS:
        d = latest(prefix, sha)
        if not d:
            continue
        t, docs = d["totals"], {x["id"]: x for x in d["docs"]}
        false_rel = sum(docs[i]["relations"]["fp"] for i in stance if i in docs)
        rej = sum(1 for i in off if i in docs and docs[i]["relevant_ok"])
        rows.append(f"| {label} | {stack} | {f(t['entities'])} | {f(t['relations'])} | "
                    f"{f(t['events'])} | {t['quote_verification_rate']:.0%} | {false_rel} | "
                    f"{rej} / {len(off)} | {t['seconds_per_doc']} |")
    return rows


def by_language(gold: list[dict], sha: str) -> list[str]:
    langs = sorted({g["lang"] for g in gold}, key=lambda x: (x != "en", x))
    lang_of = {g["id"]: g["lang"] for g in gold}
    rows = ["| Model | " + " | ".join(langs) + " |", "|---|" + "---|" * len(langs)]
    for prefix, label, stack in MODELS:
        d = latest(prefix, sha)
        if not d:
            continue
        cells = []
        for lang in langs:
            tp = fp = fn = 0
            for x in d["docs"]:
                if lang_of.get(x["id"]) == lang:
                    r = x["relations"]
                    tp, fp, fn = tp + r["tp"], fp + r["fp"], fn + r["fn"]
            p = tp / (tp + fp) if tp + fp else 1.0
            r_ = tp / (tp + fn) if tp + fn else 1.0
            cells.append(f"{2 * p * r_ / (p + r_):.2f}" if p + r_ else "0.00")
        rows.append(f"| {label} ({stack}) | " + " | ".join(cells) + " |")
    return rows


def serving() -> list[str]:
    rows = ["| Model | Weights | Concurrency | Output tok/s (total) | Decode tok/s per request "
            "| Latency p50 / p95 (s) | TTFT p50 (s) | Docs / s | Peak GPU memory (GiB) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted((HERE / "bench").glob("*.json")):
        b = json.loads(p.read_text())
        for lv in b["levels"]:
            rows.append(f"| {b['label']} | {b['weights']} | {lv['concurrency']} | "
                        f"{lv['output_tok_per_s']:.0f} | {lv['per_request_decode_tok_per_s']:.0f} | "
                        f"{lv['latency_p50_s']:.2f} / {lv['latency_p95_s']:.2f} | "
                        f"{lv['ttft_p50_s']:.2f} | {lv['requests_per_s']:.2f} | "
                        f"{lv['gpu_mem_peak_mib'] / 1024:.1f} |")
    return rows


def training() -> list[str]:
    out = []
    for cfg_p in sorted((HERE / "train").glob("*/config.json")):
        c = json.loads(cfg_p.read_text())
        run = cfg_p.parent.name
        out += [f"**{run}**: base `{c['model']}`, {c['quantization']}, LoRA r={c['rank']} "
                f"alpha={c['alpha']} on {c['lora_target_modules']}, lr {c['lr']} "
                f"({c['scheduler']}), {c['epochs']} epochs, effective batch "
                f"{c['batch'] * c['grad_accum']}, {c['optimizer']}, {c['precision']}. "
                f"{c['train_docs']} training / {c['val_docs']} validation documents "
                f"({c['dropped_too_long']} dropped as longer than {c['max_len']} tokens). "
                f"{c['steps']} steps in {c['train_minutes']} min, peak VRAM "
                f"{c['peak_vram_gb']} GB, final validation loss {c['final_eval_loss']:.4f}.", ""]
        plot(cfg_p.parent)
        if (cfg_p.parent / "loss.png").exists():
            out += [f"![{run} loss curves](train/{run}/loss.png)", ""]
    return out


def plot(run_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h = json.loads((run_dir / "log_history.json").read_text())
    tr = [(x["step"], x["loss"]) for x in h if "loss" in x]
    ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=120)
    ax.plot(*zip(*tr, strict=True), label="train", color="#4C72B0", lw=1.2)
    if ev:
        ax.plot(*zip(*ev, strict=True), label="validation", color="#DD8452", marker="o", lw=1.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (assistant tokens)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(run_dir.name)
    fig.tight_layout()
    fig.savefig(run_dir / "loss.png")
    plt.close(fig)


def main() -> None:
    gold = [json.loads(x) for x in gold_path().read_text(encoding="utf-8").splitlines() if x]
    import hashlib
    sha = hashlib.sha256(json.dumps(gold, ensure_ascii=False, sort_keys=True)
                         .encode("utf-8")).hexdigest()
    lines = ["# Results", "",
             f"Gold set: osint-agents `evals/extraction/gold.jsonl`, {len(gold)} documents, "
             f"version `{sha[:12]}`. Generated by `results/build_report.py`.", "",
             "## Extraction quality", "", *quality(gold, sha), "",
             "## Relation F1 by language", "", *by_language(gold, sha), "",
             "## Serving (vLLM, RTX 5070 Ti 16 GB)", "", *serving(), "",
             "## Training", "", *training()]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
