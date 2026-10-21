"""Steps 4-8 unattended, once teacher labeling has finished.

A Windows-side driver (Ollama, osint's eval and the WSL jobs are all reachable from here),
started detached so it outlives any terminal:

    Start-Process -WindowStyle Hidden cmd -ArgumentList `
      '/c python scripts\\after_labeling.py [--no-wait] > logs\\after_labeling.log 2>&1'

--no-wait: labeling was stopped early, so train on sft.jsonl as it is.

Order: retry failed labels -> leakage test -> train/merge/FP8 of the 8B -> serve, osint
eval and serving benchmark for the 8B distilled and base FP8 -> AWQ and 4B (optional) and
their eval/benchmark -> results/RESULTS.md. Each
stage is skipped if its output already exists, so a rerun picks up where it stopped. A
failed model is logged and skipped; the others still run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OSINT = Path(r"C:\code\osint-agents")
WSL_DATA = Path(r"\\wsl.localhost\Ubuntu\home\pc\distill-data")
UV = ["uv", "run", "--project", str(OSINT), "python"]
ENV = {"PYTHONIOENCODING": "utf-8"}

# served name, model path in WSL, benchmark label, weight format
MODELS = [
    ("qwen3-8b-distill-fp8", "~/distill-data/models/qwen3-8b-distill-fp8",
     "Qwen3-8B distilled", "FP8 W8A8"),
    ("qwen3-8b-base-fp8", "~/distill-data/models/qwen3-8b-base-fp8",
     "Qwen3-8B, not fine-tuned", "FP8 W8A8"),
    ("qwen3-8b-distill-awq", "~/distill-data/models/qwen3-8b-distill-awq",
     "Qwen3-8B distilled", "AWQ W4A16"),
    ("qwen3-4b-distill", "~/distill-data/models/qwen3-4b-distill",
     "Qwen3-4B distilled", "BF16"),
]


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def run(cmd: list[str], **kw) -> bool:
    import os
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=kw.pop("cwd", REPO), env={**os.environ, **ENV}, **kw)
    if r.returncode:
        log(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r.returncode == 0


def wsl_job(name: str) -> bool:
    return run(["wsl.exe", "-e", "bash", "-c",
                f"bash /mnt/c/code/extraction-distill/scripts/jobs/{name}.sh "
                f"> /mnt/c/code/extraction-distill/logs/job-{name}.log 2>&1"])


def wsl_exists(path: str) -> bool:
    return subprocess.run(["wsl.exe", "-e", "bash", "-c", f"test -e {path}"]).returncode == 0


def unload_ollama() -> None:
    for m in ("qwen3.6:27b", "huihui_ai/qwen3-abliterated:14b"):
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", method="POST",
                                     data=json.dumps({"model": m, "keep_alive": 0}).encode())
        try:
            urllib.request.urlopen(req, timeout=60).read()
        except OSError:
            pass
    time.sleep(5)


def healthy() -> bool:
    try:
        return urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).status == 200
    except OSError:
        return False


def stop_vllm() -> None:
    subprocess.run(["wsl.exe", "-e", "bash", "-c", "pkill -f 'vllm serve'; sleep 10"])


def serve(name: str, path: str) -> subprocess.Popen | None:
    stop_vllm()
    p = subprocess.Popen(["wsl.exe", "-e", "bash", "/mnt/c/code/extraction-distill/serve/serve.sh",
                          path, name])
    for _ in range(120):
        if healthy():
            log(f"serving {name}")
            return p
        if p.poll() is not None:
            break
        time.sleep(5)
    log(f"vLLM did not come up for {name}; see logs/serve-{name}.log")
    stop_vllm()
    return None


def done_eval(name: str) -> bool:
    return any((REPO / "results" / "eval").glob(f"extraction-{name}-20*.json"))


def evaluate(models) -> None:
    for name, path, label, weights in models:
        if not wsl_exists(f"{path}/config.json"):
            log(f"skip {name}: no checkpoint")
            continue
        if done_eval(name) and (REPO / "results" / "bench" / f"{name}.json").exists():
            continue
        if serve(name, path) is None:
            continue
        if not done_eval(name):
            run([*UV, "-m", "distill.run_eval", "--model", name])
        run(["wsl.exe", "-e", "bash", "-c",
             ". ~/venvs/distill-serve/bin/activate && cd /mnt/c/code/extraction-distill && "
             f"python bench/bench_serving.py --model {name} --label '{label}' "
             f"--weights '{weights}'"])
        stop_vllm()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true",
                    help="labeling was stopped early: use sft.jsonl as it is (no DONE in "
                         "label.log, no retry of failed labels)")
    a = ap.parse_args()
    if not a.no_wait:
        label_log = REPO / "logs" / "label.log"
        while True:
            tail = label_log.read_text(encoding="utf-8", errors="replace")[-3000:]
            if tail.rstrip().endswith("DONE"):
                break
            if "Traceback" in tail:
                sys.exit("labeling crashed: see logs/label.log")
            time.sleep(300)
        log("labeling finished")
        retried = REPO / "logs" / "after_labeling.retried"
        if not retried.exists():
            run([*UV, "-m", "distill.label", "--retry-errors"])
            retried.touch()
    if not run([*UV, "-m", "pytest", "-q", "tests"]):
        sys.exit("leakage test failed: not training on this data")
    unload_ollama()

    # The required pair (distilled vs base 8B, FP8) first; AWQ and the 4B are extras whose
    # failure is logged and skipped.
    if not wsl_exists("~/distill-data/models/qwen3-8b-distill-fp8/config.json"):
        wsl_job("train_8b")
    evaluate(MODELS[:2])
    if not wsl_exists("~/distill-data/models/qwen3-8b-distill-awq/config.json"):
        wsl_job("quant_awq")
    if not wsl_exists("~/distill-data/models/qwen3-4b-distill/config.json"):
        wsl_job("train_4b")
    evaluate(MODELS[2:])

    run(["uv", "run", "--project", str(OSINT), "--with", "matplotlib", "python",
         "results/build_report.py"])
    log("ALL DONE")


if __name__ == "__main__":
    main()
