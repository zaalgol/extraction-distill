"""Step 6: score a served model with osint-agents' own eval (`osint eval extraction`).

Runs osint's CLI against the vLLM server (OpenAI-compatible backend, role `agent`: same
prompt, schema, temperature 0.2 and 6000-token cap as production), then copies the saved
run JSON into results/eval/ so the numbers in the README are reproducible from this repo.

    uv run --project C:\\code\\osint-agents python -m distill.run_eval --model qwen3-8b-distill-fp8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from distill.paths import REPO, osint_home


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    a = ap.parse_args()
    home = osint_home()
    evals = home / "data" / "evals"
    before = set(evals.glob("extraction-*.json")) if evals.exists() else set()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(["uv", "run", "python", "-m", "osint", "eval", "extraction", "--role", "agent",
                    "--model", a.model, "--backend", "openai", "--base-url", a.base_url],
                   cwd=home, env=env, check=True)
    new = sorted(set(evals.glob("extraction-*.json")) - before)
    if not new:
        sys.exit(f"no new eval run under {evals}")
    dst = REPO / "results" / "eval" / new[-1].name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(new[-1], dst)
    t = json.loads(dst.read_text(encoding="utf-8"))["totals"]
    print(f"-> {dst}\n" + json.dumps({k: t[k] for k in ("entities", "relations", "events")}))


if __name__ == "__main__":
    main()
