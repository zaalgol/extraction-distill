"""Where data lives. Steps 1-3 run on Windows (Ollama is only reachable there) but write into
the WSL filesystem, where training reads it without /mnt/c's slow I/O."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    if env := os.environ.get("DISTILL_DATA"):
        return Path(env)
    if sys.platform == "win32":
        return Path(r"\\wsl.localhost\Ubuntu\home\pc\distill-data")
    return Path.home() / "distill-data"


def osint_home() -> Path:
    if env := os.environ.get("OSINT_HOME"):
        return Path(env)
    return Path(r"C:\code\osint-agents") if sys.platform == "win32" else Path(
        "/mnt/c/code/osint-agents")


def gold_path() -> Path:
    return osint_home() / "evals" / "extraction" / "gold.jsonl"
