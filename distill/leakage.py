"""Leakage guard: training data must not contain the gold set's entity names or text.

Two checks against every training text:
  * names: no gold entity name or alias. Latin-script names match case-insensitively on word
    boundaries; other scripts match as plain substrings, because Arabic and Hebrew attach
    prefixes (و, ال, ב) to the word. That over-rejects a little (عدن inside معدن), which only
    costs a few documents.
  * 8-grams: no sequence of 8 word tokens shared with any gold text.

    python -m distill.leakage FILE.jsonl [--field text]   # exits 1 on any hit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from distill.paths import gold_path

N = 8
_WORD = re.compile(r"\w+", re.UNICODE)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()


def _is_latin(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return all(ord(c) < 0x250 for c in letters)


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    toks = _WORD.findall(_norm(text))
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class Guard:
    latin: re.Pattern | None
    other: list[str]
    grams: set[tuple[str, ...]]

    @classmethod
    def from_gold(cls, path: Path | None = None) -> Guard:
        gold = [json.loads(line) for line in (path or gold_path()).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        names = {_norm(n).strip() for g in gold for e in g["entities"]
                 for n in [e["name"], *e.get("aliases", [])]}
        names = {n for n in names if len(n) >= 3}
        latin = sorted((n for n in names if _is_latin(n)), key=len, reverse=True)
        pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in latin) + r")(?!\w)") \
            if latin else None
        grams = set().union(*(ngrams(g["text"]) for g in gold))
        return cls(pat, sorted(n for n in names if not _is_latin(n)), grams)

    def hits(self, text: str) -> list[str]:
        t = _norm(text)
        out = []
        if self.latin:
            out += [f"name:{m.group(0)}" for m in self.latin.finditer(t)]
        out += [f"name:{n}" for n in self.other if n in t]
        out += [f"8gram:{' '.join(g)}" for g in ngrams(text) & self.grams]
        return out

    def clean(self, text: str) -> bool:
        return not self.hits(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--field", default="text",
                    help="JSON field to check; 'messages' checks every chat message")
    a = ap.parse_args()
    guard = Guard.from_gold()
    bad = n = 0
    with a.file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            v = rec[a.field]
            text = "\n".join(m["content"] for m in v) if a.field == "messages" else v
            if hits := guard.hits(text):
                bad += 1
                print(f"{rec.get('id', n)}: {hits[:5]}")
    print(f"{n} records, {bad} with leakage")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
