"""Step 1: synthetic documents, written by the local 14B (Ollama) from sampled scenario specs.

Runs on Windows (Ollama is not reachable from WSL). Resumable: specs whose id is already in
the output are skipped. Documents that fail the leakage guard go to docs_rejected.jsonl.

    uv run --project C:\\code\\osint-agents python -m distill.gen_docs --n 1400
    # top-up for the languages the script filter thinned out most:
    uv run ... python -m distill.gen_docs --n 600 --seed 8 --start 10000 --langs ar fa he
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx

from distill.leakage import Guard
from distill.paths import data_dir
from distill.specs import sample, writer_prompt

WRITER = "huihui_ai/qwen3-abliterated:14b"
OLLAMA = "http://127.0.0.1:11434"


# Letters a document may use besides Latin (handles, domains and wallets are Latin anyway).
SCRIPTS = {"ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFEFF)],
           "fa": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFEFF)],
           "ru": [(0x0400, 0x04FF)], "he": [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)]}


def script_problems(text: str, lang: str) -> list[str]:
    """The 14B sometimes drifts into another script (CJK in a Hebrew text) or into English."""
    ranges = SCRIPTS.get(lang, [])
    native = foreign = latin = 0
    for c in text:
        if not c.isalpha():
            continue
        o = ord(c)
        if o < 0x250:
            latin += 1
        elif any(a <= o <= b for a, b in ranges):
            native += 1
        else:
            foreign += 1
    out = [f"foreign script ({foreign} letters)"] if foreign else []
    if ranges and native < 0.6 * (native + latin):
        out.append("mostly not in the target script")
    return out


def _clean(text: str) -> str:
    t = text.strip()
    if len(t) > 1 and t[0] in "\"'«“" and t[-1] in "\"'»”":
        t = t[1:-1].strip()
    return t


async def write_one(client: httpx.AsyncClient, prompt: str, seed: int) -> str:
    body = {"model": WRITER, "stream": False, "think": False, "keep_alive": "10m",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 1500,
                        "num_ctx": 4096, "seed": seed}}
    r = await client.post(f"{OLLAMA}/api/chat", json=body)
    r.raise_for_status()
    return _clean(r.json()["message"]["content"])


async def run(n: int, seed: int, concurrency: int, start: int = 0,
              langs: list[str] | None = None) -> None:
    out_dir = data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out, rej = out_dir / "docs.jsonl", out_dir / "docs_rejected.jsonl"
    done = set()
    for p in (out, rej):
        if p.exists():
            done |= {json.loads(x)["id"] for x in p.read_text(encoding="utf-8").splitlines() if x}
    todo = [s for s in sample(n, seed, start, langs) if s.id not in done]
    print(f"{len(done)} done, {len(todo)} to write", flush=True)
    guard = Guard.from_gold()
    sem = asyncio.Semaphore(concurrency)
    t0, count = time.monotonic(), 0

    async with httpx.AsyncClient(timeout=300) as client:
        async def job(spec):
            nonlocal count
            async with sem:
                try:
                    text = await write_one(client, writer_prompt(spec),
                                           int(spec.id.split("-")[1]))
                except (httpx.HTTPError, KeyError) as e:
                    print(f"{spec.id}: {type(e).__name__} {e}", flush=True)
                    return
            rec = {**spec.as_dict(), "text": text, "writer": WRITER}
            hits = guard.hits(text) + script_problems(text, spec.lang)
            if "[" in text or "guidance" in text.lower():
                hits.append("writer note leaked")
            if len(text) < 40:
                hits.append("too short")
            with (rej if hits else out).open("a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "rejected": hits} if hits else rec,
                                   ensure_ascii=False) + "\n")
            count += 1
            if count % 25 == 0:
                rate = (time.monotonic() - t0) / count
                print(f"{count}/{len(todo)}  {rate:.1f} s/doc  "
                      f"eta {rate * (len(todo) - count) / 60:.0f} min", flush=True)

        await asyncio.gather(*(job(s) for s in todo))
    print("DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--start", type=int, default=0,
                    help="first spec id (top-ups use a new id range and seed)")
    ap.add_argument("--langs", nargs="*", help="only these languages")
    a = ap.parse_args()
    asyncio.run(run(a.n, a.seed, a.concurrency, a.start, a.langs))


if __name__ == "__main__":
    main()
