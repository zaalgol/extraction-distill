"""Step 3: teacher labels from osint-agents' own LLMExtractor (role agent_heavy = qwen3.6:27b).

The extractor builds the production system prompt, user prompt and JSON schema; a recording
LocalLLM captures the exact chat messages and the teacher's raw JSON. A document is kept for
SFT only if the first answer validated (no retry turn, which would put a correction dialogue
in the training data) and every quote verifies against the text with osint's QuoteIndex.

Runs on Windows in osint-agents' environment. Resumable: every outcome goes to
labels_raw.jsonl, and documents already there are skipped.

    uv run --project C:\\code\\osint-agents python -m distill.label
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time

from distill.paths import data_dir, osint_home

os.environ.setdefault("OSINT_HOME", str(osint_home()))

import httpx  # noqa: E402
from osint.core.config import get_models_config  # noqa: E402
from osint.core.context import App  # noqa: E402
from osint.core.models import DocumentView  # noqa: E402
from osint.extraction.quotes import QuoteIndex  # noqa: E402
from osint.llm.local import LocalLLM  # noqa: E402
from osint.plugins.extractors.llm import LLMExtractor  # noqa: E402

ROLE = "agent_heavy"


class RecordingLLM(LocalLLM):
    """Keeps every (messages, raw output) pair of the current document."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[tuple[list[dict], str, bool]] = []

    async def _chat(self, r, messages, schema, role_name="", attempt=1):
        content, truncated = await super()._chat(r, messages, schema, role_name, attempt)
        self.calls.append(([dict(m) for m in messages], content, truncated))
        return content, truncated


def unverified_quotes(text: str, out: dict) -> list[str]:
    qi = QuoteIndex(text)
    quotes = [x.get("quote") for k in ("entities", "relations", "events") for x in out[k]]
    return [q for q in quotes if not q or qi.find(q) is None]


async def run(limit: int | None, context: int | None, workers: int) -> None:
    d = data_dir()
    docs = [json.loads(x) for x in (d / "docs.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    raw_p, sft_p = d / "labels_raw.jsonl", d / "sft.jsonl"
    done = set()
    if raw_p.exists():
        done = {json.loads(x)["id"] for x in raw_p.read_text(encoding="utf-8").splitlines() if x}
    # hashed order mixes the top-up languages in, so a run stopped early is still balanced
    todo = sorted((x for x in docs if x["id"] not in done),
                  key=lambda x: hashlib.sha256(x["id"].encode()).hexdigest())[:limit]
    print(f"{len(done)} labeled, {len(todo)} to go", flush=True)

    app = App()
    cfg = get_models_config().local
    if context:
        # Only the KV-cache size: a smaller one keeps more of the 27B on the GPU. Prompt,
        # schema and sampling stay the production ones.
        cfg.roles[ROLE].context = context
    app.settings.extraction.llm_role = ROLE
    # The writer (14B) may still be loaded; two models don't fit in 16 GB.
    async with httpx.AsyncClient() as c:
        for m in {r.model for r in cfg.roles.values()} - {cfg.roles[ROLE].model}:
            await c.post(cfg.base_url.removesuffix("/v1") + "/api/generate",
                         json={"model": m, "keep_alive": 0})
    ext = LLMExtractor()
    ext.role = ROLE
    queue: asyncio.Queue = asyncio.Queue()
    for doc in todo:
        queue.put_nowait(doc)
    t_start, n_done, kept = time.monotonic(), 0, 0

    async def label_one(doc: dict, llm: RecordingLLM, ctx) -> None:
        nonlocal n_done, kept
        view = DocumentView(id=0, artifact_id=0, text=doc["text"],
                            investigation_focus=doc["focus"])
        llm.calls.clear()
        t0 = time.monotonic()
        status, problems, out = "ok", [], None
        try:
            await ext.extract(view, ctx)
            out = json.loads(llm.calls[-1][1])
            if len(llm.calls) > 1:
                status = "retried"
            elif bad := unverified_quotes(doc["text"], out):
                status, problems = "bad_quotes", bad[:5]
        except Exception as e:  # noqa: BLE001 - a failed document is just not kept
            status, problems = "error", [f"{type(e).__name__}: {e}"[:300]]
        secs = round(time.monotonic() - t0, 1)
        rec = {"id": doc["id"], "lang": doc["lang"], "scenario": doc["scenario"],
               "status": status, "problems": problems, "seconds": secs,
               "attempts": len(llm.calls), "teacher": cfg.roles[ROLE].model, "output": out}
        with raw_p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if status == "ok":
            kept += 1
            sys_msg, user_msg = llm.calls[0][0][:2]
            answer = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
            sft = {"id": doc["id"], "lang": doc["lang"], "scenario": doc["scenario"],
                   "messages": [sys_msg, user_msg, {"role": "assistant", "content": answer}]}
            with sft_p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sft, ensure_ascii=False) + "\n")
        n_done += 1
        el = time.monotonic() - t_start
        print(f"{n_done}/{len(todo)} {doc['id']} {doc['lang']} {status} {secs}s  kept {kept}  "
              f"eta {el / n_done * (len(todo) - n_done) / 3600:.1f} h", flush=True)

    async def worker() -> None:
        # one LocalLLM per worker, so each records only its own document's calls
        llm = RecordingLLM(cfg, slots=1)
        ctx = app.context(None, with_db=False, llm=llm)
        try:
            while not queue.empty():
                await label_one(queue.get_nowait(), llm, ctx)
        finally:
            await llm.aclose()

    await asyncio.gather(*(worker() for _ in range(workers)))
    print("DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--context", type=int, default=None,
                    help="override the role's num_ctx (Ollama KV cache size)")
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent documents (Ollama's OLLAMA_NUM_PARALLEL is 2)")
    a = ap.parse_args()
    asyncio.run(run(a.limit, a.context, a.workers))


if __name__ == "__main__":
    main()
