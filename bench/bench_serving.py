"""Step 7: serving benchmark against a running vLLM server (serve/serve.sh).

Replays the production extraction requests for the 52 gold documents (JSON-schema guided,
temperature 0, the same 6000-token cap as osint) at several concurrency levels and records
end-to-end latency p50/p95, time to first token, output tokens/s (aggregate and per
request) and GPU memory. Each level runs every prompt `--rounds` times after one warm-up
request, so all levels see the same mix of short and long documents.

    python bench/bench_serving.py --model qwen3-8b-distill-fp8 --label "distilled FP8"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
PROMPTS = Path.home() / "distill-data" / "bench_prompts.jsonl"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def gpu_mem_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


async def one(client: httpx.AsyncClient, url: str, model: str, p: dict) -> dict:
    body = {"model": model, "messages": p["messages"], "temperature": 0, "max_tokens": 6000,
            "stream": True, "stream_options": {"include_usage": True},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": p["schema_name"], "schema": p["schema"]}}}
    t0 = time.perf_counter()
    ttft, usage, text = None, {}, []
    async with client.stream("POST", f"{url}/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                if c["delta"].get("content"):
                    ttft = ttft or time.perf_counter() - t0
                    text.append(c["delta"]["content"])
    total = time.perf_counter() - t0
    out_tok = usage.get("completion_tokens", 0)
    try:
        json.loads("".join(text))
        valid = True
    except json.JSONDecodeError:
        valid = False
    return {"id": p["id"], "latency": total, "ttft": ttft or total, "out_tokens": out_tok,
            "in_tokens": usage.get("prompt_tokens", 0), "valid_json": valid,
            "decode_tps": out_tok / max(total - (ttft or 0), 1e-6)}


async def level(url: str, model: str, prompts: list[dict], conc: int, rounds: int) -> dict:
    sem = asyncio.Semaphore(conc)
    jobs = [p for _ in range(rounds) for p in prompts]
    peak = [gpu_mem_mib()]

    async with httpx.AsyncClient(timeout=900) as client:
        async def run(p):
            async with sem:
                r = await one(client, url, model, p)
                peak[0] = max(peak[0], gpu_mem_mib())
                return r

        t0 = time.perf_counter()
        res = await asyncio.gather(*(run(p) for p in jobs))
        wall = time.perf_counter() - t0
    lat = [r["latency"] for r in res]
    return {"concurrency": conc, "requests": len(res), "wall_s": round(wall, 1),
            "latency_p50_s": round(pct(lat, 0.5), 2), "latency_p95_s": round(pct(lat, 0.95), 2),
            "ttft_p50_s": round(pct([r["ttft"] for r in res], 0.5), 3),
            "output_tok_per_s": round(sum(r["out_tokens"] for r in res) / wall, 1),
            "per_request_decode_tok_per_s": round(
                statistics.median(r["decode_tps"] for r in res), 1),
            "mean_output_tokens": round(statistics.mean(r["out_tokens"] for r in res)),
            "mean_input_tokens": round(statistics.mean(r["in_tokens"] for r in res)),
            "requests_per_s": round(len(res) / wall, 2),
            "valid_json": sum(r["valid_json"] for r in res), "gpu_mem_peak_mib": peak[0]}


async def main_async(a) -> None:
    prompts = [json.loads(x) for x in PROMPTS.read_text(encoding="utf-8").splitlines() if x]
    async with httpx.AsyncClient(timeout=900) as client:   # warm-up: compile, schema cache
        await one(client, a.url, a.model, prompts[0])
    rows = []
    for c in a.concurrency:
        rounds = max(1, a.rounds if c == 1 else a.rounds * c // 4)
        r = await level(a.url, a.model, prompts, c, rounds)
        print(json.dumps(r), flush=True)
        rows.append(r)
    out = REPO / "results" / "bench" / f"{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "label": a.label, "weights": a.weights, "prompts": len(prompts),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "levels": rows}
    out.write_text(json.dumps(meta, indent=1))
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--label", default="")
    ap.add_argument("--weights", default="", help="weight format, e.g. FP8 W8A8, AWQ W4A16")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--rounds", type=int, default=1)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
