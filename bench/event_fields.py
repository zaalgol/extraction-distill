"""Which event fields a served model fills, with and without guided decoding.

Diagnoses the key-order problem described in distill/schema_order.py.

    python bench/event_fields.py --model qwen3-8b-distill-fp8 [--n 12]
"""

from __future__ import annotations

import argparse
import collections
import json
import urllib.request
from pathlib import Path

PROMPTS = Path.home() / "distill-data" / "bench_prompts.jsonl"


def ask(model: str, r: dict, guided: bool) -> dict:
    body = {"model": model, "messages": r["messages"], "temperature": 0, "max_tokens": 6000}
    if guided:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": r["schema_name"], "schema": r["schema"], "strict": True}}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"][
        "content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"events": [], "_invalid": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    rows = [json.loads(x) for x in PROMPTS.read_text(encoding="utf-8").splitlines()][: a.n]
    for guided in (True, False):
        orders, dated, total, invalid = collections.Counter(), 0, 0, 0
        for r in rows:
            out = ask(a.model, r, guided)
            invalid += out.get("_invalid", False)
            for e in out.get("events", []):
                orders[tuple(e)] += 1
                total += 1
                dated += bool(e.get("time"))
        print(f"guided={guided}: {total} events, {dated} with a date, {invalid} invalid JSON")
        for o, n in orders.most_common(3):
            print(f"  {n:3d}  {o}")


if __name__ == "__main__":
    main()
