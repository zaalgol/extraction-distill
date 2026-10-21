"""Reorder the teacher's JSON keys into the output schema's property order.

vLLM's guided decoding (xgrammar) emits object properties in schema order, and an optional
property can only come before the ones after it. Ollama does not enforce the order, so the
teacher writes events as type, description, quote, time, ... while the schema says type,
description, time, location, participants, quote. A student trained on the teacher's order
writes `quote` right after `description`, and the grammar then has no way back to `time`:
every event loses its date. Same content, schema order, fixes that.

    python -m distill.schema_order            # rewrites ~/distill-data/sft.jsonl in place
"""

from __future__ import annotations

import json
import shutil
import sys

from distill.paths import data_dir


def _resolve(node: dict, defs: dict) -> dict:
    if "$ref" in node:
        return defs[node["$ref"].rsplit("/", 1)[-1]]
    for alt in node.get("anyOf", []):
        if alt.get("type") != "null":
            return _resolve(alt, defs)
    return node


def reorder(value, node: dict, defs: dict):
    node = _resolve(node, defs)
    if isinstance(value, dict) and "properties" in node:
        props = node["properties"]
        known = [k for k in props if k in value]
        extra = [k for k in value if k not in props]
        return {k: reorder(value[k], props[k], defs) if k in props else value[k]
                for k in known + extra}
    if isinstance(value, list) and "items" in node:
        return [reorder(v, node["items"], defs) for v in value]
    return value


def schema() -> dict:
    """The production extraction schema, as exported for the benchmark (same for all docs)."""
    line = (data_dir() / "bench_prompts.jsonl").open(encoding="utf-8").readline()
    return json.loads(line)["schema"]


def main() -> None:
    s = schema()
    defs = s.get("$defs", {})
    p = data_dir() / "sft.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    changed = 0
    for r in rows:
        msg = r["messages"][-1]
        out = json.loads(msg["content"])
        fixed = reorder(out, s, defs)
        assert fixed == out  # same content: dict equality ignores order
        new = json.dumps(fixed, ensure_ascii=False, separators=(",", ":"))
        changed += new != msg["content"]
        msg["content"] = new
    backup = p.with_suffix(".teacher_order.jsonl")
    if not backup.exists():
        shutil.copyfile(p, backup)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                 encoding="utf-8")
    print(f"{len(rows)} examples, {changed} reordered -> {p} (original: {backup.name})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
