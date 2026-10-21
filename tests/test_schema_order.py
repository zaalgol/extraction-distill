import json

import pytest

from distill.paths import data_dir
from distill.schema_order import reorder, schema


def test_event_keys_follow_schema():
    s = {"$defs": {"E": {"properties": {"type": {}, "time": {}, "quote": {}}}},
         "properties": {"events": {"items": {"$ref": "#/$defs/E"}}}}
    out = reorder({"events": [{"type": "attack", "quote": "q", "time": "2026"}]}, s, s["$defs"])
    assert list(out["events"][0]) == ["type", "time", "quote"]


def test_training_answers_in_schema_order():
    # A student trained on another key order loses fields under guided decoding.
    p = data_dir() / "sft.jsonl"
    if not p.exists() or not (data_dir() / "bench_prompts.jsonl").exists():
        pytest.skip("training data not found")
    s = schema()
    for line in p.read_text(encoding="utf-8").splitlines():
        content = json.loads(line)["messages"][-1]["content"]
        out = json.loads(content)
        assert json.dumps(reorder(out, s, s["$defs"]), ensure_ascii=False,
                          separators=(",", ":")) == content
