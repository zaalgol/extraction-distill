import json

import pytest

from distill.leakage import Guard
from distill.paths import data_dir, gold_path
from distill.specs import sample


@pytest.fixture(scope="module")
def guard() -> Guard:
    if not gold_path().exists():
        pytest.skip("osint-agents gold set not found")
    return Guard.from_gold()


@pytest.fixture(scope="module")
def gold() -> list[dict]:
    return [json.loads(x) for x in gold_path().read_text(encoding="utf-8").splitlines() if x]


def test_catches_gold_names(guard):
    assert guard.hits("Officials named Farid Okonjo as the commander.")
    assert guard.hits("the CRIMSON DAWN BRIGADE said")             # case-insensitive
    assert guard.hits("أعلنت سرايا الفجر مسؤوليتها")                # non-Latin substring
    assert not guard.hits("The Silver River Front claimed the attack.")
    assert not guard.hits("Ardanelli Logistics")                    # word boundary for Latin


def test_catches_gold_8grams(guard, gold):
    text = gold[0]["text"]
    words = text.split()
    assert any(h.startswith("8gram:") for h in guard.hits(" ".join(words[2:12])))


def test_every_gold_text_is_caught(guard, gold):
    assert all(guard.hits(g["text"]) for g in gold if len(g["text"].split()) >= 8)


def test_spec_names_are_clean(guard):
    for s in sample(2000, seed=7):
        for v in s.names.values():
            assert guard.clean(v), (s.id, v)


def test_sampling_is_deterministic():
    assert [s.as_dict() for s in sample(50, 7)] == [s.as_dict() for s in sample(50, 7)]


@pytest.mark.parametrize("name", ["docs.jsonl", "sft.jsonl"])
def test_training_data_is_clean(guard, name):
    """The real guarantee: run on the generated data, wherever it lives."""
    p = data_dir() / name
    if not p.exists():
        pytest.skip(f"{p} not generated yet")
    bad = []
    for line in p.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        text = (rec["text"] if "text" in rec
                else "\n".join(m["content"] for m in rec["messages"]))
        if hits := guard.hits(text):
            bad.append((rec["id"], hits[:3]))
    assert not bad, bad[:10]
