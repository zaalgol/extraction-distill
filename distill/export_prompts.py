"""Export the production extraction requests for the gold documents (messages + JSON schema),
so the serving benchmark replays exactly the eval's workload without importing osint.

    uv run --project C:\\code\\osint-agents python -m distill.export_prompts
"""

from __future__ import annotations

import json
import os

from distill.paths import data_dir, gold_path, osint_home

os.environ.setdefault("OSINT_HOME", str(osint_home()))

from osint.core.context import App  # noqa: E402
from osint.core.models import DocumentView  # noqa: E402
from osint.plugins.extractors.llm import SYSTEM, build_user_prompt, schema_for  # noqa: E402

FOCUS = "terrorism, armed groups, attacks, threats, financing"   # as in osint's eval


def main() -> None:
    onto = App().ontology
    schema = schema_for(onto)
    out = data_dir() / "bench_prompts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for line in gold_path().read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            view = DocumentView(id=0, artifact_id=0, text=g["text"], investigation_focus=FOCUS)
            f.write(json.dumps({
                "id": g["id"],
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": build_user_prompt(view, onto)}],
                "schema_name": schema.__name__, "schema": schema.model_json_schema(),
            }, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
