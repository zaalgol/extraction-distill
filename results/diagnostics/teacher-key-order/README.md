# Diagnostic: teacher key order vs guided decoding

First training run, superseded. The teacher (Ollama, no enforced key order) wrote events as
`type, description, quote, time, ...`; the schema orders them `type, description, time,
location, participants, quote, ...`. vLLM's guided decoding emits properties in schema
order, so the student, having written `quote`, could no longer emit `time`: every event
lost its date, and the scorer requires the date when the gold event has one.

- Qwen3-8B distilled FP8: entities 0.92, relations 0.73, **events 0.26** (`eval/`).
  The untuned base, which follows the schema order: events 0.73.
- `python bench/event_fields.py --model qwen3-4b-distill --n 12` on the 4B student of this run:

```
guided=True: 14 events, 0 with a date, 0 invalid JSON
   14  ('type', 'description', 'quote', 'confidence')
guided=False: 12 events, 9 with a date, 0 invalid JSON
    4  ('type', 'description', 'quote', 'time', 'location', 'participants')
    3  ('type', 'description', 'quote', 'time', 'location', 'confidence')
    2  ('type', 'description', 'quote', 'time', 'quote_en')
```

Fix: `distill/schema_order.py` rewrites the training answers in schema order (same
content), enforced by `tests/test_schema_order.py`; both students were retrained.
