# JevBench-mini (held-out cluster H6)

The headline zero-shot test for Tiny-Jev (design doc 15 §5.4): **300 hand-written, realistic
app decisions**, ~20 decision types, option sets the model never saw, labelled by you. Public
benchmarks are likely in Qwen3's pre-training data; this set is new, so it is the number we
trust.

## Rules

1. **Written by hand.** You can use an LLM for ideas, but write or rewrite every item yourself.
   No item may be copied from a public dataset.
2. **Never trained on, never tuned on.** No templates, option sets or texts from here go into
   `tasks.yaml` or any training data. Thresholds (τ) and temperatures are *not* fitted here.
3. **Freeze before training starts** (`python scripts/jevbench.py freeze`). Freezing records
   the SHA-256 of `items.jsonl` in `FROZEN.json`; evaluation refuses a changed file.
4. **Second annotator on 100 items** (`label_2`), for agreement (Cohen's κ, reported in the
   paper). Disagreements are discussed and the gold `label` fixed *before* freezing.
5. Balanced-ish: per type, no option should hold more than ~60% of the gold labels.

## Item format (`items.jsonl`, one JSON object per line)

```json
{"id": "jb-001", "type": "email_urgency",
 "question": "How urgent is this email?",
 "options": ["can wait", "reply today", "act now"],
 "state": {"header": "From: ops@... Subject: ...", "segments": []},
 "label": 2, "label_2": null, "author": "subham", "notes": ""}
```

- `state.header` holds the input. Use `segments` (list of `{"title", "text"}`) when the input is
  naturally several pieces (passages, messages in a thread, two tickets to compare).
- `options`: 2–10 options, in any order (the model is order-invariant). The gold option is
  `options[label]`.
- `type`: free text, ~20 distinct values, ~15 items each.

## Suggested types (~20)

email urgency · ticket routing · PII present · refund eligible (policy as a segment) ·
risky code change · bug severity · spam / not spam · contract clause type · invoice overdue ·
meeting request present · needs human follow-up · duplicate ticket (two segments) ·
question answerable from the doc · policy violation (policy + message) · action item present ·
calendar conflict · tone too informal for a customer · product category · order status intent ·
language of the message

`draft_items.jsonl` has 20 **drafts** (one per type) written by Claude, as a format example.
They are not part of the benchmark. Rewrite them in your own words or delete them, then write
the rest into `items.jsonl`.

## Commands

```bash
python scripts/jevbench.py validate jevbench/items.jsonl   # format + balance report
python scripts/jevbench.py agreement jevbench/items.jsonl  # Cohen's kappa on label_2
python scripts/jevbench.py freeze jevbench/items.jsonl     # writes jevbench/FROZEN.json
```
