"""JevBench-mini (H6, design doc 15 §5.4): load, validate, freeze, agreement, -> packs.

See jevbench/README.md for the rules. Decisions are named "jevbench:{type}", so evaluation
uses the global T_custom (never a temperature fitted on JevBench itself).
"""

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from ..schema import validate

FROZEN_NAME = "FROZEN.json"
MAX_SHARE = 0.6


def read(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_pack(item: dict) -> dict:
    return {"id": item["id"], "source": "jevbench", "split": "test",
            "state": {"header": item["state"].get("header", ""),
                      "segments": item["state"].get("segments", [])},
            "decisions": [{"name": f"jevbench:{item['type']}", "kind": "choice", "scope": "global",
                           "question": item["question"], "options": list(item["options"]),
                           "label": item["label"]}],
            "meta": {"family": "H6", "type": item["type"]}}


def check(items: list[dict]) -> dict:
    """Validate every item; return a report {n, types, problems, imbalanced}."""
    problems, ids = [], Counter(i.get("id") for i in items)
    for it in items:
        tag = it.get("id", "?")
        missing = [k for k in ("id", "type", "question", "options", "state", "label") if k not in it]
        if missing:
            problems.append(f"{tag}: missing {', '.join(missing)}")
            continue
        if ids[it["id"]] > 1:
            problems.append(f"{tag}: duplicate id")
        if not 2 <= len(it["options"]) <= 10:
            problems.append(f"{tag}: needs 2-10 options")
        if len(set(it["options"])) != len(it["options"]):
            problems.append(f"{tag}: duplicate options")
        l2 = it.get("label_2")
        if l2 is not None and not 0 <= l2 < len(it["options"]):
            problems.append(f"{tag}: label_2 out of range")
        try:
            validate(to_pack(it))
        except ValueError as e:
            problems.append(f"{tag}: {e}")
    by_type: dict[str, Counter] = {}
    for it in items:
        ok = {"type", "label", "options"} <= set(it) and 0 <= it["label"] < len(it["options"])
        if ok:
            by_type.setdefault(it["type"], Counter())[it["options"][it["label"]]] += 1
    imbalanced = {t: c.most_common(1)[0] for t, c in by_type.items()
                  if sum(c.values()) >= 5 and c.most_common(1)[0][1] / sum(c.values()) > MAX_SHARE}
    return {"n": len(items), "types": {t: sum(c.values()) for t, c in by_type.items()},
            "problems": problems, "imbalanced": imbalanced}


def cohen_kappa(items: list[dict]) -> tuple[float | None, int]:
    """Agreement between `label` and `label_2` on doubly-annotated items (option text level)."""
    pairs = [(it["options"][it["label"]], it["options"][it["label_2"]])
             for it in items if it.get("label_2") is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return (1.0 if pe == 1 else (po - pe) / (1 - pe)), n


def freeze(path) -> dict:
    path = Path(path)
    rep = check(read(path))
    if rep["problems"]:
        raise ValueError(f"fix {len(rep['problems'])} problems before freezing")
    rec = {"file": path.name, "sha256": sha256(path), "n": rep["n"], "types": rep["types"],
           "frozen_on": date.today().isoformat()}
    (path.parent / FROZEN_NAME).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def load_frozen(path) -> list[dict]:
    """Packs for evaluation; refuses a file that changed since `freeze`."""
    path = Path(path)
    frozen = path.parent / FROZEN_NAME
    if not frozen.exists():
        raise FileNotFoundError("JevBench-mini is not frozen yet (scripts/jevbench.py freeze)")
    rec = json.loads(frozen.read_text(encoding="utf-8"))
    if rec["sha256"] != sha256(path):
        raise ValueError(f"{path} changed after it was frozen on {rec['frozen_on']}")
    return [to_pack(it) for it in read(path)]
