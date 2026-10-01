import json
import shutil
from pathlib import Path

import pytest

from jevcore.data import jevbench

DRAFTS = Path(__file__).resolve().parents[1] / "jevbench" / "draft_items.jsonl"


def test_drafts_are_valid():
    items = jevbench.read(DRAFTS)
    rep = jevbench.check(items)
    assert rep["n"] == 20 and not rep["problems"] and len(rep["types"]) == 20


def test_check_catches_problems():
    items = jevbench.read(DRAFTS)[:2]
    items[1] = {**items[1], "id": items[0]["id"], "label": 9}
    probs = jevbench.check(items)["problems"]
    assert any("duplicate id" in p for p in probs) and any("label out of range" in p for p in probs)


def test_kappa():
    base = {"options": ["a", "b"], "type": "t"}
    items = [{**base, "label": i % 2, "label_2": i % 2} for i in range(10)]
    assert jevbench.cohen_kappa(items) == (1.0, 10)
    items[0]["label_2"] = 1
    k, _ = jevbench.cohen_kappa(items)
    assert 0 < k < 1


def test_freeze_and_tamper(tmp_path):
    p = tmp_path / "items.jsonl"
    shutil.copy(DRAFTS, p)
    with pytest.raises(FileNotFoundError):
        jevbench.load_frozen(p)
    rec = jevbench.freeze(p)
    assert rec["n"] == 20
    packs = jevbench.load_frozen(p)
    assert packs[0]["decisions"][0]["name"].startswith("jevbench:")
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(jevbench.read(DRAFTS)[0] | {"id": "new"}) + "\n")
    with pytest.raises(ValueError, match="changed"):
        jevbench.load_frozen(p)
