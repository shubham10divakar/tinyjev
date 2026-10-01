"""Offline tests for augmentation and pack views (dataset downloads are not exercised)."""

import random
from collections import Counter

from conftest import sample_example

from jevcore.data.augment import AugConfig, augment, paraphrase, templates
from jevcore.data.builders import chunk_packs, qa_packs, to_nano_rows, unpack
from jevcore.schema import DECISIONS, validate


def _qa(minus_relevance="new", n_distract=1, seed=0):
    paras = [(f"T{i}", f"text {i}" + (" paris" if i == 0 else "")) for i in range(10)]
    return qa_packs("q1", "hotpot", "where ?", paras, [0, 1], ["paris"], random.Random(seed),
                    n_distract, minus_relevance)


def _labels(pack, name):
    for d in pack["decisions"]:
        if d["name"] == name:
            return d.get("labels", d.get("label"))
    return None


def test_qa_packs_eval_mix_matches_nano():
    full, minus = _qa()
    validate(full)
    validate(minus)
    assert _labels(full, "sufficient") == 0 and _labels(minus, "sufficient") == 1
    assert len(full["state"]["segments"]) == 3 and len(minus["state"]["segments"]) == 3
    # relevance: 2 gold (2 and 1) + 2 distractors per question, no item twice
    rel = Counter(_labels(full, "relevance") + _labels(minus, "relevance"))
    assert rel == {2: 1, 1: 1, 0: 2}


def test_qa_packs_train_keep_all():
    full, minus = _qa(minus_relevance=True, n_distract=None)
    assert len(full["state"]["segments"]) == 10 and len(minus["state"]["segments"]) == 9
    assert sorted(full["meta"]["gold"]) == sorted(
        k for k, s in enumerate(full["state"]["segments"]) if s["title"] in ("T0", "T1"))


def test_augment_keeps_labels_consistent():
    full, _ = _qa(minus_relevance=True, n_distract=None)
    rng = random.Random(0)
    for _ in range(200):
        a = augment(full, rng, AugConfig(verb_p=0.5, qpara_p=0.5))
        validate(a)
        segs = a["state"]["segments"]
        assert 2 <= len(segs) <= 10
        for d in a["decisions"]:
            canon = DECISIONS[d["name"]].options
            table = [tuple(o) for o in templates()[d["name"]]["options"]["train"]]
            # the option wording is one of the train wordings, possibly permuted
            assert any(sorted(d["options"]) == sorted(t) for t in table)
            if d["name"] == "relevance":
                for t, lab in zip(d["targets"], d["labels"]):
                    wording = next(w for w in table if sorted(w) == sorted(d["options"]))
                    expected = 2 if "paris" in segs[t]["text"] else (1 if segs[t]["title"] == "T1" else 0)
                    assert wording.index(d["options"][lab]) == expected
            else:
                wording = next(w for w in table if sorted(w) == sorted(d["options"]))
                assert wording.index(d["options"][d["label"]]) == 0
            assert len(canon) == len(d["options"])


def test_augment_question_templates_fill_claim():
    ex = sample_example()
    rng = random.Random(1)
    seen = set()
    for _ in range(100):
        for d in augment(ex, rng, AugConfig(qpara_p=1.0, verb_p=0.0)).get("decisions"):
            if d["name"] == "grounded":
                assert "penguins cannot fly" in d["question"] and "{" not in d["question"]
                seen.add(d["question"])
    assert len(seen) > 1


def test_heldout_templates_never_in_train():
    for name, t in templates().items():
        assert not set(t["questions"]["heldout"]) & set(t["questions"]["train"]), name
        assert not {tuple(o) for o in t["options"]["heldout"]} & {tuple(o) for o in t["options"]["train"]}
        assert t["questions"]["train"][0] == DECISIONS[name].question
        assert tuple(t["options"]["train"][0]) == DECISIONS[name].options


def test_paraphrase():
    p = paraphrase(sample_example(), 1, 0)
    validate(p)
    grd = next(d for d in p["decisions"] if d["name"] == "grounded")
    assert grd["options"] == ["confirmed", "unconfirmed"] and "penguins" in grd["question"]
    assert p["decisions"][-1]["question"] == "rate the tone ?"      # custom decisions untouched


def test_views():
    full, minus = _qa()
    rows = to_nano_rows(full) + to_nano_rows(minus)
    assert Counter(r["decision"] for r in rows) == {"relevance": 4, "sufficient": 2}
    assert all("where ?" in r["question"] for r in rows)
    parts = unpack(full)
    assert len(parts) == 4                     # 3 relevance + 1 sufficient
    for p in parts:
        validate(p)
    chunks = chunk_packs(full, 1, random.Random(0))
    assert len(chunks) == 3 and all(len(c["state"]["segments"]) == 1 for c in chunks)
    assert sorted(c["meta"]["orig_seg"][0] for c in chunks) == [0, 1, 2]
