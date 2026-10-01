"""Tiny-Jev data code, offline: tasks.yaml, converters (fake rows), augmentation, mixture
allocation, synthetic format family, held-out views. Real datasets: prepare_mixture.py --check."""

import random
import re
import string

import pytest

from jevcore.data import tasks as T
from jevcore.data.mixture import allocate, build_heldout, build_p2
from jevcore.data.synthetic import FEATURES, format_packs, has
from jevcore.data.tiny_augment import TinyAugConfig, augment_tiny, heldout_view, subsample_options
from jevcore.packing import MARKERS, PackConfig, add_markers, render
from jevcore.schema import validate

FAKE_ROWS = {
    "mnli": {"premise": "A man plays guitar.", "hypothesis": "A man makes music.", "label": 0},
    "snli": {"premise": "Two dogs run.", "hypothesis": "Cats sleep.", "label": 2},
    "fever": {"claim": "Paris is in France.", "evidence": [["Paris", 0, "Paris is the capital of France."]],
              "label": "SUPPORTS"},
    "boolq": {"question": "is paris the capital of france", "answer": True, "passage": "Paris is the capital."},
    "ag_news": {"text": "The team won the cup.", "label": 1},
    "dbpedia": {"title": "Kiwi", "content": " A flightless bird.", "label": 9},
    "yahoo": {"question_title": "Why is the sky blue?", "question_content": "", "best_answer": "x", "topic": 1},
    "trec": {"text": "Who wrote Hamlet?", "coarse_label": 3},
    "banking77": {"text": "My card has not arrived.", "label": 1},
    "clinc150": {"text": "set an alarm for 7", "intent": 2},
    "arc_easy": {"question": "Which is a mammal?", "choices": {"text": ["shark", "whale"], "label": ["A", "B"]},
                 "answerKey": "B"},
    "arc_challenge": {"question": "Q?", "choices": {"text": ["a", "b", "c"], "label": ["1", "2", "3"]},
                      "answerKey": "3"},
    "openbookqa": {"question_stem": "Sun is a", "choices": {"text": ["star", "planet"], "label": ["A", "B"]},
                   "answerKey": "A"},
    "commonsense_qa": {"question": "Where is a fork?", "choices": {"text": ["drawer", "sky"], "label": ["A", "B"]},
                       "answerKey": "A"},
    "sciq": {"question": "What do plants make?", "correct_answer": "oxygen", "distractor1": "gold",
             "distractor2": "iron", "distractor3": "salt", "support": "Plants make oxygen."},
    "hellaswag": {"ctx": "He picks up the ball and", "endings": ["throws it.", "eats it.", "sings.", "flies."],
                  "label": "0"},
    "paws": {"sentence1": "A is B.", "sentence2": "B is A.", "label": 1},
    "mrpc": {"sentence1": "It rained.", "sentence2": "The sun shone.", "label": 0},
    "sst2": {"sentence": "a lovely film", "label": 1},
    "imdb": {"text": "Terrible.", "label": 0},
    "yelp": {"text": "Great food!", "label": 1},
    "copa": {"premise": "The man fell.", "choice1": "He slipped.", "choice2": "He sang.", "question": "cause",
             "label": 0},
    "winogrande": {"sentence": "The cup is in the box because _ is small.", "option1": "cup", "option2": "box",
                   "answer": "1"},
    "tweet_offensive": {"text": "have a nice day", "label": 0},
    "tweet_hate": {"text": "hello", "label": 0},
}
NAMES = {"banking77": ["activate_my_card", "card_arrival", "card_linking"],
         "clinc150": ["oos", "alarm_query", "set_alarm"]}
CONVERTED = {k: s for k, s in T.SOURCES.items() if s.convert}


def _fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def _tok():
    from conftest import tiny_tokenizer
    return tiny_tokenizer()


# ------------------------------------------------------------------------------ tasks.yaml

def test_task_table_shape_and_fields():
    table = T.task_table()
    for name, t in table.items():
        q = t["questions"]
        assert len(q["train"]) == 8 and len(q["heldout"]) == 2, name
        assert not set(q["train"]) & set(q["heldout"]), name
        declared = set(t.get("fields", []))
        for tpl in q["train"] + q["heldout"]:
            assert _fields(tpl) <= declared, (name, tpl)
        if "options" in t:
            o = t["options"]
            assert len(o["train"]) >= 2 and len(o["heldout"]) >= 1, name
            assert len({len(w) for w in o["train"] + o["heldout"]}) == 1, name
            assert not {tuple(w) for w in o["train"]} & {tuple(w) for w in o["heldout"]}, name
        assert t["family"] in T.FAMILIES + ("heldout",), name


def test_every_source_has_a_task_and_quota_family():
    for s in T.SOURCES.values():
        assert s.task in T.task_table(), s.key
        assert s.family in T.FAMILIES or s.family in T.CLUSTERS, s.key
        assert s.convert or s.build, s.key
    assert abs(sum(T.QUOTAS.values()) - 1) < 1e-9
    assert all(s.release_ok for s in T.training_sources(release_only=True))
    assert {"ag_news", "yahoo", "sciq"} & {s.key for s in T.training_sources()}
    assert not {"ag_news", "yahoo", "sciq"} & {s.key for s in T.training_sources(release_only=True)}


def test_fake_rows_cover_all_converters():
    assert set(CONVERTED) == set(FAKE_ROWS)


# ------------------------------------------------------------------------------ converters

@pytest.mark.parametrize("key", sorted(CONVERTED))
def test_converter_builds_valid_renderable_pack(key):
    src = T.SOURCES[key]
    conv = src.convert(FAKE_ROWS[key], NAMES.get(key))
    assert conv is not None
    p = T.pack_from(src, 0, conv, "train")
    validate(p)
    d = p["decisions"][0]
    assert "{" not in d["question"] and 0 <= d["label"] < len(d["options"])
    tok = _tok()
    M = add_markers(tok, tuple(MARKERS))
    render(p, tok, M, PackConfig(max_len=512, causal=True, sink_id=tok.pad_token_id))


def test_converter_labels():
    c = lambda k: T.SOURCES[k].convert(FAKE_ROWS[k], NAMES.get(k))  # noqa: E731
    assert c("boolq")["label"] == 0 and c("boolq")["fields"]["question"] == "Is paris the capital of france"
    assert c("arc_easy")["options"][c("arc_easy")["label"]] == "whale"
    sci = c("sciq")
    assert sci["options"][sci["label"]] == "oxygen"
    assert c("paws")["label"] == 0 and c("mrpc")["label"] == 1          # yes = paraphrase
    assert c("banking77")["options"][1] == "card arrival"
    assert c("clinc150")["options"][0] == "out of scope"
    assert c("winogrande")["label"] == 0 and c("copa")["fields"]["relation"] == "cause"
    assert c("fever")["header"] == "Paris is the capital of France." and c("fever")["label"] == 0
    assert T.SOURCES["snli"].convert({**FAKE_ROWS["snli"], "label": -1}, None) is None


# ------------------------------------------------------------------------------ augmentation

def _gold_canonical_index(name: str, options: list[str], label: int) -> int:
    t = T.task_table()[name]
    for w in t["options"]["train"] + t["options"]["heldout"]:
        if set(options) <= set(w):
            return w.index(options[label])
    raise AssertionError((name, options))


def _which_pack(n_seg=6, gold=2):
    segs = [(f"T{i}", f"text {i}") for i in range(n_seg)]
    p = {"id": "w", "source": "t", "state": {"header": "query: q", "segments": [dict(title=a, text=b) for a, b in segs]},
         "decisions": [{"name": "relevance", "kind": "score", "scope": "segment",
                        "question": "How relevant is this passage to the query?",
                        "options": ["irrelevant", "partially relevant", "directly answers"],
                        "targets": list(range(n_seg)), "labels": [2 if i == gold else 0 for i in range(n_seg)]}],
         "meta": {"gold": [gold]}}
    return T.add_which_passage(p)


def test_which_passage_follows_shuffle_and_k():
    rng = random.Random(0)
    base = _which_pack()
    for _ in range(300):
        a = augment_tiny(base, rng, TinyAugConfig(subset_p=1.0, multi_template_p=0.0, verb_p=0.5))
        validate(a)
        w = [d for d in a["decisions"] if d["name"] == "which_passage"]
        if not w:
            continue
        d = w[0]
        gold_opt = d["options"][d["label"]]
        num = int(re.search(r"\d+", gold_opt).group())
        assert a["state"]["segments"][num - 1]["title"] == "T2"
        assert len(d["options"]) == len(a["state"]["segments"]) + 1


def test_which_passage_none():
    p = _which_pack(gold=0)
    p["decisions"][0]["labels"] = [0] * 6
    p = T.add_which_passage({**p, "decisions": p["decisions"][:1]})
    d = p["decisions"][-1]
    assert d["options"][d["label"]] == "none of them"


def test_augment_keeps_gold_meaning():
    rng = random.Random(1)
    src = T.SOURCES["ag_news"]
    base = T.pack_from(src, 0, src.convert(FAKE_ROWS["ag_news"], None), "train")   # sports
    mc = T.pack_from(T.SOURCES["hellaswag"], 0, T.SOURCES["hellaswag"].convert(FAKE_ROWS["hellaswag"], None), "train")
    heldout_qs = set(T.task_table()["news_topic"]["questions"]["heldout"])
    seen_q, seen_w = set(), set()
    for _ in range(300):
        a = augment_tiny(base, rng)
        for d in a["decisions"]:
            assert _gold_canonical_index("news_topic", d["options"], d["label"]) == 1
            assert d["question"] not in heldout_qs
            seen_q.add(d["question"])
            seen_w.add(tuple(sorted(d["options"])))
        b = augment_tiny(mc, rng)
        for d in b["decisions"]:
            assert d["options"][d["label"]] == "throws it."
    assert len(seen_q) == 8 and len(seen_w) == 2      # all train templates, both train wordings


def test_single_template_ablation_is_canonical():
    rng = random.Random(2)
    src = T.SOURCES["dbpedia"]
    base = T.pack_from(src, 0, src.convert(FAKE_ROWS["dbpedia"], None), "train")
    q0, opts = T.canonical("entity_type")
    for _ in range(100):
        a = augment_tiny(base, rng, TinyAugConfig(single_template=True, opt_subsample=False))
        (d,) = a["decisions"]
        assert d["question"] == q0 and sorted(d["options"]) == sorted(opts)


def test_subsample_options_keeps_gold():
    rng = random.Random(3)
    for _ in range(200):
        d = {"scope": "global", "options": [str(i) for i in range(77)], "label": 41}
        subsample_options(d, rng)
        assert d["options"][d["label"]] == "41" and 2 <= len(d["options"]) <= 77
    small = {"scope": "global", "options": ["a", "b", "c"], "label": 2}
    subsample_options(small, rng)
    assert small["options"] == ["a", "b", "c"]


def test_multi_template_packs():
    rng = random.Random(4)
    src = T.SOURCES["trec"]
    base = T.pack_from(src, 0, src.convert(FAKE_ROWS["trec"], None), "train")
    sizes = set()
    for _ in range(100):
        a = augment_tiny(base, rng, TinyAugConfig(multi_template_p=1.0))
        validate(a)
        sizes.add(len(a["decisions"]))
        qs = [d["question"] for d in a["decisions"]]
        assert len(set(qs)) == len(qs)                   # different templates in one pack
    assert sizes == {2, 3}


def test_heldout_view():
    src = T.SOURCES["boolq"]
    p = T.pack_from(src, 0, src.convert(FAKE_ROWS["boolq"], None), "val")
    v = heldout_view(p, 1, 0)
    d = v["decisions"][0]
    assert d["question"] == T.task_table()["yes_no_qa"]["questions"]["heldout"][1].format(
        question="Is paris the capital of france")
    assert d["options"] == ["correct", "incorrect"] and d["label"] == 0
    w = heldout_view(_which_pack(), 0, 0)["decisions"][-1]
    assert w["options"][0] == "document 1" and w["options"][-1] == "none"


# ------------------------------------------------------------------------------ synthetic

def test_format_labels_are_truthful():
    for p in format_packs(300, seed=0):
        validate(p)
        for d in p["decisions"]:
            assert (d["label"] == 0) == has(d["feature"], p["state"]["header"])
    labels = [d["label"] for p in format_packs(300) for d in p["decisions"]]
    assert 0.1 < sum(1 for x in labels if x == 0) / len(labels) < 0.9


def test_format_generators_detected():
    rng = random.Random(0)
    for f, (_, rx, gen) in FEATURES.items():
        for _ in range(20):
            assert rx.search(gen(rng)), f


# ------------------------------------------------------------------------------ mixture

def test_allocate_quotas_caps_and_sqrt_weighting():
    avail = {"a": 100_000, "b": 2_500, "c": 50_000, "d": 10}
    fam = {"a": "rag", "b": "rag", "c": "topic", "d": "topic"}
    alloc, short = allocate(avail, fam, 10_000, {"rag": 0.5, "topic": 0.5}, alpha=0.5, cap=20_000)
    assert alloc["a"] + alloc["b"] == 5000 and alloc["c"] + alloc["d"] == 5000
    # sqrt(20000)/sqrt(2500) ~ 2.83
    assert 2.5 < alloc["a"] / alloc["b"] < 3.2
    assert alloc["d"] <= 10 and not short
    alloc, short = allocate({"a": 100}, {"a": "rag"}, 10_000, {"rag": 0.5, "topic": 0.5})
    assert alloc["a"] == 100 and short == {"rag": 4900, "topic": 5000}


def _fake_loader(src, split, n, seed):
    if src.convert:
        row = FAKE_ROWS[src.key]
        return [T.pack_from(src, i, src.convert(row, NAMES.get(src.key)), split) for i in range(min(n, 50))]
    if src.family == "rag":
        return [{**_which_pack(), "id": f"{src.key}-{split}-{i}", "source": src.key,
                 "meta": {"gold": [2], "family": "rag"}} for i in range(min(n, 50))]
    if src.key.startswith("cyber"):
        return [{"id": f"{src.key}-{i}", "source": src.key, "state": {"header": "GET /?id=1 OR 1=1", "segments": []},
                 "decisions": [{"name": "http_attack", "kind": "noul", "scope": "global",
                                "question": T.canonical("http_attack")[0], "options": ["safe", "attack"],
                                "label": 1}], "meta": {"family": src.family}} for i in range(min(n, 50))]
    return []


def test_build_p2_with_fake_loader():
    out = build_p2(total=1000, cap=50, loader=_fake_loader, log=lambda *_: None)
    card = out["card"]
    assert card["total"] == len(out["train"]) <= 1000
    assert card["families"]["format"] == 30                 # 3% synthetic, generated to size
    for p in out["train"][:200]:
        validate(p)
    assert len(out["val_seen"]) == len(out["val_unseen"]) > 0 and len(out["test"]) > 0
    heldout_q = {q for t in T.task_table().values() for q in t["questions"]["heldout"]}
    filled = sum(1 for p in out["val_unseen"] for d in p["decisions"]
                 if d["name"] not in ("relevance", "which_passage") or d["question"] in heldout_q)
    assert filled == sum(len(p["decisions"]) for p in out["val_unseen"])
    rel = build_p2(total=1000, cap=50, loader=_fake_loader, release_only=True, log=lambda *_: None)
    assert all(T.SOURCES[k].release_ok for k in rel["card"]["sources"] if k in T.SOURCES)


def test_build_heldout_with_fake_loader():
    h = build_heldout(5, loader=_fake_loader, clusters=("H1", "H2", "H3"), all_templates=True)
    assert set(h) == {"H1", "H2", "H3"} and all(h.values())
    assert {p["decisions"][0]["name"] for p in h["H1"]} == {"sentiment"}
