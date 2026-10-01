"""Tiny-Jev data sources (design doc 15 §5.2, §5.4): one entry per dataset with its HF id,
splits, licence, family and a converter from a dataset row to a packed example.

**Every HF id, config and field name here is from memory and must be confirmed in M1**
(`scripts/prepare_mixture.py --check` loads a few rows of each and runs the converter).
Licences are our best reading of each dataset card; `commercial=None` means unclear. The
release mixture keeps only `commercial is True` sources (§5.2 rule 4).

A converter gets one row (and the label names, for datasets whose labels are class ids) and
returns
    {"header": str, "segments": [(title, text)], "label": int, "options": [...] | None,
     "fields": {...}}
or None to skip the row. `options=None` means the task's canonical wording from tasks.yaml.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..schema import IGNORE, make_state, query_header
from . import builders

TASKS_PATH = Path(__file__).resolve().parents[1] / "tasks.yaml"
FAMILIES = ("rag", "verification", "yesno", "topic", "mcq", "paraphrase", "security", "format")
QUOTAS = {"rag": 0.35, "verification": 0.12, "yesno": 0.05, "topic": 0.15, "mcq": 0.15,
          "paraphrase": 0.05, "security": 0.10, "format": 0.03}            # §5.2
CLUSTERS = {"H1": "sentiment", "H2": "commonsense causal / coreference", "H3": "toxicity",
            "H4": "RAG OOD", "H5": "security OOD", "H6": "JevBench-mini"}

# which_passage option wordings: [i] is 1-based. Last entry = "none" wording.
PASSAGE_FORMATS = {"train": [("[{i}]", "none of them"), ("passage {i}", "no passage")],
                   "heldout": [("document {i}", "none")]}


@lru_cache(maxsize=None)
def task_table() -> dict:
    return yaml.safe_load(TASKS_PATH.read_text(encoding="utf-8"))


def canonical(task: str) -> tuple[str, list[str] | None]:
    t = task_table()[task]
    opts = t.get("options", {}).get("train", [None])[0]
    return t["questions"]["train"][0], (list(opts) if opts else None)


@dataclass(frozen=True)
class Source:
    key: str
    task: str                         # decision name = tasks.yaml key
    family: str                       # one of FAMILIES, or a held-out cluster "H1".."H5"
    hf: str | None                    # HF dataset id (None: local file / custom builder)
    config: str | None = None
    splits: dict = field(default_factory=lambda: {"train": "train", "val": "validation"})
    licence: str = "unknown"
    commercial: bool | None = None    # True ok for release; False non-commercial; None unclear
    convert: Callable | None = None   # row, names -> dict | None
    label_feature: str | None = None  # read class names from this feature (Banking77, CLINC)
    build: Callable | None = None     # custom: (split, n, rng, seed) -> packs (RAG, local files)
    note: str = ""

    @property
    def release_ok(self) -> bool:
        return self.commercial is True


# ------------------------------------------------------------------------------ helpers

def humanise(label: str) -> str:
    return label.replace("_", " ").replace("oos", "out of scope").strip()


def _ans_index(choices: dict, key: str) -> int | None:
    labels = list(choices["label"])
    return labels.index(key) if key in labels else None


def pack_from(src: Source, rid, conv: dict, split: str) -> dict:
    """A converted row -> packed example with the canonical question / options."""
    q, opts = canonical(src.task)
    fields = conv.get("fields") or {}
    options = conv.get("options") or opts
    if options is None:
        raise ValueError(f"{src.key}: task {src.task} needs options from the data")
    t = task_table()[src.task]
    d = {"name": src.task, "kind": t.get("kind", "choice"), "scope": "global",
         "question": q.format(**fields), "options": list(options), "label": conv["label"]}
    if fields:
        d["fields"] = dict(fields)
    if conv.get("options") is not None:
        d["data_options"] = True                  # option texts come from the row: no rewording
    return {"id": f"{src.key}-{split}-{rid}", "source": src.key, "split": split,
            "state": make_state(conv.get("header", ""), conv.get("segments", ())),
            "decisions": [d], "meta": {"family": src.family}}


# ------------------------------------------------------------------------------ converters

def _nli(row, _):
    if row["label"] not in (0, 1, 2):
        return None
    return {"header": row["premise"], "label": row["label"], "fields": {"hypothesis": row["hypothesis"]}}


FEVER_LABELS = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def _fever(row, _):
    lab = FEVER_LABELS.get(str(row["label"]).upper())
    ev = row.get("evidence") or ""
    if isinstance(ev, list):     # list of sentences, or of [title, sent_id, text] triples
        ev = " ".join(e[-1] if isinstance(e, list | tuple) else str(e) for e in ev)
    if lab is None or not ev.strip():
        return None
    return {"header": ev, "label": lab, "fields": {"claim": row["claim"]}}


def _boolq(row, _):
    q = row["question"].strip().rstrip("?")
    return {"header": row["passage"], "label": 0 if row["answer"] else 1,
            "fields": {"question": q[:1].upper() + q[1:]}}


def _text_label(text_key="text", label_key="label", remap=None):
    def conv(row, _):
        lab = row[label_key]
        lab = remap[lab] if remap else lab
        return None if lab is None or lab < 0 else {"header": row[text_key], "label": lab}
    return conv


def _dbpedia(row, _):
    return {"header": f"{row['title']}: {row['content'].strip()}", "label": row["label"]}


def _yahoo(row, _):
    text = " ".join(x for x in (row["question_title"], row["question_content"]) if x).strip()
    return {"header": text, "label": row["topic"]}


def _named_labels(text_key, label_key):
    """Class-id datasets with many classes: options = all humanised class names."""
    def conv(row, names):
        return {"header": row[text_key], "label": row[label_key],
                "options": [humanise(n) for n in names]}
    return conv


def _mc(question_key="question", context=None):
    def conv(row, _):
        lab = _ans_index(row["choices"], row["answerKey"])
        if lab is None:
            return None
        ctx = row.get(context) if context else None
        return {"header": f"question: {row[question_key]}", "segments": [("", ctx)] if ctx else [], "label": lab,
                "options": list(row["choices"]["text"])}
    return conv


def _sciq(row, _):
    opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    order = list(range(4))
    random.Random(row["question"]).shuffle(order)      # deterministic per row
    segs = [("", row["support"])] if row.get("support") else []
    return {"header": f"question: {row['question']}", "segments": segs,
            "options": [opts[i] for i in order], "label": order.index(0)}


def _hellaswag(row, _):
    if str(row.get("label", "")).strip() == "":
        return None
    return {"header": row["ctx"], "options": list(row["endings"]), "label": int(row["label"])}


def _pair(yes_value=1):
    def conv(row, _):
        return {"segments": [("text 1", row["sentence1"]), ("text 2", row["sentence2"])],
                "label": 0 if row["label"] == yes_value else 1}
    return conv


def _copa(row, _):
    return {"header": row["premise"], "options": [row["choice1"], row["choice2"]],
            "label": row["label"], "fields": {"relation": row["question"]}}


def _winogrande(row, _):
    if row["answer"] not in ("1", "2"):
        return None
    return {"header": row["sentence"], "options": [row["option1"], row["option2"]],
            "label": int(row["answer"]) - 1}


# ------------------------------------------------------------------------------ RAG builders

def passage_options(n: int, fmt: tuple[str, str] = PASSAGE_FORMATS["train"][0]) -> list[str]:
    return [fmt[0].format(i=i + 1) for i in range(n)] + [fmt[1]]


def which_passage_decision(n_seg: int, answer_seg: int | None) -> dict:
    """Global choice over the passages (+ "none"). `answer_seg` is kept so augmentation can
    re-derive options and label after shuffling / dropping segments."""
    q, _ = canonical("which_passage")
    opts = passage_options(n_seg)
    return {"name": "which_passage", "kind": "choice", "scope": "global", "question": q,
            "options": opts, "label": n_seg if answer_seg is None else answer_seg,
            "answer_seg": -1 if answer_seg is None else answer_seg}


def add_which_passage(pack: dict) -> dict:
    """Add which_passage from the pack's relevance labels: exactly one 'directly answers'
    segment -> it; none -> "none of them"; several -> not added (ambiguous)."""
    rel = next((d for d in pack["decisions"] if d["name"] == "relevance"), None)
    if rel is None or len(rel["targets"]) != len(pack["state"]["segments"]):
        return pack
    answering = [t for t, lab in zip(rel["targets"], rel.get("labels", [])) if lab == 2]
    if len(answering) <= 1:
        pack["decisions"].append(which_passage_decision(len(pack["state"]["segments"]),
                                                        answering[0] if answering else None))
    return pack


def _tag(packs: list[dict], family: str) -> list[dict]:
    for p in packs:
        p.setdefault("meta", {})["family"] = family
    return packs


def hotpot_build(split, n, rng, seed=0):
    return _tag([add_which_passage(p) for p in builders.hotpot_packs(split, n, rng, seed=seed)], "rag")


def twowiki_build(split, n, rng, seed=0):
    """2WikiMultihopQA in the HotpotQA layout (context titles / sentences, supporting_facts)."""
    ds = builders._load(TWOWIKI_ID, split=split).shuffle(seed=seed).select(range(n))
    out = []
    for row in ds:
        paras = builders._hotpot_paragraphs(row)
        titles = list(paras)
        gold_t = [t for t in dict.fromkeys(row["supporting_facts"]["title"]) if t in paras]
        if len(gold_t) < 2 or len(titles) - len(gold_t) < 3:
            continue
        out += [add_which_passage(p) for p in builders.qa_packs(
            row["_id"] if "_id" in row else row["id"], "2wiki", row["question"],
            [(t, paras[t]) for t in titles], [titles.index(t) for t in gold_t],
            [row["answer"].strip().lower()], rng)]
    return _tag(out, "rag")


def squad2_build(split, n, rng, seed=0):
    """sufficient (one passage) + which_passage over 3–5 contexts of the same article."""
    ds = builders._load(builders.SQUAD2[0], split=split).shuffle(seed=seed)
    by_title: dict[str, list[str]] = {}
    rows = []
    for row in ds:
        ctxs = by_title.setdefault(row["title"], [])
        if row["context"] not in ctxs:
            ctxs.append(row["context"])
        rows.append(row)
        if len(rows) >= 4 * n:
            break
    out = []
    for row in rows[:n]:
        answerable = len(row["answers"]["text"]) > 0
        others = [c for c in by_title[row["title"]] if c != row["context"]]
        k = min(len(others), rng.randint(2, 4))
        segs = [row["context"]] + rng.sample(others, k)
        rng.shuffle(segs)
        title = row["title"].replace("_", " ")
        gold = segs.index(row["context"]) if answerable else None
        out.append({"id": f"squad2w-{split}-{row['id']}", "source": "squad2", "split": split,
                    "state": make_state(query_header(row["question"]), [(title, s) for s in segs]),
                    "decisions": [which_passage_decision(len(segs), gold)],
                    "meta": {"query": row["question"]}})
    return _tag(builders.squad2_packs(split, n, rng) + out, "rag")


def nq_build(split, n, rng, seed=0):
    """Natural Questions (query, gold Wikipedia passage) pairs; negatives are other rows'
    passages. Pack = query + gold + 2–6 negatives: relevance + which_passage."""
    ds = builders._load(NQ_ID, split="train").shuffle(seed=seed)
    # the HF set has one split: hold out the tail for val
    total = len(ds)
    lo, hi = (0, total - 2000) if split == "train" else (total - 2000, total)
    rows = ds.select(range(lo, min(hi, lo + 3 * n)))
    passages = [r["answer"] for r in rows]
    out = []
    for i, row in enumerate(rows):
        if len(out) >= n:
            break
        negs = rng.sample([p for j, p in enumerate(passages) if j != i], rng.randint(2, 6))
        segs = [row["answer"]] + negs
        order = list(range(len(segs)))
        rng.shuffle(order)
        segs = [segs[k] for k in order]
        gold = order.index(0)
        labels = [2 if k == gold else 0 for k in range(len(segs))]
        out.append(add_which_passage({
            "id": f"nq-{split}-{i}", "source": "nq", "split": split,
            "state": make_state(query_header(row["query"]), [("", s) for s in segs]),
            "decisions": [{"name": "relevance", "kind": "score", "scope": "segment",
                           "question": canonical("relevance")[0],
                           "options": canonical("relevance")[1],
                           "targets": list(range(len(segs))), "labels": labels}],
            "meta": {"query": row["query"], "gold": [gold]}}))
    return _tag(out, "rag")


def cyber_build(path):
    def build(split, n, rng, seed=0):
        packs = builders.cyber_packs(path)
        rng2 = random.Random(seed)
        rng2.shuffle(packs)
        for p in packs:
            for d in p["decisions"]:      # canonical tasks.yaml question (= Cyber-Jev's)
                d["question"] = canonical(d["name"])[0]
        return _tag(packs[:n], "security")
    return build


def musique_build(split, n, rng, seed=0):
    return _tag(builders.musique_packs(n, rng, seed, n_distract=1), "H4")


def vitaminc_build(split, n, rng, seed=0):
    return _tag(builders.vitaminc_packs(n, rng, seed), "H4")


# ------------------------------------------------------------------------------ registry

TWOWIKI_ID = "framolfese/2WikiMultihopQA"          # confirm in M1 (several mirrors exist)
NQ_ID = "sentence-transformers/natural-questions"   # (query, answer passage) pairs
CYBER_DIR = Path(__file__).resolve().parents[3] / "cyber_jev"   # code_repo/cyber_jev


def _s(key, task, family, hf, **kw) -> Source:
    return Source(key=key, task=task, family=family, hf=hf, **kw)


SOURCES: dict[str, Source] = {s.key: s for s in [
    # ---- RAG control (35%)
    _s("hotpot", "relevance", "rag", "hotpotqa/hotpot_qa", config="distractor",
       licence="CC BY-SA 4.0", commercial=True, build=hotpot_build,
       note="relevance + sufficient + which_passage packs (Micro-Jev phase A)"),
    _s("2wiki", "relevance", "rag", TWOWIKI_ID, licence="Apache-2.0 (check mirror)",
       commercial=True, build=twowiki_build, splits={"train": "train", "val": "validation"}),
    _s("squad2", "sufficient", "rag", "rajpurkar/squad_v2", licence="CC BY-SA 4.0",
       commercial=True, build=squad2_build),
    _s("nq", "which_passage", "rag", NQ_ID, licence="CC BY-SA 3.0", commercial=True,
       build=nq_build, splits={"train": "train", "val": "val"},
       note="FlashRAG NQ has no passages; this pair set does. Val = last 2000 rows"),
    # ---- verification / NLI (12%)
    _s("mnli", "nli", "verification", "nyu-mll/multi_nli", convert=_nli,
       splits={"train": "train", "val": "validation_matched"},
       licence="mixed (OANC; parts CC BY-SA 3.0 / CC BY 3.0)", commercial=True),
    _s("snli", "nli", "verification", "stanfordnlp/snli", convert=_nli,
       licence="CC BY-SA 4.0", commercial=True),
    _s("fever", "fact_check", "verification", "copenlu/fever_gold_evidence", convert=_fever,
       licence="CC BY-SA 3.0", commercial=True,
       note="needs a FEVER variant with evidence text; confirm id and fields"),
    # ---- yes / no QA (5%)
    _s("boolq", "yes_no_qa", "yesno", "google/boolq", convert=_boolq, licence="CC BY-SA 3.0",
       commercial=True),
    # ---- topic / intent (15%)
    _s("ag_news", "news_topic", "topic", "fancyzhx/ag_news", convert=_text_label(),
       splits={"train": "train", "val": "test"}, licence="non-commercial research (AG corpus)",
       commercial=False),
    _s("dbpedia", "entity_type", "topic", "fancyzhx/dbpedia_14", convert=_dbpedia,
       splits={"train": "train", "val": "test"}, licence="CC BY-SA 3.0", commercial=True),
    _s("yahoo", "question_topic", "topic", "community-datasets/yahoo_answers_topics",
       convert=_yahoo, splits={"train": "train", "val": "test"},
       licence="Yahoo Webscope (non-commercial)", commercial=False),
    _s("trec", "question_type", "topic", "CogComp/trec",
       convert=_text_label(label_key="coarse_label"), splits={"train": "train", "val": "test"},
       licence="unspecified", commercial=None),
    _s("banking77", "banking_intent", "topic", "PolyAI/banking77",
       convert=_named_labels("text", "label"), label_feature="label",
       splits={"train": "train", "val": "test"}, licence="CC BY 4.0", commercial=True),
    _s("clinc150", "intent", "topic", "clinc/clinc_oos", config="plus",
       convert=_named_labels("text", "intent"), label_feature="intent",
       licence="CC BY 3.0", commercial=True),
    # ---- multiple choice (15%)
    _s("arc_easy", "science_qa", "mcq", "allenai/ai2_arc", config="ARC-Easy", convert=_mc(),
       licence="CC BY-SA 4.0", commercial=True),
    _s("arc_challenge", "science_qa", "mcq", "allenai/ai2_arc", config="ARC-Challenge",
       convert=_mc(), licence="CC BY-SA 4.0", commercial=True),
    _s("openbookqa", "science_qa", "mcq", "allenai/openbookqa", config="main",
       convert=_mc("question_stem"), licence="Apache-2.0 (check card)", commercial=True),
    _s("commonsense_qa", "commonsense_qa", "mcq", "tau/commonsense_qa", convert=_mc(),
       licence="MIT", commercial=True),
    _s("sciq", "science_qa", "mcq", "allenai/sciq", convert=_sciq, licence="CC BY-NC 3.0",
       commercial=False),
    _s("hellaswag", "story_completion", "mcq", "Rowan/hellaswag", convert=_hellaswag,
       licence="MIT", commercial=True),
    # ---- paraphrase (5%)
    _s("paws", "paraphrase", "paraphrase", "google-research-datasets/paws",
       config="labeled_final", convert=_pair(1), licence="free use with attribution (Google)",
       commercial=True),
    _s("mrpc", "paraphrase", "paraphrase", "nyu-mll/glue", config="mrpc", convert=_pair(1),
       licence="unspecified (MSR)", commercial=None),
    # ---- security (10%): Cyber-Jev splits, local JSONL
    _s("cyber", "http_attack", "security", None, build=cyber_build(CYBER_DIR / "data" / "train.jsonl"),
       licence="per source, see cyber_jev/DATA.md (3 sources research-only)", commercial=None,
       note="http_attack + prompt_injection + phishing_url; val uses data/calib.jsonl"),
    # format robustness (3%) is synthetic: data/synthetic.py

    # ---- held-out clusters (never trained)
    _s("sst2", "sentiment", "H1", "stanfordnlp/sst2", convert=_text_label("sentence"),
       splits={"test": "validation"}, licence="unspecified", commercial=None),
    _s("imdb", "sentiment", "H1", "stanfordnlp/imdb", convert=_text_label(),
       splits={"test": "test"}, licence="unspecified (academic)", commercial=None),
    _s("yelp", "sentiment", "H1", "fancyzhx/yelp_polarity", convert=_text_label(),
       splits={"test": "test"}, licence="Yelp dataset terms", commercial=False),
    _s("copa", "copa", "H2", "aps/super_glue", config="copa", convert=_copa,
       splits={"test": "validation"}, licence="BSD-2", commercial=True),
    _s("winogrande", "coreference", "H2", "allenai/winogrande", config="winogrande_xl",
       convert=_winogrande, splits={"test": "validation"}, licence="CC BY 4.0", commercial=True),
    _s("tweet_offensive", "offensive", "H3", "cardiffnlp/tweet_eval", config="offensive",
       convert=_text_label(), splits={"test": "test"}, licence="see tweet_eval card",
       commercial=None),
    _s("tweet_hate", "hate_speech", "H3", "cardiffnlp/tweet_eval", config="hate",
       convert=_text_label(), splits={"test": "test"}, licence="see tweet_eval card",
       commercial=None),
    _s("musique", "relevance", "H4", "bdsaglam/musique", build=musique_build,
       splits={"test": "dev"}, licence="CC BY 4.0", commercial=True),
    _s("vitaminc", "grounded", "H4", "tals/vitaminc", build=vitaminc_build,
       splits={"test": "test"}, licence="CC BY-SA 3.0", commercial=True),
    _s("cyber_heldout", "http_attack", "H5", None,
       build=cyber_build(CYBER_DIR / "data_heldout" / "test.jsonl"), splits={"test": "test"},
       licence="see cyber_jev/DATA.md", commercial=None),
]}

CYBER_VAL = CYBER_DIR / "data" / "calib.jsonl"


def training_sources(release_only: bool = False) -> list[Source]:
    out = [s for s in SOURCES.values() if s.family in FAMILIES]
    return [s for s in out if s.release_ok] if release_only else out


def heldout_sources(clusters=("H1", "H2", "H3", "H4", "H5")) -> list[Source]:
    return [s for s in SOURCES.values() if s.family in clusters]


def load_source(src: Source, split: str, n: int, seed: int = 0) -> list[dict]:
    """Up to n packs of `split` ("train" / "val" / "test"). Needs network on first use."""
    rng = random.Random(f"{src.key}/{split}/{seed}")
    hf_split = src.splits.get(split)
    if hf_split is None:
        raise KeyError(f"{src.key} has no {split!r} split")
    if src.key == "cyber" and split == "val":
        return cyber_build(CYBER_VAL)(split, n, rng, seed)
    if src.build:
        return src.build(hf_split, n, rng, seed)
    args = (src.hf, src.config) if src.config else (src.hf,)
    ds = builders._load(*args, split=hf_split).shuffle(seed=seed)
    names = ds.features[src.label_feature].names if src.label_feature else None
    out = []
    for i, row in enumerate(ds):
        conv = src.convert(row, names)
        if conv is not None:
            out.append(pack_from(src, i, conv, split))
        if len(out) >= n:
            break
    return out


def labelled(pack: dict) -> bool:
    return any(d.get("label", IGNORE) != IGNORE or any(x != IGNORE for x in d.get("labels", []))
               for d in pack["decisions"])
