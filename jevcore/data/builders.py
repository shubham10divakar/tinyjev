"""Build packed examples from public datasets (design §5.1).

Labelling follows Nano-Jev (`nanojev/data.py`) so the comparison is architecture-only:
    relevance: gold paragraph containing the answer -> directly answers (2),
               other gold -> partially relevant (1), distractor -> irrelevant (0)
    sufficient: all gold present -> yes (0); one gold removed -> no (1)
    grounded:   entailment / SUPPORTS -> yes (0); otherwise -> no (1)

Train packs carry every paragraph and mark the gold ones in meta.gold; augmentation (A-k) then
samples the distractors. Eval packs use a fixed composition:
    eval_distractors=None -> all paragraphs (k-sweep / long-context sets)
    eval_distractors=n    -> gold + n distractors (n=1 or 2 mirrors Nano's 3-4 passage states)

`to_nano_rows` turns packs into Nano-format rows, so Nano v1.0 is re-scored on identical
(query, passage) pairs (§5.1).
"""

import json
import random
from pathlib import Path

from ..schema import DECISIONS, decision, from_nano_row, make_state, query_header

HOTPOT = ("hotpotqa/hotpot_qa", "distractor")
SQUAD2 = ("rajpurkar/squad_v2", None)
MNLI = ("nyu-mll/multi_nli", None)
MUSIQUE = ("bdsaglam/musique", "musique_ans_v1.0_dev.jsonl")            # CC BY 4.0
MUSIQUE_TRAIN = ("bdsaglam/musique", "musique_ans_v1.0_train.jsonl")    # OOD calibration only
VITAMINC = ("tals/vitaminc", "test.jsonl")                              # CC BY-SA 3.0


def _load(*args, **kw):
    from datasets import load_dataset   # only needed when building data
    return load_dataset(*args, **kw)


def _rel_label(text: str, answers: list[str], is_gold: bool) -> int:
    if not is_gold:
        return 0
    return 2 if any(a and a not in ("yes", "no") and a in text.lower() for a in answers) else 1


def qa_packs(qid: str, source: str, query: str, paras: list[tuple[str, str]], gold: list[int],
             answers: list[str], rng: random.Random, n_distract: int | None = None,
             minus_relevance: bool | str = True) -> list[dict]:
    """Two packs per multi-hop question: "full" (all gold -> sufficient yes) and
    "minus-one-gold" (-> no). The full pack carries relevance on every segment. The minus pack
    carries it on every segment (True), none (False), or only on segments not in the full pack
    ("new"; for eval with n_distract=1 this gives Nano's 2 gold + 2 distractors per question,
    with no item counted twice)."""
    distract = [i for i in range(len(paras)) if i not in gold]
    if n_distract is not None:
        if len(distract) < n_distract + 1:
            return []
        distract = rng.sample(distract, n_distract + 1)   # +1 replaces the removed gold
        full_d, minus_d = distract[:n_distract], distract
    else:
        full_d = minus_d = distract
    removed = rng.choice(gold)
    out = []
    for kind, idxs, suff in (("full", list(gold) + full_d, 0),
                             ("minus", [g for g in gold if g != removed] + minus_d, 1)):
        rng.shuffle(idxs)
        segs = [paras[i] for i in idxs]
        gold_new = [k for k, i in enumerate(idxs) if i in gold]
        decs = []
        if kind == "full" or minus_relevance is True:
            targets = list(range(len(idxs)))
        elif minus_relevance == "new":
            targets = [k for k, i in enumerate(idxs) if i not in full_d and i not in gold]
        else:
            targets = []
        if targets:
            labels = [_rel_label(paras[idxs[k]][1], answers, idxs[k] in gold) for k in targets]
            decs.append(decision("relevance", targets=targets, labels=labels))
        decs.append(decision("sufficient", label=suff))
        out.append({"id": f"{source}-{qid}-{kind}", "source": source,
                    "state": make_state(query_header(query), segs), "decisions": decs,
                    "meta": {"gold": gold_new, "query": query, "answers": answers}})
    return out


# ------------------------------------------------------------------------------ HotpotQA

def _hotpot_paragraphs(row, max_sents: int = 3) -> dict[str, str]:
    """Title -> paragraph text, keeping the first sentences plus every supporting one
    (same as Nano)."""
    support: dict[str, set[int]] = {}
    for t, s in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        support.setdefault(t, set()).add(s)
    paras = {}
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        keep = [s for i, s in enumerate(sents) if i < max_sents or i in support.get(title, ())]
        paras[title] = " ".join(x.strip() for x in keep)
    return paras


def hotpot_packs(split: str, n: int, rng: random.Random, start: int = 0, seed: int = 0,
                 n_distract: int | None = None, minus_relevance: bool | str = True) -> list[dict]:
    """Questions [start, start + n) of the split after a fixed shuffle (same selection as Nano)."""
    ds = _load(*HOTPOT, split=split).shuffle(seed=seed).select(range(start, start + n))
    out = []
    for row in ds:
        paras = _hotpot_paragraphs(row)
        titles = list(paras)
        gold_t = [t for t in dict.fromkeys(row["supporting_facts"]["title"]) if t in paras]
        if len(gold_t) != 2 or len(titles) - 2 < 3:
            continue
        out += qa_packs(row["id"], "hotpot", row["question"],
                        [(t, paras[t]) for t in titles], [titles.index(t) for t in gold_t],
                        [row["answer"].strip().lower()], rng, n_distract, minus_relevance)
    return out


# ------------------------------------------------------------------------------ SQuAD 2.0

def squad2_packs(split: str, n: int, rng: random.Random, relevance: bool = False) -> list[dict]:
    """One segment per pack; balanced answerable / unanswerable. relevance=True adds the
    ablation labels (answerable -> directly answers, unanswerable -> partially relevant)."""
    ds = _load(SQUAD2[0], split=split).shuffle(seed=rng.randint(0, 10**6))
    pos, neg = [], []
    for row in ds:
        answerable = len(row["answers"]["text"]) > 0
        bucket = pos if answerable else neg
        if len(bucket) >= n // 2:
            if len(pos) >= n // 2 and len(neg) >= n // 2:
                break
            continue
        decs = [decision("sufficient", label=0 if answerable else 1)]
        if relevance:
            decs.insert(0, decision("relevance", targets=[0], labels=[2 if answerable else 1]))
        bucket.append({"id": f"squad2-{row['id']}", "source": "squad2",
                       "state": make_state(query_header(row["question"]),
                                           [(row["title"].replace("_", " "), row["context"])]),
                       "decisions": decs, "meta": {"query": row["question"]}})
    return pos + neg


# ------------------------------------------------------------------------------ NLI / claims

def grounded_pack(pid: str, source: str, evidence: str, claim: str, label: int) -> dict:
    return {"id": pid, "source": source, "state": make_state(evidence),
            "decisions": [decision("grounded", label=label, claim=claim)],
            "meta": {"claim": claim}}


def mnli_packs(split: str, n: int, rng: random.Random) -> list[dict]:
    ds = _load(MNLI[0], split=split).shuffle(seed=rng.randint(0, 10**6))
    pos, neg = [], []
    for i, row in enumerate(ds):
        if row["label"] not in (0, 1, 2):
            continue
        entailed = row["label"] == 0
        bucket = pos if entailed else neg
        if len(bucket) >= n // 2:
            if len(pos) >= n // 2 and len(neg) >= n // 2:
                break
            continue
        bucket.append(grounded_pack(f"mnli-{split}-{i}", "mnli", row["premise"],
                                    row["hypothesis"], 0 if entailed else 1))
    return pos + neg


def vitaminc_packs(n: int, rng: random.Random, seed: int = 0) -> list[dict]:
    """VitaminC test: SUPPORTS -> yes; REFUTES / NOT ENOUGH INFO -> no (balanced)."""
    ds = _load(VITAMINC[0], data_files={"test": VITAMINC[1]}, split="test").shuffle(seed=seed)
    pos, neg = [], []
    for i, row in enumerate(ds):
        supported = row["label"] == "SUPPORTS"
        bucket = pos if supported else neg
        if len(bucket) < n // 2:
            bucket.append(grounded_pack(f"vitaminc-{i}", "vitaminc", row["evidence"],
                                        row["claim"], 0 if supported else 1))
        if len(pos) >= n // 2 and len(neg) >= n // 2:
            break
    return pos + neg


# ------------------------------------------------------------------------------ MuSiQue

def musique_packs(n: int, rng: random.Random, seed: int = 0, train: bool = False,
                  n_distract: int | None = None, two_hop_sufficient_only: bool = True) -> list[dict]:
    """MuSiQue answerable (dev by default; train=True for the OOD calibration set).

    n_distract=None keeps all ~20 paragraphs (k-sweep, G6). Sufficiency is labelled on 2-hop
    questions only by default (as Nano), so 3/4-hop packs carry relevance only.
    """
    spec = MUSIQUE_TRAIN if train else MUSIQUE
    name = "train" if train else "dev"
    ds = _load(spec[0], data_files={name: spec[1]}, split=name)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        answers = [a.strip().lower() for a in [row["answer"], *row["answer_aliases"]] if a.strip()]
        paras = [(p["title"], p["paragraph_text"].strip()) for p in row["paragraphs"]]
        gold = [k for k, p in enumerate(row["paragraphs"]) if p["is_supporting"]]
        if len(gold) < 2 or len(paras) - len(gold) < 3:
            continue
        packs = qa_packs(row["id"], "musique", row["question"], paras, gold, answers, rng,
                         n_distract, minus_relevance="new" if n_distract is not None else False)
        if two_hop_sufficient_only and len(gold) != 2:
            for p in packs:
                p["decisions"] = [d for d in p["decisions"] if d["name"] != "sufficient"]
            packs = [p for p in packs if p["decisions"]]
        out += packs
    return out


# ------------------------------------------------------------------------------ phase B / C

def twowiki_packs(*_, **__):
    raise NotImplementedError("phase B: verify the 2WikiMultihopQA HF id and licence at M5")


def fever_packs(*_, **__):
    raise NotImplementedError("phase B: verify the FEVER HF id and licence before release (M5)")


def cyber_packs(path: str | Path) -> list[dict]:
    """Phase C: Cyber-Jev JSONL rows (one global decision each) -> packs (header = input)."""
    with open(path, encoding="utf-8") as f:
        return [from_nano_row(json.loads(line), i) for i, line in enumerate(f)]


# ------------------------------------------------------------------------------ assembly

def build_phase_a(sizes: dict, seed: int = 0, eval_distractors: int | None = 1,
                  squad_relevance: bool = False) -> dict[str, list[dict]]:
    """{"train", "calib", "test"} packs on Nano v1.0's source questions.

    Calib / test are the same validation halves as Nano (Hotpot split by question).
    """
    rng = random.Random(seed)
    train = (hotpot_packs("train", sizes["hotpot_train"], rng, seed=seed)
             + squad2_packs("train", sizes["squad_train"], rng, relevance=squad_relevance)
             + mnli_packs("train", sizes["mnli_train"], rng))
    n_hp = sizes["hotpot_eval"]
    calib = hotpot_packs("validation", n_hp, rng, start=0, seed=seed,
                         n_distract=eval_distractors, minus_relevance="new")
    test = hotpot_packs("validation", n_hp, rng, start=n_hp, seed=seed,
                        n_distract=eval_distractors, minus_relevance="new")
    for part in (squad2_packs("validation", 2 * sizes["squad_eval"], rng),
                 mnli_packs("validation_matched", 2 * sizes["mnli_eval"], rng)):
        rng.shuffle(part)
        calib += part[: len(part) // 2]
        test += part[len(part) // 2:]
    rng.shuffle(train)
    return {"train": train, "calib": calib, "test": test}


def build_heldout(sizes: dict, seed: int = 0, n_distract: int | None = 1) -> list[dict]:
    rng = random.Random(seed)
    return (musique_packs(sizes["musique"], rng, seed, n_distract=n_distract)
            + vitaminc_packs(sizes["vitaminc"], rng, seed))


# ------------------------------------------------------------------------------ views

NANO_QUESTIONS = {   # Nano-Jev's templates (nanojev/schema.py): the query is in the question
    "relevance": "How relevant is this passage to the query: {query}",
    "sufficient": "Does the context contain enough information to answer: {query}",
    "grounded": "Is this claim supported by the context? Claim: {claim}",
}


def _format_passages(segs) -> str:
    return "\n".join(f"[{i + 1}] {s['title']}: {s['text']}" for i, s in enumerate(segs))


def to_nano_rows(pack: dict) -> list[dict]:
    """Nano-format rows {decision, question, options, state, label, source, pack_id, seg}
    for every labelled group of a pack (builtin RAG decisions only)."""
    meta, segs = pack.get("meta", {}), pack["state"]["segments"]
    rows = []
    for d in pack["decisions"]:
        if d["name"] not in NANO_QUESTIONS:
            continue
        opts = list(DECISIONS[d["name"]].options)
        base = {"decision": d["name"], "options": opts, "source": pack["source"],
                "pack_id": pack["id"]}
        if d["name"] == "relevance":
            q = NANO_QUESTIONS["relevance"].format(query=meta["query"])
            for t, lab in zip(d["targets"], d.get("labels", [])):
                rows.append({**base, "question": q, "seg": t, "label": lab,
                             "state": f"{segs[t]['title']}: {segs[t]['text']}"})
        elif d["name"] == "sufficient":
            rows.append({**base, "question": NANO_QUESTIONS["sufficient"].format(query=meta["query"]),
                         "state": _format_passages(segs), "label": d["label"], "seg": -1})
        else:
            rows.append({**base, "question": NANO_QUESTIONS["grounded"].format(claim=meta["claim"]),
                         "state": pack["state"]["header"], "label": d["label"], "seg": -1})
    return rows


def unpack(pack: dict) -> list[dict]:
    """B-pair / "unpacked" view: one single-group example per labelled group, same text as
    the pack (relevance: header + only its own segment)."""
    out = []
    for d in pack["decisions"]:
        if d["scope"] == "segment":
            for t, lab in zip(d["targets"], d.get("labels") or [-100] * len(d["targets"])):
                seg = pack["state"]["segments"][t]
                out.append({**pack, "id": f"{pack['id']}#{d['name']}@{t}",
                            "state": {"header": pack["state"]["header"], "segments": [seg]},
                            "decisions": [{**d, "targets": [0], "labels": [lab]}]})
        else:
            out.append({**pack, "id": f"{pack['id']}#{d['name']}", "decisions": [d]})
    return out


def chunk_packs(pack: dict, k: int, rng: random.Random) -> list[dict]:
    """k-sweep (H2 / B-k1): split a pack's segments into packs of <= k segments, relevance only.

    k=1 reads each chunk alone; k >= len(segments) keeps the whole pack.
    """
    rel = [d for d in pack["decisions"] if d["name"] == "relevance"]
    if not rel:
        return []
    rel = rel[0]
    lab = dict(zip(rel["targets"], rel.get("labels") or [-100] * len(rel["targets"])))
    idx = list(range(len(pack["state"]["segments"])))
    rng.shuffle(idx)
    out = []
    for c in range(0, len(idx), k):
        part = idx[c: c + k]
        segs = [pack["state"]["segments"][i] for i in part]
        out.append({**pack, "id": f"{pack['id']}#k{k}.{c // k}",
                    "state": {"header": pack["state"]["header"], "segments": segs},
                    "decisions": [{**rel, "targets": list(range(len(part))),
                                   "labels": [lab.get(i, -100) for i in part]}],
                    "meta": {**pack.get("meta", {}), "orig_seg": part}})
    return out
