"""Training-time augmentation (design §5.2) and the paraphrase test set (§6.3).

| Id        | What                                             | Rate                 |
|-----------|--------------------------------------------------|----------------------|
| A-shuffle | shuffle segment order                            | always               |
| A-k       | keep gold + k ~ U{k_lo..k_hi} distractors        | always (if meta.gold)|
| A-subset  | keep each decision with p = subset_p (>= 1 kept) | always               |
| A-optperm | shuffle option order, remap labels               | always               |
| A-verb    | swap option wording (train table)                | verb_p               |
| A-qpara   | swap question template (train table)             | qpara_p              |
"""

import copy
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from ..schema import IGNORE

TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "templates.yaml"


@lru_cache(maxsize=None)
def templates() -> dict:
    return yaml.safe_load(TEMPLATES_PATH.read_text(encoding="utf-8"))


@dataclass
class AugConfig:
    shuffle: bool = True
    k_distractors: tuple[int, int] | None = (0, 8)
    max_segments: int = 10
    subset_p: float = 0.8
    optperm: bool = True
    verb_p: float = 0.3
    qpara_p: float = 0.3

    @classmethod
    def from_dict(cls, d: dict) -> "AugConfig":
        kd = d.get("k_distractors", (0, 8))
        return cls(shuffle=d.get("shuffle", True), k_distractors=tuple(kd) if kd else None,
                   subset_p=d.get("subset_p", 0.8), optperm=d.get("optperm", True),
                   verb_p=d.get("verb_p", 0.3), qpara_p=d.get("qpara_p", 0.3))


def select_segments(ex: dict, keep: list[int]) -> dict:
    """Keep segments `keep` (in that order); remap targets/labels; drop emptied decisions."""
    new_idx = {old: new for new, old in enumerate(keep)}
    ex["state"]["segments"] = [ex["state"]["segments"][i] for i in keep]
    meta = ex.setdefault("meta", {})
    if "gold" in meta:
        meta["gold"] = [new_idx[g] for g in meta["gold"] if g in new_idx]
    decs = []
    for d in ex["decisions"]:
        if d["scope"] == "segment":
            pairs = [(new_idx[t], lab) for t, lab in
                     zip(d["targets"], d.get("labels") or [IGNORE] * len(d["targets"]))
                     if t in new_idx]
            if not pairs:
                continue
            d["targets"] = [t for t, _ in pairs]
            if "labels" in d:
                d["labels"] = [lab for _, lab in pairs]
        decs.append(d)
    ex["decisions"] = decs
    return ex


def _set_label(d: dict, fn) -> None:
    if d["scope"] == "segment":
        if "labels" in d:
            d["labels"] = [fn(lab) if lab != IGNORE else lab for lab in d["labels"]]
    elif d.get("label", IGNORE) != IGNORE:
        d["label"] = fn(d["label"])


def reword(d: dict, question: str | None = None, options: list[str] | None = None) -> None:
    """Replace a decision's question template and/or option wording (aligned, same order)."""
    if question is not None:
        d["question"] = question.format(**d.get("fields", {}))
    if options is not None:
        assert len(options) == len(d["options"])
        d["options"] = list(options)


def permute_options(d: dict, order: list[int]) -> None:
    """New option j is old option order[j]; labels follow."""
    inv = {old: new for new, old in enumerate(order)}
    d["options"] = [d["options"][i] for i in order]
    _set_label(d, lambda lab: inv[lab])


def augment(example: dict, rng: random.Random, cfg: AugConfig = AugConfig()) -> dict:
    ex = copy.deepcopy(example)
    segs = ex["state"]["segments"]
    gold = set(ex.get("meta", {}).get("gold", []))

    # A-k: gold + k distractors (only for packs that mark their gold segments)
    keep = list(range(len(segs)))
    if cfg.k_distractors and gold:
        distract = [i for i in keep if i not in gold]
        lo, hi = cfg.k_distractors
        k = min(rng.randint(lo, hi), len(distract), max(cfg.max_segments - len(gold), 0))
        keep = sorted(gold) + rng.sample(distract, k)
    if cfg.shuffle:
        rng.shuffle(keep)
    ex = select_segments(ex, keep)

    # A-subset
    decs = ex["decisions"]
    kept = [d for d in decs if rng.random() < cfg.subset_p] or [rng.choice(decs)]
    ex["decisions"] = kept

    table = templates()
    for d in ex["decisions"]:
        t = table.get(d["name"])
        if t and rng.random() < cfg.verb_p:                                  # A-verb
            reword(d, options=rng.choice(t["options"]["train"]))
        if t and rng.random() < cfg.qpara_p:                                 # A-qpara
            reword(d, question=rng.choice(t["questions"]["train"]))
        if cfg.optperm:                                                      # A-optperm
            order = list(range(len(d["options"])))
            rng.shuffle(order)
            permute_options(d, order)
    return ex


def paraphrase(example: dict, q_idx: int | None = 0, o_idx: int | None = 0) -> dict:
    """Paraphrase-test variant (§6.3): held-out question template q_idx and held-out option
    wording o_idx for every templated decision (None keeps the training form)."""
    ex = copy.deepcopy(example)
    table = templates()
    for d in ex["decisions"]:
        t = table.get(d["name"])
        if not t:
            continue
        reword(d,
               question=t["questions"]["heldout"][q_idx] if q_idx is not None else None,
               options=t["options"]["heldout"][o_idx] if o_idx is not None else None)
    return ex
