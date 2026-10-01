"""Tiny-Jev augmentation (design doc 15 §5.3), on top of Micro-Jev's (data/augment.py).

| Id           | What                                                         | Rate                      |
|--------------|--------------------------------------------------------------|---------------------------|
| A-shuffle    | shuffle segment order (which_passage options re-derived)    | always                    |
| A-k          | gold + k ~ U{k_lo..k_hi} distractors                         | always (if meta.gold)     |
| A-subset     | keep each decision with p = subset_p (>= 1 kept)             | always                    |
| A-multi      | single-decision pack: add 1–2 copies with other templates    | multi_template_p          |
| A-qpara      | question = a random *train* template of the task             | always ("all_train")      |
| A-verb       | option wording = a random *train* wording (aligned)          | verb_p                    |
| A-subsample  | n > 4 options: keep m ~ U{2..n}, always with the gold one    | always (opt_subsample)    |
| A-optperm    | shuffle option order, remap labels                           | always                    |

`single_template=True` is ablation T-A4: canonical question and canonical options only (no
A-multi, A-qpara, A-verb). `heldout_view` builds the unseen-template val / test (§5.3).
"""

import copy
import random
from dataclasses import dataclass

from ..schema import IGNORE
from .augment import permute_options, select_segments
from .tasks import PASSAGE_FORMATS, passage_options, task_table


@dataclass
class TinyAugConfig:
    shuffle: bool = True
    k_distractors: tuple[int, int] | None = (0, 8)
    max_segments: int = 10
    subset_p: float = 0.8
    optperm: bool = True
    verb_p: float = 0.3
    qpara: str = "all_train_templates"      # or "canonical"
    opt_subsample: bool = True
    subsample_above: int = 4
    multi_template_p: float = 0.3
    single_template: bool = False           # T-A4

    @classmethod
    def from_dict(cls, d: dict) -> "TinyAugConfig":
        kd = d.get("k_distractors", (0, 8))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known["k_distractors"] = tuple(kd) if kd else None
        return cls(**known)


def _fill(template: str, d: dict) -> str:
    return template.format(**d.get("fields", {}))


def _which_options(d: dict, n_seg: int, fmt=PASSAGE_FORMATS["train"][0]) -> None:
    """Re-derive which_passage options / label for the current segment list."""
    d["options"] = passage_options(n_seg, fmt)
    d["label"] = d["answer_seg"] if d["answer_seg"] >= 0 else n_seg


def _shuffle_segments(ex: dict, rng: random.Random, cfg: TinyAugConfig) -> dict:
    segs = ex["state"]["segments"]
    gold = set(ex.get("meta", {}).get("gold", []))
    keep = list(range(len(segs)))
    if cfg.k_distractors and gold:
        distract = [i for i in keep if i not in gold]
        lo, hi = cfg.k_distractors
        k = min(rng.randint(lo, hi), len(distract), max(cfg.max_segments - len(gold), 0))
        keep = sorted(gold) + rng.sample(distract, k)
    if cfg.shuffle:
        rng.shuffle(keep)
    which = [d for d in ex["decisions"] if d["name"] == "which_passage"]
    new_idx = {old: new for new, old in enumerate(keep)}
    for d in which:
        if d["answer_seg"] >= 0 and d["answer_seg"] not in new_idx:
            ex["decisions"].remove(d)              # its answer was dropped: can't relabel
            continue
        d["answer_seg"] = new_idx.get(d["answer_seg"], -1)
    ex = select_segments(ex, keep)
    for d in ex["decisions"]:
        if d["name"] == "which_passage":
            _which_options(d, len(keep))
    return ex


def subsample_options(d: dict, rng: random.Random, above: int = 4) -> None:
    """Keep m ~ U{2..n} options including the gold one (global, labelled decisions)."""
    n = len(d["options"])
    if n <= above or d["scope"] != "global" or d.get("label", IGNORE) == IGNORE:
        return
    m = rng.randint(2, n)
    others = [i for i in range(n) if i != d["label"]]
    keep = sorted(rng.sample(others, m - 1) + [d["label"]])
    d["options"] = [d["options"][i] for i in keep]
    d["label"] = keep.index(d["label"])


def _multi(ex: dict, rng: random.Random, table: dict) -> None:
    d = ex["decisions"][0]
    t = table.get(d["name"])
    if not t or d["scope"] != "global" or len(t["questions"]["train"]) < 2:
        return
    for _ in range(rng.randint(1, 2)):
        ex["decisions"].append(copy.deepcopy(d))


def augment_tiny(example: dict, rng: random.Random, cfg: TinyAugConfig = TinyAugConfig()) -> dict:
    ex = copy.deepcopy(example)
    table = task_table()
    if ex["state"].get("segments") and not ex.get("meta", {}).get("no_shuffle"):
        ex = _shuffle_segments(ex, rng, cfg)

    decs = ex["decisions"]                                                    # A-subset
    ex["decisions"] = [d for d in decs if rng.random() < cfg.subset_p] or [rng.choice(decs)]
    if (not cfg.single_template and len(ex["decisions"]) == 1
            and rng.random() < cfg.multi_template_p):                         # A-multi
        _multi(ex, rng, table)

    used_q: set[str] = set()
    for d in ex["decisions"]:
        t = table.get(d["name"])
        if t and not cfg.single_template and cfg.qpara == "all_train_templates":   # A-qpara
            qs = [q for q in t["questions"]["train"] if q not in used_q] or t["questions"]["train"]
            d["question"] = _fill(rng.choice(qs), d)
            used_q.add(d["question"])
        if not cfg.single_template and rng.random() < cfg.verb_p:            # A-verb
            if d["name"] == "which_passage":
                _which_options(d, len(ex["state"]["segments"]),
                               rng.choice(PASSAGE_FORMATS["train"]))
            elif t and "options" in t and not d.get("data_options"):
                w = rng.choice(t["options"]["train"])
                if len(w) == len(d["options"]):
                    d["options"] = list(w)
        if cfg.opt_subsample and d["name"] != "which_passage":               # A-subsample
            subsample_options(d, rng, cfg.subsample_above)
        if cfg.optperm:                                                       # A-optperm
            order = list(range(len(d["options"])))
            rng.shuffle(order)
            permute_options(d, order)
    return ex


def heldout_view(example: dict, q_idx: int = 0, o_idx: int | None = 0) -> dict:
    """Unseen-template view: held-out question q_idx and held-out option wording o_idx
    (None keeps the canonical wording) for every decision whose task has templates."""
    ex = copy.deepcopy(example)
    table = task_table()
    for d in ex["decisions"]:
        t = table.get(d["name"])
        if not t:
            continue
        d["question"] = _fill(t["questions"]["heldout"][q_idx], d)
        if o_idx is None:
            continue
        if d["name"] == "which_passage":
            _which_options(d, len(ex["state"]["segments"]), PASSAGE_FORMATS["heldout"][o_idx])
        elif "options" in t and not d.get("data_options"):
            w = t["options"]["heldout"][o_idx]
            if len(w) == len(d["options"]):
                d["options"] = list(w)
    ex["id"] = f"{ex['id']}#unseen{q_idx}"
    return ex
