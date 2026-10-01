"""Invariance probes (design §6.4), shared by tests/test_mask_invariance.py and
scripts/invariance.py.

For a probe pack, a decision's probabilities are compared across:
    alone            each decision in its own pack
    reversed         decisions in reverse order
    options_permuted every decision's options shuffled (compared by option text)
    extra_decision   an unrelated decision inserted
    row_packed       the pack shares a row with another example
    batched_padded   the pack is batched next to a longer row (padding)
"""

import copy
import random

import torch

from .collate import collate, to_device
from .packing import PackConfig, pack_row, render
from .scoring import autocast

EXTRA = {"name": "extra", "kind": "choice", "scope": "global",
         "question": "Is the query time-sensitive?", "options": ["yes", "no", "unclear"]}


def probs_of(model, tok, M, examples, cfg: PackConfig, device="cpu", rows_of=None, bf16=False):
    """{(example id, decision name, seg): {option: p}} from one batch."""
    rendered = [render(e, tok, M, cfg) for e in examples]
    ids = [e["id"] for e in examples]
    groups = rows_of or [[i] for i in range(len(examples))]
    rows = [pack_row([rendered[i] for i in g], [ids[i] for i in g]) for g in groups]
    b = collate(rows, tok.pad_token_id, cfg, model.cfg.get("readout") == "mean")
    with torch.no_grad(), autocast(device, bf16):
        z = model(**to_device(b, device))
    p = torch.softmax(z.float(), -1).cpu()
    by_id = {e["id"]: e for e in examples}
    out = {}
    for n in range(len(b["names"])):
        ex = by_id[b["g_example"][n]]
        dec = ex["decisions"][b["g_dec"][n]]
        out[(ex["id"], dec["name"], b["g_seg"][n])] = dict(zip(dec["options"], p[n].tolist()))
    return out


def max_diff(a: dict, b: dict) -> float:
    keys = a.keys() & b.keys()
    if not keys:
        raise ValueError("no shared groups")
    return max(abs(a[k][o] - b[k][o]) for k in keys for o in a[k])


def perturbations(base: dict, seed: int = 0) -> dict:
    rng = random.Random(seed)
    alone = []
    for d in base["decisions"]:
        e = copy.deepcopy(base)
        e["decisions"] = [copy.deepcopy(d)]
        alone.append(e)
    rev = copy.deepcopy(base)
    rev["decisions"] = rev["decisions"][::-1]
    perm = copy.deepcopy(base)
    for d in perm["decisions"]:
        order = list(range(len(d["options"])))
        rng.shuffle(order)
        d["options"] = [d["options"][i] for i in order]
        d.pop("label", None)
        d.pop("labels", None)
    extra = copy.deepcopy(base)
    extra["decisions"].insert(min(1, len(extra["decisions"])), dict(EXTRA))
    return {"alone": alone, "reversed": rev, "options_permuted": perm, "extra_decision": extra}


def probe(model, tok, M, base: dict, other: dict, cfg: PackConfig, device="cpu",
          bf16=False) -> dict[str, float]:
    """max |Δp| per perturbation for one probe pack. `other` is a different (longer) pack used
    for the row-packed / padded cases; it must have a different id."""
    f = lambda exs, **kw: probs_of(model, tok, M, exs, cfg, device, bf16=bf16, **kw)  # noqa: E731
    ref = f([base])
    v = perturbations(base)
    alone = {}
    for e in v["alone"]:
        alone.update(f([e]))
    out = {"alone": max_diff(ref, alone)}
    for name in ("reversed", "options_permuted", "extra_decision"):
        out[name] = max_diff(ref, f([v[name]]))
    out["row_packed"] = max_diff(ref, f([other, base], rows_of=[[0, 1]]))
    out["batched_padded"] = max_diff(ref, f([base, other]))
    return out


def fits_untruncated(pack: dict, tok, M, cfg: PackConfig) -> bool:
    """Probe packs must not trigger truncation (it depends on the decision set)."""
    try:
        with_extra = copy.deepcopy(pack)
        with_extra["decisions"].append(dict(EXTRA))
        return not render(with_extra, tok, M, cfg).truncated
    except ValueError:
        return False
