"""P2 decision-instruction mixture (design doc 15 §5.2) and the evaluation sets (§5.4).

Rules: cap per dataset (20k packs), family quotas (tasks.QUOTAS), and inside a family
datasets are sampled with weight n^alpha (alpha = 0.5). A family that can't fill its quota is
reported, not topped up from other families, so the mixture stays what the design says.

The cap counts packs, not groups. Most P2 packs hold one decision (before A-multi); RAG packs
hold several, so RAG gets somewhat more groups than its 35% of packs. Reported in the data card.
"""

import random
from collections import Counter
from collections.abc import Callable

from .synthetic import format_packs
from .tasks import (FAMILIES, QUOTAS, SOURCES, Source, heldout_sources, load_source,
                    training_sources)
from .tiny_augment import heldout_view


def allocate(avail: dict[str, int], family_of: dict[str, str], total: int,
             quotas: dict[str, float] = QUOTAS, alpha: float = 0.5,
             cap: int = 20_000) -> tuple[dict[str, int], dict[str, int]]:
    """Packs to draw per source. Returns (allocation, shortfall per family)."""
    alloc: dict[str, int] = {}
    short: dict[str, int] = {}
    for fam, q in quotas.items():
        keys = [k for k in avail if family_of[k] == fam]
        budget = round(q * total)
        if not keys:
            if budget:
                short[fam] = budget
            continue
        left = {k: min(avail[k], cap) for k in keys}
        got = {k: 0 for k in keys}
        remaining = budget
        while remaining > 0:                      # water-fill: saturated sources drop out
            open_ = [k for k in keys if got[k] < left[k]]
            if not open_:
                break
            w = {k: max(left[k], 1) ** alpha for k in open_}
            wsum = sum(w.values())
            step = {k: min(left[k] - got[k], max(1, int(remaining * w[k] / wsum))) for k in open_}
            for k in sorted(open_, key=lambda k: -w[k]):
                take = min(step[k], remaining)
                got[k] += take
                remaining -= take
                if remaining == 0:
                    break
        alloc.update(got)
        if remaining > 0:
            short[fam] = remaining
    return alloc, short


Loader = Callable[[Source, str, int, int], list[dict]]


def build_p2(total: int, cap: int = 20_000, alpha: float = 0.5, seed: int = 0,
             release_only: bool = False, n_val: int = 200, loader: Loader = load_source,
             sources: list[Source] | None = None, log=print) -> dict:
    """{"train", "val_seen", "val_unseen", "test", "card"}.

    train       P2 packs (canonical templates; augmentation varies them each epoch)
    val_seen    per-source val packs, canonical templates (per-decision temperatures)
    val_unseen  the same packs under held-out templates and wordings (T_custom, eval every
                1k steps, §5.3 / §6)
    test        a disjoint half of each source's val split, canonical templates (in-domain test;
                evaluate.py adds its unseen-template view)
    card        per-source counts and licences for the data card (M2)
    """
    srcs = sources if sources is not None else training_sources(release_only)
    pools = {s.key: loader(s, "train", cap, seed) for s in srcs}
    avail = {k: len(v) for k, v in pools.items()}
    family_of = {s.key: s.family for s in srcs}
    if "format" in QUOTAS:                        # synthetic, generated to size
        avail["format"], family_of["format"] = cap, "format"
    alloc, short = allocate(avail, family_of, total, QUOTAS, alpha, cap)
    rng = random.Random(seed)
    train = []
    for k, n in alloc.items():
        if k == "format":
            texts = [p["state"]["header"] for v in pools.values() for p in v[:500]
                     if p["state"]["header"] and not p["state"]["segments"]]
            train += format_packs(n, seed, texts or None)
        else:
            train += rng.sample(pools[k], n)
    rng.shuffle(train)
    val_seen, val_unseen, test = [], [], []
    for s in srcs:
        try:
            v = loader(s, "val", 2 * n_val, seed)
        except KeyError:
            continue
        half = len(v) // 2                       # calib half / in-domain test half
        val_seen += v[:half]
        val_unseen += [heldout_view(p, q_idx=i % 2, o_idx=0) for i, p in enumerate(v[:half])]
        test += v[half:]
    if "format" in alloc:
        fv = format_packs(2 * n_val, seed + 1)
        for p in fv:
            p["id"] = p["id"].replace("format-", "format-val-")
        val_seen += fv[:n_val]
        val_unseen += [heldout_view(p, q_idx=i % 2, o_idx=0) for i, p in enumerate(fv[:n_val])]
        test += fv[n_val:]
    card = {"total": len(train), "requested": total, "cap": cap, "alpha": alpha,
            "release_only": release_only, "shortfall": short,
            "families": dict(Counter(p.get("meta", {}).get("family", "?") for p in train)),
            "sources": {k: {"available": avail[k], "used": alloc.get(k, 0),
                            "family": family_of[k],
                            "licence": SOURCES[k].licence if k in SOURCES else "synthetic",
                            "commercial": SOURCES[k].commercial if k in SOURCES else True}
                        for k in avail}}
    if short:
        log(f"warning: families short of their quota: {short}")
    return {"train": train, "val_seen": val_seen, "val_unseen": val_unseen, "test": test,
            "card": card}


def build_p1(sizes: dict, seed: int = 0) -> dict:
    """P1 parity: exactly Micro-Jev phase A (doc 14 §5.1), so Nano / Micro / Tiny compare at
    matched data (H6)."""
    from .builders import build_phase_a
    return build_phase_a(sizes, seed)


def build_heldout(n_per_source: int = 1000, seed: int = 0,
                  clusters=("H1", "H2", "H3", "H4", "H5"), loader: Loader = load_source,
                  all_templates: bool = False) -> dict[str, list[dict]]:
    """{cluster: packs}. Canonical template; all_templates=True adds the two held-out-template
    views too (robustness to wording on unseen tasks)."""
    out: dict[str, list[dict]] = {c: [] for c in clusters}
    for s in heldout_sources(clusters):
        packs = loader(s, "test", n_per_source, seed)
        out[s.family] += packs
        if all_templates:
            out[s.family] += [heldout_view(p, q, 0) for p in packs for q in (0, 1)]
    return out


def family_counts(packs: list[dict]) -> dict[str, int]:
    return dict(Counter(p.get("meta", {}).get("family", "?") for p in packs))


__all__ = ["allocate", "build_p1", "build_p2", "build_heldout", "family_counts", "FAMILIES"]
