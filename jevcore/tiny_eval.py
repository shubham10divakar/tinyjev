"""Tiny-Jev evaluation (design doc 15 §6 calibration, §7.2 metrics).

Groups of one decision name can have different option counts (multiple choice with 4 or 5
answers, which_passage over n passages, held-out tasks), so logits are stacked "ragged": padded
to the widest group with a large negative logit (finite, so temperature fitting stays NaN-free).

Calibration (§6):
  1. per built-in decision (every training task name): T fitted on its canonical val split;
  2. `custom`: one global T_custom fitted on the unseen-template val (all tasks pooled);
  3. held-out clusters and JevBench use T_custom (honest OOD calibration).
"""

import math
from collections import Counter

import numpy as np
import torch

from .calibration import fit_temperature, metrics

PAD_LOGIT = -1e4
CUSTOM = "custom"


def stack_ragged(groups: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    groups = [g for g in groups if g["label"] >= 0]
    K = max(len(g["logits"]) for g in groups)
    z = torch.full((len(groups), K), PAD_LOGIT)
    for i, g in enumerate(groups):
        z[i, : len(g["logits"])] = g["logits"].float()
    return z, torch.tensor([g["label"] for g in groups])


def by(scored: list[dict], key) -> dict:
    out: dict = {}
    for g in scored:
        if g["label"] >= 0:
            out.setdefault(key(g), []).append(g)
    return out


def nll_by_name(scored: list[dict], temps: dict | None = None) -> dict[str, float]:
    """Mean NLL per decision name and the macro mean ("macro"): the selection metric."""
    out = {}
    for name, gs in by(scored, lambda g: g["name"]).items():
        z, y = stack_ragged(gs)
        t = (temps or {}).get(name, (temps or {}).get(CUSTOM, 1.0))
        out[name] = float(torch.nn.functional.cross_entropy(z / t, y))
    out["macro"] = sum(out.values()) / max(len(out), 1)
    return out


def fit_temperatures(val_seen: list[dict], val_unseen: list[dict] | None = None) -> dict:
    """{decision name: T} from canonical val, plus {"custom": T_custom} from unseen templates."""
    temps = {name: fit_temperature(*stack_ragged(gs))
             for name, gs in by(val_seen, lambda g: g["name"]).items()}
    if val_unseen:
        temps[CUSTOM] = fit_temperature(*stack_ragged([g for g in val_unseen if g["label"] >= 0]))
    return temps


def temperature_for(name: str, temps: dict, ood: bool) -> float:
    return temps.get(CUSTOM, 1.0) if ood else temps.get(name, temps.get(CUSTOM, 1.0))


def report(scored: list[dict], temps: dict, ood: bool = False, key=lambda g: g["name"]) -> dict:
    """Metrics per key (decision name by default) with raw and calibrated numbers."""
    out = {}
    for k, gs in sorted(by(scored, key).items()):
        z, y = stack_ragged(gs)
        names = Counter(g["name"] for g in gs)
        t = temperature_for(names.most_common(1)[0][0], temps, ood)
        out[k] = {"temperature": t, "raw": metrics(z, y), "calibrated": metrics(z, y, t)}
    return out


def cluster_report(scored_by_cluster: dict[str, list[dict]], temps: dict) -> dict:
    """Held-out clusters with T_custom: per cluster accuracy / NLL / ECE / AURC and the
    macro average over clusters (§7.2)."""
    rows = {}
    for c, scored in scored_by_cluster.items():
        gs = [g for g in scored if g["label"] >= 0]
        if not gs:
            continue
        z, y = stack_ragged(gs)
        m = metrics(z, y, temps.get(CUSTOM, 1.0))
        rows[c] = {k: m[k] for k in ("n", "accuracy", "nll", "ece", "aurc", "macro_f1")}
        rows[c]["per_task"] = {n: metrics(*stack_ragged(v), temps.get(CUSTOM, 1.0))["accuracy"]
                               for n, v in by(gs, lambda g: g["name"]).items()}
    if rows:
        rows["macro"] = {k: float(np.mean([r[k] for r in rows.values()]))
                         for k in ("accuracy", "nll", "ece", "aurc")}
    return rows


def accuracy_by_option_count(scored: list[dict],
                             bins=((2, 2), (3, 4), (5, 10), (11, 40), (41, 100), (101, 255))) -> dict:
    """T5: accuracy vs number of options."""
    out = {}
    for lo, hi in bins:
        gs = [g for g in scored if g["label"] >= 0 and lo <= len(g["logits"]) <= hi]
        if gs:
            acc = float(np.mean([int(g["logits"].argmax()) == g["label"] for g in gs]))
            out[f"{lo}-{hi}"] = {"n": len(gs), "accuracy": acc}
    return out


def unpacked_view(packs: list[dict]) -> tuple[list[dict], dict]:
    """B4 / B-pair: one decision group per sequence (segment-scope: header + only that
    segment). Returns (parts, part id -> (pack id, decision index, segment))."""
    parts, origin = [], {}
    for p in packs:
        st = p["state"]
        for di, d in enumerate(p["decisions"]):
            if d["scope"] == "segment":
                labels = d.get("labels") or [-100] * len(d["targets"])
                for t, lab in zip(d["targets"], labels):
                    pid = f"{p['id']}#{di}@{t}"
                    parts.append({**p, "id": pid,
                                  "state": {"header": st["header"], "segments": [st["segments"][t]]},
                                  "decisions": [{**d, "targets": [0], "labels": [lab]}]})
                    origin[pid] = (p["id"], di, t)
            else:
                pid = f"{p['id']}#{di}"
                parts.append({**p, "id": pid, "decisions": [d]})
                origin[pid] = (p["id"], di, -1)
    return parts, origin


def score_unpacked(score_fn, packs: list[dict]) -> list[dict]:
    """Score with score_fn(parts) on the unpacked view, mapped back to the packs' groups."""
    parts, origin = unpacked_view(packs)
    out = []
    for g in score_fn(parts):
        pid, dec, seg = origin[g["pack"]]
        out.append({**g, "pack": pid, "dec": dec, "seg": seg})
    return out


def ci95_acc(correct: np.ndarray) -> tuple[float, float]:
    p, n = float(correct.mean()), len(correct)
    h = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))
    return p - h, p + h
