"""Shared evaluation flow (from nanojev/report.py, extended): score packs, fit temperatures on
calib, report test metrics, render tables."""

import json
from pathlib import Path

import torch

from .calibration import fit_temperature, metrics
from .schema import DECISIONS

ORDINAL = {n for n, d in DECISIONS.items() if d.kind == "score"}


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_by_name(scored: list[dict]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """scored: [{name, logits (1-D), label, ...}] -> {name: (logits [N, K], labels [N])}.
    Unlabelled groups are dropped."""
    out: dict[str, list] = {}
    for s in scored:
        if s["label"] >= 0:
            out.setdefault(s["name"], []).append(s)
    return {n: (torch.stack([s["logits"] for s in v]), torch.tensor([s["label"] for s in v]))
            for n, v in out.items()}


def decision_report(calib, test, ms: float | None = None, name: str = "",
                    temperature: float | None = None) -> dict:
    """calib/test: (logits, labels). The temperature is fitted on calib unless given."""
    t = temperature if temperature is not None else fit_temperature(*calib)
    ordinal = name in ORDINAL
    r = {"temperature": t,
         "raw": metrics(*test, ordinal=ordinal),
         "calibrated": metrics(*test, temperature=t, ordinal=ordinal)}
    if ms is not None:
        r["ms_per_pack"] = ms
    return r


def render_table(title: str, report: dict) -> str:
    lines = [f"## {title}", "",
             "| decision | n | T | acc | macro-F1 | AUROC / QWK | NLL raw → cal "
             "| ECE raw → cal | AURC | acc @20% esc. |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for dec, r in report.items():
        raw, cal = r["raw"], r["calibrated"]
        extra = cal.get("auroc", cal.get("qwk"))
        lines.append(
            f"| {dec} | {raw['n']} | {r['temperature']:.2f} | {cal['accuracy']:.3f} "
            f"| {cal['macro_f1']:.3f} | {'' if extra is None else f'{extra:.3f}'} "
            f"| {raw['nll']:.3f} → {cal['nll']:.3f} | {raw['ece']:.3f} → {cal['ece']:.3f} "
            f"| {cal['aurc']:.3f} | {cal['acc_escalated']['20%']:.3f} |")
    return "\n".join(lines)


def save(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
