"""Evaluate a Tiny-Jev run or a letter-readout baseline (design doc 15 §6 calibration, §7).

    python scripts/evaluate.py --run runs/tiny-p2-s0 --data data/p2 --heldout data/heldout \
        --jevbench jevbench/items.jsonl
    python scripts/evaluate.py --run runs/b4-p1 --data data/p1 --unpacked               # B4
    python scripts/evaluate.py --lm Qwen/Qwen3-0.6B --name b0 --data data/p2 --heldout data/heldout   # B0
    python scripts/evaluate.py --lm Qwen/Qwen3-4B-Instruct-2507 --name b1 ...                        # B1
    python scripts/evaluate.py --b5 runs/b5-p2 --data data/p2 --heldout data/heldout                 # B5
    python scripts/evaluate.py --dry-run

Reads from --data: val_seen.jsonl (or calib.jsonl) -> per-decision temperatures,
val_unseen.jsonl -> T_custom, test.jsonl -> in-domain test, and from --heldout: H1..H5.jsonl.
Writes calibration.json into the run folder (Tiny-Jev runs), and report.json / report.md /
predictions_*.jsonl into --results (default results/<name>).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from common import ROOT, device, dry_run_packs, read_dir, save_json, tiny_random, write_jsonl

from jevcore.data.tiny_augment import heldout_view
from jevcore.report import render_table
from jevcore.tiny_eval import (CUSTOM, accuracy_by_option_count, cluster_report,
                               fit_temperatures, report, temperature_for)

CLUSTERS = ("H1", "H2", "H3", "H4", "H5")


def _predictions(scored, temps, ood):
    rows = []
    for g in scored:
        t = temperature_for(g["name"], temps, ood)
        p = torch.softmax(g["logits"].float() / t, 0).tolist()
        rows.append({"pack": g["pack"], "name": g["name"], "dec": g["dec"], "seg": g["seg"],
                     "options": g["options"], "probs": p, "label": g["label"]})
    return rows


def _md(rep: dict) -> str:
    out = [f"# {rep['name']}", ""]
    for k in ("in_domain", "unseen_templates"):
        if rep.get(k):
            out += [render_table(k.replace("_", " "), rep[k]), ""]
    if rep.get("clusters"):
        out += ["## held-out clusters (T_custom)", "",
                "| cluster | n | acc | NLL | ECE | AURC |", "|---|---|---|---|---|---|"]
        for c, r in rep["clusters"].items():
            out.append(f"| {c} | {r.get('n', '')} | {r['accuracy']:.3f} | {r['nll']:.3f} "
                       f"| {r['ece']:.3f} | {r['aurc']:.3f} |")
        out.append("")
    if rep.get("by_option_count"):
        out += ["## accuracy vs number of options", "", "| options | n | acc |", "|---|---|---|"]
        out += [f"| {k} | {v['n']} | {v['accuracy']:.3f} |" for k, v in rep["by_option_count"].items()]
    out += ["", f"temperatures: {json.dumps(rep['temperatures'])}"]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="Tiny-Jev run folder")
    ap.add_argument("--b5", help="B5 run folder (letter readout + LoRA)")
    ap.add_argument("--lm", help="zero-shot letter readout model id (B0 / B1)")
    ap.add_argument("--unpacked", action="store_true", help="score one group per sequence (B4)")
    ap.add_argument("--data")
    ap.add_argument("--heldout", help="folder with H1.jsonl .. H5.jsonl")
    ap.add_argument("--jevbench", help="frozen JevBench-mini items.jsonl (H6)")
    ap.add_argument("--allow-unfrozen", action="store_true", help="dev only: skip the freeze check")
    ap.add_argument("--name")
    ap.add_argument("--results")
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--n", type=int, default=0, help="cap packs per set (quick checks)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dev = "cpu" if a.dry_run else device()
    cap = (lambda xs: xs[: a.n]) if a.n else (lambda xs: xs)

    # ---------------------------------------------------------------- model -> score function
    if a.dry_run:
        model, tok, M = tiny_random()
        name = a.name or "dry-run"
        from jevcore.tiny_trainer import PackedFeeder
        f = PackedFeeder(tok, M, model.pack_config(1024), [], None, 8192, unpacked=a.unpacked)
        score = lambda ps: f.score(model, ps, dev, bf16=False)  # noqa: E731
    elif a.run:
        from jevcore.backbones.qwen3 import TinyJev
        from jevcore.tiny_trainer import PackedFeeder
        model, tok, M, saved = TinyJev.load(a.run, dev, merge=True,
                                            dtype="bf16" if dev == "cuda" else "fp32")
        name = a.name or Path(a.run).name
        max_len = a.max_len or saved.get("max_len_eval", 4096)
        f = PackedFeeder(tok, M, model.pack_config(max_len), [], None, 16384, unpacked=a.unpacked)
        score = lambda ps: f.score(model, ps, dev)  # noqa: E731
    else:
        from jevcore.lm_readout import LetterReadout, score_prompts
        if a.b5:
            model = LetterReadout.load(a.b5, dev)
            name = a.name or Path(a.b5).name
        elif a.lm:
            model = LetterReadout.from_base(a.lm).to(dev)
            name = a.name or a.lm.split("/")[-1]
        else:
            raise SystemExit("one of --run / --b5 / --lm / --dry-run")
        score = lambda ps: score_prompts(model, ps, dev)  # noqa: E731

    # ---------------------------------------------------------------- sets
    if a.dry_run:
        packs = dry_run_packs(12)
        sets = {"val_seen": packs[:8], "val_unseen": [heldout_view(p, 0, 0) for p in packs[:8]],
                "test": packs[8:], "H1": packs[:4], "H2": packs[4:8]}
    else:
        sets = {"val_seen": read_dir(a.data, "val_seen") or read_dir(a.data, "calib"),
                "val_unseen": read_dir(a.data, "val_unseen"), "test": read_dir(a.data, "test")} if a.data else {}
        for c in CLUSTERS:
            if a.heldout and (Path(a.heldout) / f"{c}.jsonl").exists():
                sets[c] = read_dir(a.heldout, c)
        if a.jevbench:
            from jevcore.data import jevbench
            sets["H6"] = ([jevbench.to_pack(i) for i in jevbench.read(a.jevbench)]
                          if a.allow_unfrozen else jevbench.load_frozen(a.jevbench))
    sets = {k: cap(v) for k, v in sets.items() if v}
    if sets.get("test"):
        sets["test_unseen"] = [heldout_view(p, i % 2, 0) for i, p in enumerate(sets["test"])]

    results = Path(a.results or ROOT / "results" / ("dry" if a.dry_run else "") / name)
    t0 = time.time()
    scored = {}
    for k, ps in sets.items():
        s0 = time.time()
        scored[k] = score(ps)
        print(f"{k}: {len(ps)} packs, {len(scored[k])} groups, {time.time() - s0:.1f}s", flush=True)

    temps = fit_temperatures(scored.get("val_seen", []), scored.get("val_unseen"))
    if CUSTOM not in temps:
        temps[CUSTOM] = 1.0
    if a.run and not a.dry_run:
        save_json(temps, Path(a.run) / "calibration.json")

    rep = {"name": name, "temperatures": temps, "unpacked": a.unpacked,
           "sets": {k: len(v) for k, v in sets.items()}}
    if scored.get("test"):
        rep["in_domain"] = report(scored["test"], temps)
        rep["unseen_templates"] = report(scored["test_unseen"], temps, ood=True)
    clusters = {k: scored[k] for k in (*CLUSTERS, "H6") if k in scored}
    if clusters:
        rep["clusters"] = cluster_report(clusters, temps)
    every = [g for k, v in scored.items() if k not in ("val_seen", "val_unseen") for g in v]
    rep["by_option_count"] = accuracy_by_option_count(every)
    rep["wall_s"] = round(time.time() - t0, 1)

    results.mkdir(parents=True, exist_ok=True)
    save_json(rep, results / "report.json")
    (results / "report.md").write_text(_md(rep), encoding="utf-8")
    for k, v in scored.items():
        ood = k not in ("val_seen", "test")
        write_jsonl(_predictions(v, temps, ood), results / f"predictions_{k}.jsonl")
    print(_md(rep))
    return rep


if __name__ == "__main__":
    main()
