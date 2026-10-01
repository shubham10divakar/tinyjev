"""Cascade (design doc 15 §7.6): accept Tiny-Jev when max p >= τ, else escalate to the teacher
(B1, Qwen3-4B, or a larger LLM). Uses the predictions_*.jsonl files from evaluate.py.

    python scripts/cascade.py --student results/tiny-p2-s0 --teacher results/b1 \
        --val val_unseen --test H4 --test H6

τ is fitted on --val only (the smallest escalation rate that keeps >= --keep of the teacher's
accuracy there), then reported on each --test set. Because thresholds don't transfer across
tasks (JEV-as-a-Judge), each test set also gets its own "oracle" τ, clearly labelled.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from common import read_jsonl, save_json


def joined(student_dir, teacher_dir, name):
    key = lambda r: (r["pack"], r["name"], r["dec"], r["seg"])  # noqa: E731
    s = {key(r): r for r in read_jsonl(Path(student_dir) / f"predictions_{name}.jsonl") if r["label"] >= 0}
    t = {key(r): r for r in read_jsonl(Path(teacher_dir) / f"predictions_{name}.jsonl") if r["label"] >= 0}
    ks = sorted(s.keys() & t.keys())
    conf = np.array([max(s[k]["probs"]) for k in ks])
    s_ok = np.array([int(np.argmax(s[k]["probs"])) == s[k]["label"] for k in ks])
    t_ok = np.array([int(np.argmax(t[k]["probs"])) == t[k]["label"] for k in ks])
    return conf, s_ok, t_ok, len(s), len(t)


def curve(conf, s_ok, t_ok, n_points=101):
    out = []
    for tau in np.unique(np.concatenate([[0.0, 1.01], np.quantile(conf, np.linspace(0, 1, n_points))])):
        acc_mask = conf >= tau
        acc = float(np.where(acc_mask, s_ok, t_ok).mean())
        out.append({"tau": float(tau), "escalated": float(1 - acc_mask.mean()), "accuracy": acc})
    return out


def pick(points, teacher_acc, keep):
    ok = [p for p in points if p["accuracy"] >= keep * teacher_acc]
    return min(ok, key=lambda p: (p["escalated"], -p["accuracy"])) if ok else None


def at_tau(conf, s_ok, t_ok, tau):
    m = conf >= tau
    return {"tau": tau, "escalated": float(1 - m.mean()), "accuracy": float(np.where(m, s_ok, t_ok).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--val", default="val_unseen")
    ap.add_argument("--test", action="append", default=[])
    ap.add_argument("--keep", type=float, default=0.98)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    conf, s_ok, t_ok, *_ = joined(a.student, a.teacher, a.val)
    vpts = curve(conf, s_ok, t_ok)
    chosen = pick(vpts, t_ok.mean(), a.keep)
    res = {"keep": a.keep, "val": {"set": a.val, "n": len(conf), "student_acc": float(s_ok.mean()),
                                    "teacher_acc": float(t_ok.mean()), "chosen": chosen}, "test": {}}
    for name in a.test:
        c, s, t, ns, nt = joined(a.student, a.teacher, name)
        pts = curve(c, s, t)
        res["test"][name] = {
            "n": len(c), "unmatched": max(ns, nt) - len(c),
            "student_acc": float(s.mean()), "teacher_acc": float(t.mean()),
            "with_val_tau": at_tau(c, s, t, chosen["tau"]) if chosen else None,
            "oracle_tau_on_this_set": pick(pts, t.mean(), a.keep), "curve": pts}
    out = Path(a.out or Path(a.student) / f"cascade_vs_{Path(a.teacher).name}.json")
    save_json(res, out)
    print(json.dumps({k: v for k, v in res.items() if k != "test"}, indent=2))
    for name, r in res["test"].items():
        print(name, json.dumps({k: v for k, v in r.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
