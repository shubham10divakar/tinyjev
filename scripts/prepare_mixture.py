"""Build Tiny-Jev data (design doc 15 §5). Needs network (Hugging Face datasets).

    python scripts/prepare_mixture.py --check                         # M1: every source loads + converts
    python scripts/prepare_mixture.py --phase P1 --out data/p1        # Micro-Jev phase A (parity)
    python scripts/prepare_mixture.py --phase P2 --total 200000 --out data/p2
    python scripts/prepare_mixture.py --phase P2 --release-only --out data/p2_release
    python scripts/prepare_mixture.py --heldout --out data/heldout    # H1..H5
    python scripts/prepare_mixture.py --dry-run --out <tmp>           # offline, fake rows

Writes JSONL packs plus card.json (per-source counts, licences, families, token lengths
with --tokenizer).
"""

import argparse
import json
import statistics
import traceback

from common import ROOT, save_json, write_jsonl  # noqa: F401

from jevcore.data import tasks as T
from jevcore.data.mixture import build_heldout, build_p1, build_p2, family_counts
from jevcore.schema import validate

P1_SIZES = {"hotpot_train": 20000, "squad_train": 10000, "mnli_train": 10000,
            "hotpot_eval": 500, "squad_eval": 500, "mnli_eval": 500}   # = Micro-Jev phase A


def check_sources(n: int = 3) -> int:
    """Load a few rows of every source / split and run its converter (M1)."""
    bad = 0
    for s in T.SOURCES.values():
        for split in s.splits:
            try:
                packs = T.load_source(s, split, n)
                for p in packs:
                    validate(p)
                d = packs[0]["decisions"][0] if packs else None
                print(f"ok   {s.key:16s} {split:5s} {len(packs)} packs  "
                      f"{(d['name'], d['options'][:4]) if d else 'EMPTY'}")
                bad += not packs
            except Exception as e:      # report every failing source, don't stop at the first
                bad += 1
                print(f"FAIL {s.key:16s} {split:5s} {type(e).__name__}: {e}")
                traceback.print_exc(limit=1)
    print(f"{bad} problems")
    return bad


def token_stats(packs, tokenizer: str) -> dict:
    from transformers import AutoTokenizer

    from jevcore.backbones.qwen3 import TINY_MARKERS, sink_id
    from jevcore.packing import PackConfig, add_markers, render
    tok = AutoTokenizer.from_pretrained(tokenizer)
    M = add_markers(tok, TINY_MARKERS)
    cfg = PackConfig(max_len=10**6, causal=True, sink_id=sink_id(tok))
    lens = sorted(len(render(p, tok, M, cfg)) for p in packs[:5000])
    q = lambda f: lens[min(len(lens) - 1, int(f * len(lens)))]  # noqa: E731
    return {"n": len(lens), "mean": statistics.mean(lens), "p50": q(0.5), "p90": q(0.9),
            "p99": q(0.99), "max": lens[-1], "over_2048": sum(x > 2048 for x in lens)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--phase", choices=["P1", "P2"])
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--out", default="data/p2")
    ap.add_argument("--total", type=int, default=200_000)
    ap.add_argument("--cap", type=int, default=20_000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--release-only", action="store_true")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-heldout", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", help="e.g. Qwen/Qwen3-0.6B: add token-length stats to the card")
    ap.add_argument("--dry-run", action="store_true", help="fake rows from tests, no network")
    a = ap.parse_args(argv)

    if a.check:
        raise SystemExit(1 if check_sources() else 0)
    out = a.out
    loader = T.load_source
    if a.dry_run:
        import sys
        sys.path.insert(0, str(ROOT / "tests"))
        from test_tiny_data import _fake_loader
        loader = _fake_loader

    if a.phase == "P1":
        d = build_p1(P1_SIZES, a.seed)
        for k, v in d.items():
            write_jsonl(v, f"{out}/{k}.jsonl")
        save_json({"phase": "P1", "sizes": P1_SIZES, **{k: len(v) for k, v in d.items()}},
                  f"{out}/card.json")
    elif a.phase == "P2":
        d = build_p2(a.total, a.cap, a.alpha, a.seed, a.release_only, a.n_val, loader)
        for k in ("train", "val_seen", "val_unseen", "test"):
            write_jsonl(d[k], f"{out}/{k}.jsonl")
        card = d["card"]
        if a.tokenizer:
            card["tokens"] = token_stats(d["train"], a.tokenizer)
        save_json(card, f"{out}/card.json")
        print(json.dumps({k: card[k] for k in ("total", "families", "shortfall")}, indent=2))
    if a.heldout:
        h = build_heldout(a.n_heldout, a.seed, loader=loader)
        for c, packs in h.items():
            write_jsonl(packs, f"{out}/{c}.jsonl")
        save_json({c: {"n": len(v), "families": family_counts(v)} for c, v in h.items()},
                  f"{out}/heldout_card.json")
        print({c: len(v) for c, v in h.items()})
    if not (a.phase or a.heldout):
        ap.print_help()


if __name__ == "__main__":
    main()
