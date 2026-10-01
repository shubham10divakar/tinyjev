"""Latency (design doc 15 §7.5): batch 1, median of --n runs after --warmup.

S10: query + 10 passages; decisions = relevance x10 + sufficient.
    prefill+decide   new Session + decide (what a one-shot call costs)
    decide (warm)    decide on an existing Session (the cache's point, T6 / aims M-4)
    one-pass         training-path forward over the whole row, no cache
    unpacked (B4)    every group as its own sequence (11 sequences, batched)

Loop: decide (relevance x10 + sufficient) -> extend by 5 chunks -> decide (relevance x5 on
the new chunks + sufficient + a next-action question). Cached session vs re-encoding the full
state for both calls.

    python scripts/bench_latency.py --run runs/tiny-p2-s0
    python scripts/bench_latency.py --base Qwen/Qwen3-0.6B      # untrained weights: speed only
    python scripts/bench_latency.py --dry-run
"""

import argparse
import json
import statistics
import time

import torch
from common import ROOT, device, save_json, tiny_random

from jevcore.backbones.qwen3 import Session
from jevcore.collate import collate, to_device
from jevcore.packing import pack_row, render
from jevcore.schema import decision, make_state, query_header
from jevcore.scoring import autocast
from jevcore.tiny_eval import unpacked_view

WORDS = ("the river city was founded near the old bridge and grew into a market town with a "
         "busy port where traders sold salt grain wool and later machine parts ").split()

NEXT = {"name": "next_action", "kind": "choice", "scope": "global",
        "question": "What should the system do next?",
        "options": ["answer now", "retrieve more", "rewrite the query", "abstain"]}


def passage(i: int, n_words: int) -> dict:
    words = [WORDS[(i * 7 + k) % len(WORDS)] for k in range(n_words)]
    return {"title": f"Doc {i}", "text": " ".join(words) + "."}


def timeit(fn, n: int, warmup: int, cuda: bool) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return {"median_ms": statistics.median(ts), "p90_ms": ts[int(0.9 * (len(ts) - 1))]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--base", help="untrained Tiny-Jev on this backbone (speed only)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--words", type=int, default=100, help="words per passage")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dev = "cpu" if a.dry_run else device()
    cuda = dev == "cuda"
    if a.dry_run:
        model, tok, M = tiny_random()
        a.n, a.warmup, a.words, a.dtype = min(a.n, 5), min(a.warmup, 1), min(a.words, 12), "fp32"
        global WORDS
        WORDS = "penguins live in antarctica and cannot fly paris is the capital of france".split()
    elif a.run:
        from jevcore.backbones.qwen3 import TinyJev
        model, tok, M, _ = TinyJev.load(a.run, dev, merge=True, dtype=a.dtype)
    else:
        from jevcore.backbones.qwen3 import TinyJev
        model, tok, M = TinyJev.from_base(a.base or "Qwen/Qwen3-0.6B", {"dtype": a.dtype})
        model.merge_lora()
        model.to(dev)
    model.eval()
    cfg = model.pack_config(8192)
    k = a.k
    segs = [passage(i, a.words) for i in range(k + 5)]
    state = make_state(query_header("When was the river city founded and what did it trade?"), segs[:k])
    decs = [decision("relevance", targets=list(range(k))), decision("sufficient")]
    pack = {"id": "s10", "state": state, "decisions": decs}
    res = {"device": dev, "dtype": a.dtype, "k": k, "words_per_passage": a.words}

    with torch.no_grad():
        warm = Session(model, tok, cfg, state, dev)
        res["state_tokens"] = warm.S
        res["prefill+decide"] = timeit(lambda: Session(model, tok, cfg, state, dev).logits(decs),
                                       a.n, a.warmup, cuda)
        res["decide (warm cache)"] = timeit(lambda: warm.logits(decs), a.n, a.warmup, cuda)
        b = to_device(collate([pack_row([render(pack, tok, M, cfg)])], tok.pad_token_id, cfg), dev)

        def one_pass():
            with autocast(dev, a.dtype == "bf16"):
                model(**b)
        res["one-pass (no cache)"] = timeit(one_pass, a.n, a.warmup, cuda)
        parts, _ = unpacked_view([pack])
        ub = to_device(collate([pack_row([render(p, tok, M, cfg)]) for p in parts],
                               tok.pad_token_id, cfg), dev)

        def unpacked():
            with autocast(dev, a.dtype == "bf16"):
                model(**ub)
        res["unpacked B4 (batched)"] = timeit(unpacked, a.n, a.warmup, cuda)

        def loop_cached():
            s = Session(model, tok, cfg, state, dev)
            s.logits(decs)
            s.extend(segs[k:])
            s.logits([decision("relevance", targets=list(range(k, k + 5))), decision("sufficient"),
                      dict(NEXT)])

        def loop_recompute():
            Session(model, tok, cfg, state, dev).logits(decs)
            full = make_state(state["header"], segs)
            Session(model, tok, cfg, full, dev).logits(
                [decision("relevance", targets=list(range(k, k + 5))), decision("sufficient"),
                 dict(NEXT)])
        res["loop: cached session"] = timeit(loop_cached, a.n, a.warmup, cuda)
        res["loop: re-encode each call"] = timeit(loop_recompute, a.n, a.warmup, cuda)
    res["warm speed-up vs prefill+decide (aims M-4: >= 3x)"] = (
        res["prefill+decide"]["median_ms"] / res["decide (warm cache)"]["median_ms"])
    save_json(res, ROOT / "results" / ("dry/latency.json" if a.dry_run else "latency.json"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
