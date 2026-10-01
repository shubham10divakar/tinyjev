"""Train Tiny-Jev (design doc 15 §6), or the B4 / B5 baselines with the same recipe.

    python scripts/train.py --data data/p1 --out runs/tiny-p1-s0 --seed 0          # M1 (P1 parity)
    python scripts/train.py --data data/p2 --out runs/tiny-p2-s0                     # M3 (P2)
    python scripts/train.py --data data/p2 --out runs/tiny-p2-A4 --ablation T-A4     # M4
    python scripts/train.py --data data/p1 --out runs/b4-p1 --baseline B4            # unpacked
    python scripts/train.py --data data/p2 --out runs/b5-p2 --baseline B5            # letter readout
    python scripts/train.py --data data/p2 --out runs/overfit --overfit 200          # M2-style check
    python scripts/train.py --dry-run                                                # offline smoke test

--data DIR holds train.jsonl and a validation file: val_unseen.jsonl (P2; held-out templates)
or calib.jsonl (P1). --drop-family X removes one family from training (T-A8).
"""

import argparse
import json
import time

from common import (ABLATIONS, build_tiny, device, dry_run_packs, load_config, read_dir,
                    save_json, tiny_random)

from jevcore.data.tiny_augment import TinyAugConfig
from jevcore.tiny_trainer import PackedFeeder, PromptFeeder, TinyTrainConfig, train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--data", help="folder with train.jsonl + val_unseen.jsonl / calib.jsonl")
    ap.add_argument("--out", default="runs/tiny-dev")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--baseline", choices=["B4", "B5"])
    ap.add_argument("--ablation", choices=sorted(ABLATIONS))
    ap.add_argument("--drop-family", help="T-A8: leave one family out of training")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--overfit", type=int, help="train on N packs, no augmentation, constant lr")
    ap.add_argument("--n-val", type=int, default=2000, help="cap on validation packs")
    ap.add_argument("--dry-run", action="store_true", help="tiny random model, offline packs, CPU")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.ablation)
    tc = TinyTrainConfig.from_dict(cfg["train"])
    if a.seed is not None:
        tc.seed = a.seed
    if a.max_steps:
        tc.max_steps = a.max_steps
    aug = TinyAugConfig.from_dict(cfg["data"].get("aug", {}))
    dev = device()

    if a.dry_run:
        dev = "cpu"
        packs = dry_run_packs()
        train_packs, val = packs, packs[:8]
        tc.tokens_per_microbatch, tc.tokens_per_step = 2048, 1024
        tc.grad_ckpt, tc.bf16, tc.max_steps, tc.eval_every, tc.log_every = False, False, tc.max_steps or 6, 3, 2
        tc.epochs = 100
        max_len = 1024
    else:
        if not a.data:
            raise SystemExit("--data is required (or --dry-run)")
        train_packs = read_dir(a.data, "train")
        val = (read_dir(a.data, "val_unseen") or read_dir(a.data, "calib"))[: a.n_val]
        max_len = cfg["data"]["max_len"]
    if a.drop_family:
        n0 = len(train_packs)
        train_packs = [p for p in train_packs if p.get("meta", {}).get("family") != a.drop_family]
        print(f"T-A8: dropped family {a.drop_family}: {n0 - len(train_packs)} packs")
    if a.overfit:
        train_packs, val, aug = train_packs[: a.overfit], train_packs[: a.overfit], None
        tc.constant_lr, tc.epochs, tc.warmup = True, max(tc.epochs, 30), 0.0

    t0 = time.time()
    extra = {"version": "dev", "config": cfg, "baseline": a.baseline, "seed": tc.seed,
             "data": a.data, "max_len_eval": cfg["data"].get("max_len_eval", max_len)}
    if a.baseline == "B5":
        from jevcore.lm_readout import LetterReadout
        if a.dry_run:
            raise SystemExit("B5 dry run: see tests/test_tiny_trainer.py::test_b5_prompt_feeder_trains")
        model = LetterReadout.from_base(cfg["model"]["backbone"], lora=cfg["model"]["lora"])
        feeder = PromptFeeder(model, train_packs, aug, tc.tokens_per_microbatch)
        save = lambda out, ex: model.save(out, {**extra, **ex})  # noqa: E731
    else:
        if a.dry_run:
            model, tok, M = tiny_random({k: v for k, v in cfg["model"].items()
                                         if k in ("option_mode", "echo", "head", "sink_token")})
        else:
            model, tok, M = build_tiny(cfg)
        pcfg = model.pack_config(max_len)
        feeder = PackedFeeder(tok, M, pcfg, train_packs, aug, tc.tokens_per_microbatch,
                              unpacked=a.baseline == "B4", row_packing=cfg["data"].get("row_packing", True),
                              with_spans=model.cfg["readout"] == "mean")
        save = lambda out, ex: model.save(out, tok, {**extra, **ex})  # noqa: E731
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({"train_packs": len(train_packs), "val_packs": len(val), "device": dev,
                      "trainable_params": n_train, "baseline": a.baseline, "ablation": a.ablation}))
    hist = train(model, feeder, val, tc, a.out, dev, save)
    hist["wall_s"] = round(time.time() - t0)
    save_json(hist, f"{a.out}/history.json")
    print(json.dumps({"best_val_macro_nll": hist["best_val_macro_nll"], "steps": hist["steps"],
                      "wall_s": hist["wall_s"], "out": a.out}))
    return hist


if __name__ == "__main__":
    main()
