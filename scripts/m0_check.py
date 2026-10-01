"""M0 on the real backbone (design doc 15 §2, §7.4, §10). Needs network the first time.

    python scripts/m0_check.py                     # config facts + T-A..T-E, real Qwen3-0.6B, fp32
    python scripts/m0_check.py --dtype bf16        # same in bf16 on the GPU (threshold 1e-2)
    python scripts/m0_check.py --throughput        # GPU: tokens/s with LoRA + grad ckpt at 2048
    python scripts/m0_check.py --run runs/tiny-p2-s0 --dtype bf16   # probes on trained weights
    python scripts/m0_check.py --dry-run           # tiny random model, offline

The untrained head's last layer is zero (uniform output), so for the probes on fresh weights
the head and LoRA-B are randomised: the probes need outputs that differ between options.
Results go to results/m0.json.
"""

import argparse
import copy
import json
import statistics
import time

import torch
from common import ROOT, device, save_json, tiny_random

from jevcore.backbones.qwen3 import Session
from jevcore.invariance import max_diff, probe, probs_of
from jevcore.schema import decision, make_state, query_header

EXPECTED = {"num_hidden_layers": 28, "hidden_size": 1024, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
            "max_position_embeddings": 32768, "vocab_size": 151936, "tie_word_embeddings": True}

PASSAGES = [
    ("Penguin", "Penguins are a group of aquatic flightless birds. They live almost exclusively "
                "in the Southern Hemisphere, and only one species lives north of the equator."),
    ("Ostrich", "The common ostrich is a species of flightless bird native to large areas of "
                "Africa. It is the heaviest living bird and lays the largest eggs of any land animal."),
    ("Paris", "Paris is the capital and largest city of France, with an estimated population of "
              "two million residents in its administrative area."),
    ("Eiffel Tower", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
                     "Paris. It was built from 1887 to 1889 as the centrepiece of a world's fair."),
    ("Kiwi", "Kiwi are flightless birds endemic to New Zealand. They are about the size of a "
             "domestic chicken and are the smallest living ratites."),
]


def probe_pack(n_seg=3, pid="probe"):
    return {"id": pid, "source": "m0",
            "state": make_state(query_header("Which birds cannot fly?"), PASSAGES[:n_seg]),
            "decisions": [decision("relevance", targets=list(range(n_seg))),
                          decision("sufficient"),
                          decision("grounded", claim="Penguins cannot fly."),
                          {"name": "topic", "kind": "choice", "scope": "global",
                           "question": "What is the topic of the query?",
                           "options": ["animals", "geography", "history", "sports"]}]}


def randomise(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.head.parameters():
            if p.abs().sum() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        for n, p in model.bb.named_parameters():
            if "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator=g) * 0.02).to(p.dtype))


def session_checks(model, tok, M, cfg, dev) -> dict:
    ex = probe_pack(5)
    ref = probs_of(model, tok, M, [ex], cfg, dev, bf16=False)
    s = Session(model, tok, cfg, ex["state"], dev)
    as_probs = lambda gs: {(ex["id"], g["name"], g["seg"]):  # noqa: E731
                           dict(zip(g["options"], torch.softmax(g["logits"], 0).tolist())) for g in gs}
    out = {"T-C cache vs training path": max_diff(ref, as_probs(s.logits(ex["decisions"])))}
    out["T-E cache length == S after decide"] = s.cache_len() == s.S
    first = copy.deepcopy(ex["state"])
    first["segments"] = first["segments"][:2]
    s2 = Session(model, tok, cfg, first, dev)
    s2.extend(ex["state"]["segments"][2:])
    out["T-D extend vs fresh session"] = max_diff(as_probs(s.logits(ex["decisions"])),
                                                  as_probs(s2.logits(ex["decisions"])))
    out["T-E cache length == S after extend + decide"] = s2.cache_len() == s2.S
    return out


def config_facts(model, tok, M) -> dict:
    c = model.base_config
    found = {k: getattr(c, k, None) for k in EXPECTED}
    facts = {"found": found, "mismatch": {k: (EXPECTED[k], v) for k, v in found.items()
                                         if v != EXPECTED[k]},
             "layer_types": sorted(set(c.layer_types)), "sliding_window": getattr(c, "sliding_window", None),
             "use_sliding_window": getattr(c, "use_sliding_window", None),
             "tokenizer_len": len(tok), "embedding_rows": model.bb.get_input_embeddings().num_embeddings,
             "markers": M, "markers_inside_table": all(i < model.bb.get_input_embeddings().num_embeddings
                                                      for i in M.values()),
             "sink_id": model.sink_id, "sink_token": tok.convert_ids_to_tokens(model.sink_id),
             "pad_token": tok.pad_token, "params_total": sum(p.numel() for p in model.parameters()),
             "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return facts


def throughput(model, tok, M, dev, seq=2048, rows=4, iters=10) -> dict:
    from jevcore.collate import collate
    from jevcore.loss import packed_loss
    from jevcore.packing import pack_row, render
    from jevcore.scoring import autocast
    cfg = model.pack_config(seq)
    model.train()
    model.enable_grad_ckpt()
    long = probe_pack(5)
    long["state"]["segments"] = [{"title": t, "text": (x + " ") * 6} for t, x in PASSAGES] * 2
    long["decisions"][0]["targets"] = list(range(10))
    r = render(long, tok, M, cfg)
    b = collate([pack_row([r])] * rows, tok.pad_token_id, cfg)
    b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(iters + 3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast(dev, True):
            z = model(**b)
        loss, _ = packed_loss(z, b["labels"], b["names"])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t0)
    tok_per_step = int(b["valid"].sum())
    t = statistics.median(times)
    return {"row_tokens": len(r), "rows": rows, "tokens_per_microbatch": tok_per_step,
            "s_per_microbatch": t, "tokens_per_s": tok_per_step / t,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--run", help="trained run folder: probe trained weights")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--p2-tokens", type=float, help="tokens in one P2 epoch, for the hours estimate")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dev = "cpu" if a.dry_run else device()
    res = {"base": a.base, "dtype": a.dtype, "attn": a.attn, "device": dev}

    if a.dry_run:
        model, tok, M = tiny_random()
        res["note"] = "dry run: tiny random model and word-level tokenizer"
    elif a.run:
        from jevcore.backbones.qwen3 import TinyJev
        model, tok, M, _ = TinyJev.load(a.run, dev, dtype=a.dtype)
    else:
        from jevcore.backbones.qwen3 import TinyJev
        from common import load_config, model_cfg
        cfg = load_config()
        mc = {**model_cfg(cfg), "dtype": a.dtype, "attn": a.attn}
        model, tok, M = TinyJev.from_base(a.base, mc)
        randomise(model)
        model.to(dev).eval()
        res["config_facts"] = config_facts(model, tok, M)
    model.eval()

    if a.throughput:
        if dev != "cuda":
            raise SystemExit("--throughput needs the GPU")
        res["throughput"] = throughput(model, tok, M, dev)
        if a.p2_tokens:
            res["throughput"]["p2_epoch_hours"] = a.p2_tokens / res["throughput"]["tokens_per_s"] / 3600
    else:
        cfg = model.pack_config(4096 if not a.dry_run else 1024)
        tol = 1e-3 if a.dtype == "fp32" else 1e-2
        base, other = probe_pack(3), probe_pack(5, "other")
        if a.dry_run:     # the word-level test tokenizer can't read the English probe texts
            import sys
            sys.path.insert(0, str(ROOT / "tests"))
            from conftest import sample_example
            base, other = sample_example(), sample_example(n_seg=5, long=True) | {"id": "other"}
        diffs = probe(model, tok, M, base, other, cfg, dev, bf16=a.dtype == "bf16")
        res["T-A/T-B probes (max |dp|)"] = diffs
        if not a.dry_run:
            res.update(session_checks(model, tok, M, cfg, dev))
        res["threshold"] = tol
        res["pass"] = (all(v <= tol for v in diffs.values())
                       and all(v <= tol if isinstance(v, float) else v
                               for k, v in res.items() if k.startswith("T-")
                               and not isinstance(v, dict)))
    save_json(res, ROOT / "results" / ("dry/m0.json" if a.dry_run else "m0.json"))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
