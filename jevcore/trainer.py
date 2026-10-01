"""Training loop (design §5.3–5.5): token-budget micro-batches, gradient accumulation to
~packs_per_step packs per optimizer step, bf16 autocast, best epoch by macro dev NLL."""

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from .collate import epoch_batches, to_device
from .data.augment import AugConfig, augment
from .loss import packed_loss
from .packing import PackConfig
from .report import split_by_name
from .scoring import autocast, score_packs


@dataclass
class TrainConfig:
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    wd: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    warmup: float = 0.06
    epochs: int = 3
    tokens_per_microbatch: int = 16384
    packs_per_step: int = 32
    bf16: bool = True
    grad_ckpt: bool = True
    seed: int = 0
    clip: float = 1.0
    log_every: int = 50
    loss_weights: dict = field(default_factory=dict)
    max_steps: int | None = None          # stop early (smoke runs)
    constant_lr: bool = False             # overfit runs

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "betas" in known:
            known["betas"] = tuple(known["betas"])
        return cls(**known)


def param_groups(model, tc: TrainConfig):
    """Backbone at lr_backbone (no decay on norms / biases / embeddings); head and the marker
    delta at lr_head (no decay on biases / delta)."""
    groups = {"bb_decay": [], "bb_plain": [], "head_decay": [], "head_plain": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("enc."):
            plain = p.ndim < 2 or "norm" in n or "embeddings" in n
            groups["bb_plain" if plain else "bb_decay"].append(p)
        else:
            plain = p.ndim < 2 or n == "marker_delta"
            groups["head_plain" if plain else "head_decay"].append(p)
    lr = {"bb": tc.lr_backbone, "head": tc.lr_head}
    return [{"params": ps, "lr": lr[k.split("_")[0]], "weight_decay": tc.wd if k.endswith("decay") else 0.0,
             "name": k} for k, ps in groups.items() if ps]


def dev_nll(model, tok, M, dev: list[dict], cfg: PackConfig, device, tc: TrainConfig) -> dict:
    """Mean NLL per decision name and their macro mean (the model-selection metric)."""
    scored = score_packs(model, tok, M, dev, cfg, device, tc.tokens_per_microbatch, bf16=tc.bf16)
    out = {}
    for name, (z, y) in split_by_name(scored).items():
        out[name] = float(torch.nn.functional.cross_entropy(z, y))
    out["macro"] = sum(out.values()) / max(len(out), 1)
    return out


def estimate_steps(train: list[dict], tc: TrainConfig) -> int:
    return max(1, math.ceil(len(train) / tc.packs_per_step)) * tc.epochs


def train(model, tok, M, train_packs: list[dict], dev_packs: list[dict], cfg: PackConfig,
          tc: TrainConfig, out: str | Path | None, device, aug: AugConfig | None = AugConfig(),
          row_packing: bool = True, extra_save: dict | None = None, log=print) -> dict:
    """Train; save the best epoch (by macro dev NLL) to `out`. Returns the history."""
    rng = random.Random(tc.seed)
    torch.manual_seed(tc.seed)
    model.to(device)
    if getattr(model, "marker_delta", None) is None:
        model.add_marker_delta(M)
    if tc.grad_ckpt:
        model.enc.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(param_groups(model, tc), betas=tc.betas, eps=tc.eps)
    total = tc.max_steps or estimate_steps(train_packs, tc)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0) if tc.constant_lr else
             get_linear_schedule_with_warmup(opt, int(tc.warmup * total), total))
    with_spans = model.cfg.get("readout") == "mean"
    augment_fn = (lambda ex, r: augment(ex, r, aug)) if aug else None
    out = Path(out) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train_log.jsonl", "a", encoding="utf-8") if out else None

    def emit(rec):
        log(json.dumps(rec))
        if log_f:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    history = {"dev": [], "train_loss": []}
    if dev_packs:
        d0 = dev_nll(model, tok, M, dev_packs, cfg, device, tc)
        emit({"event": "init", "dev_nll": d0, "train_packs": len(train_packs), "steps": total})
        history["dev"].append(d0)
    best, step, t0 = float("inf"), 0, time.time()
    for epoch in range(tc.epochs):
        model.train()
        acc_packs, running, n_run = 0, 0.0, 0
        parts_run: dict[str, float] = {}
        for b in epoch_batches(train_packs, tok, M, cfg, rng, tc.tokens_per_microbatch,
                               augment_fn, row_packing=row_packing, with_spans=with_spans):
            n_packs = len(set(b["g_example"]))
            b = to_device(b, device)
            with autocast(device, tc.bf16):
                z = model(**b)
            loss, parts = packed_loss(z, b["labels"], b["names"], tc.loss_weights)
            (loss * n_packs / tc.packs_per_step).backward()
            acc_packs += n_packs
            running += float(loss.detach())
            n_run += 1
            for k, v in parts.items():
                parts_run[k] = parts_run.get(k, 0.0) + v
            if acc_packs < tc.packs_per_step:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            acc_packs, step = 0, step + 1
            if step % tc.log_every == 0:
                emit({"event": "step", "epoch": epoch + 1, "step": step, "loss": running / n_run,
                      "per_decision": {k: v / n_run for k, v in parts_run.items()},
                      "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)})
                history["train_loss"].append(running / n_run)
                running, n_run, parts_run = 0.0, 0, {}
            if tc.max_steps and step >= tc.max_steps:
                break
        if acc_packs:        # flush the last partial accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if n_run:
            history["train_loss"].append(running / n_run)
        rec = {"event": "epoch", "epoch": epoch + 1, "step": step,
               "elapsed_s": round(time.time() - t0)}
        if dev_packs:
            d = dev_nll(model, tok, M, dev_packs, cfg, device, tc)
            rec["dev_nll"] = d
            history["dev"].append(d)
            if d["macro"] < best and out:
                best = d["macro"]
                model.save(out, tok, {**(extra_save or {}), "epochs_trained": epoch + 1,
                                      "dev_nll": d})
                rec["saved"] = str(out)
        emit(rec)
        if tc.max_steps and step >= tc.max_steps:
            break
    if out and not dev_packs:
        model.save(out, tok, {**(extra_save or {}), "epochs_trained": tc.epochs})
    if log_f:
        log_f.close()
    return history
