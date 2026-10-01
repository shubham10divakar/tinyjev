"""Tiny-Jev training loop (design doc 15 §6).

- LoRA 2e-4, head 1e-3, markers 1e-3; AdamW (0.9, 0.95); weight decay only on the head's
  matrices (0.01). Full fine-tune (T-A3) puts the backbone at lr_backbone.
- 3% warmup, cosine to 10% of the peak.
- Micro-batches of ~tokens_per_microbatch tokens, accumulated to ~tokens_per_step tokens per
  optimizer step (each micro-batch's loss weighted by its token share).
- bf16 autocast, fp32 head / LoRA / markers, gradient checkpointing.
- Evaluate macro NLL on the validation packs (the unseen-template val in P2) every
  `eval_every` steps and at each epoch end; keep the best.

Feeders adapt the loop to a model family: `PackedFeeder` (TinyJev; packed, or unpacked = B4)
and `PromptFeeder` (LetterReadout = B5).
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .collate import collate, make_rows, render_all, to_device, token_batches
from .data.tiny_augment import TinyAugConfig, augment_tiny
from .loss import packed_loss
from .packing import PackConfig
from .scoring import autocast, score_packs
from .tiny_eval import nll_by_name, score_unpacked, unpacked_view


@dataclass
class TinyTrainConfig:
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    lr_markers: float = 1e-3
    lr_backbone: float = 2e-5          # T-A3 only
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    wd_head: float = 0.01
    warmup: float = 0.03
    min_lr_ratio: float = 0.1
    tokens_per_microbatch: int = 8192
    tokens_per_step: int = 65536
    grad_ckpt: bool = True
    epochs: int = 1
    eval_every: int = 1000
    seed: int = 0
    clip: float = 1.0
    log_every: int = 20
    bf16: bool = True
    loss_weights: dict = field(default_factory=dict)
    max_steps: int | None = None       # stop early (smoke / overfit runs)
    constant_lr: bool = False          # overfit runs

    @classmethod
    def from_dict(cls, d: dict) -> "TinyTrainConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "betas" in known:
            known["betas"] = tuple(known["betas"])
        return cls(**known)


def param_groups(model, tc: TinyTrainConfig) -> list[dict]:
    lr = {"lora": tc.lr_lora, "backbone": tc.lr_backbone, "head": tc.lr_head,
          "markers": tc.lr_markers}
    groups = []
    for name, ps in model.trainable_groups().items():
        ps = [p for p in ps if p.requires_grad]
        if not ps:
            continue
        if name == "head":
            groups.append({"params": [p for p in ps if p.ndim >= 2], "lr": lr[name],
                           "weight_decay": tc.wd_head, "name": "head_decay"})
            groups.append({"params": [p for p in ps if p.ndim < 2], "lr": lr[name],
                           "weight_decay": 0.0, "name": "head_plain"})
        else:
            groups.append({"params": ps, "lr": lr[name], "weight_decay": 0.0, "name": name})
    return [g for g in groups if g["params"]]


def lr_lambda(total: int, warmup: float, min_ratio: float, constant: bool = False):
    w = max(1, int(warmup * total))

    def f(step: int) -> float:
        if constant:
            return 1.0
        if step < w:
            return (step + 1) / w
        prog = min(1.0, (step - w) / max(1, total - w))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


# ------------------------------------------------------------------------------ feeders

class PackedFeeder:
    """TinyJev on packed rows. unpacked=True is baseline B4 (one decision group per
    sequence, same model and recipe)."""

    def __init__(self, tok, M, cfg: PackConfig, packs: list[dict], aug: TinyAugConfig | None,
                 tokens_per_batch: int, unpacked: bool = False, row_packing: bool = True,
                 with_spans: bool = False, mega: int = 512):
        self.tok, self.M, self.cfg, self.packs = tok, M, cfg, packs
        self.aug, self.tpb, self.unpacked = aug, tokens_per_batch, unpacked
        self.row_packing, self.with_spans, self.mega = row_packing, with_spans, mega

    def _prep(self, exs: list[dict]) -> list[dict]:
        return unpacked_view(exs)[0] if self.unpacked else exs

    def epoch(self, rng: random.Random):
        order = list(range(len(self.packs)))
        rng.shuffle(order)
        for s in range(0, len(order), self.mega):
            exs = [augment_tiny(self.packs[i], rng, self.aug) if self.aug else self.packs[i]
                   for i in order[s: s + self.mega]]
            exs = self._prep(exs)
            rendered, kept = render_all(exs, self.tok, self.M, self.cfg,
                                        ids=[e["id"] for e in exs], warn=False)
            rows = make_rows(rendered, self.cfg.max_len, self.row_packing, ids=kept)
            batches = token_batches(rows, self.tpb)
            rng.shuffle(batches)
            for b in batches:
                yield collate(b, self.tok.pad_token_id, self.cfg, self.with_spans)

    @staticmethod
    def tokens(b) -> int:
        return int(b["valid"].sum())

    @staticmethod
    def forward(model, b, device):
        return model(**to_device(b, device))

    def score(self, model, packs, device, bf16=True) -> list[dict]:
        fn = lambda ps: score_packs(model, self.tok, self.M, ps, self.cfg, device,  # noqa: E731
                                    self.tpb, bf16=bf16)
        return score_unpacked(fn, packs) if self.unpacked else fn(packs)

    def mean_tokens(self, n: int = 300, seed: int = 0) -> float:
        rng = random.Random(seed)
        sample = rng.sample(self.packs, min(n, len(self.packs)))
        rendered, _ = render_all(self._prep(sample), self.tok, self.M, self.cfg, warn=False)
        return sum(map(len, rendered)) / max(len(sample), 1)


class PromptFeeder:
    """LetterReadout (B5): one prompt per group, letter-logit softmax."""

    def __init__(self, model, packs: list[dict], aug: TinyAugConfig | None, tokens_per_batch: int):
        self.model, self.packs, self.aug, self.tpb = model, packs, aug, tokens_per_batch

    def epoch(self, rng: random.Random):
        from .lm_readout import prompt_batches, prompt_items
        order = list(range(len(self.packs)))
        rng.shuffle(order)
        for s in range(0, len(order), 512):
            exs = [augment_tiny(self.packs[i], rng, self.aug) if self.aug else self.packs[i]
                   for i in order[s: s + 512]]
            items, _ = prompt_items(exs)
            batches = list(prompt_batches(self.model, items, self.tpb))
            rng.shuffle(batches)
            yield from batches

    @staticmethod
    def tokens(b) -> int:
        return int(b["attention_mask"].sum())

    @staticmethod
    def forward(model, b, device):
        return model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()})

    def score(self, model, packs, device, bf16=True) -> list[dict]:
        from .lm_readout import score_prompts
        return score_prompts(model, packs, device, self.tpb, bf16)

    def mean_tokens(self, n: int = 300, seed: int = 0) -> float:
        from .lm_readout import prompt_items
        sample = random.Random(seed).sample(self.packs, min(n, len(self.packs)))
        items, _ = prompt_items(sample)
        return sum(map(len, self.model.encode_items(items))) / max(len(sample), 1)


# ------------------------------------------------------------------------------ loop

def estimate_steps(feeder, n_packs: int, tc: TinyTrainConfig) -> int:
    tokens = feeder.mean_tokens(seed=tc.seed) * n_packs * tc.epochs
    return max(1, math.ceil(tokens / tc.tokens_per_step))


def train(model, feeder, val_packs: list[dict], tc: TinyTrainConfig, out: str | Path | None,
          device, save_fn=None, log=print) -> dict:
    """Train; save the best checkpoint by macro val NLL via save_fn(out, extra)."""
    rng = random.Random(tc.seed)
    torch.manual_seed(tc.seed)
    model.to(device)
    if tc.grad_ckpt:
        model.enable_grad_ckpt()
    opt = torch.optim.AdamW(param_groups(model, tc), betas=tc.betas, eps=tc.eps)
    total = tc.max_steps or estimate_steps(feeder, len(feeder.packs), tc)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(total, tc.warmup, tc.min_lr_ratio, tc.constant_lr))
    out = Path(out) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train_log.jsonl", "a", encoding="utf-8") if out else None

    def emit(rec):
        log(json.dumps(rec))
        if log_f:
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    def evaluate() -> dict | None:
        if not val_packs:
            return None
        model.eval()
        r = nll_by_name(feeder.score(model, val_packs, device, tc.bf16))
        model.train()
        return r

    history = {"val": [], "train_loss": []}
    best = {"macro": float("inf")}
    v0 = evaluate()
    emit({"event": "init", "val_nll": v0, "train_packs": len(feeder.packs), "steps": total})
    if v0:
        history["val"].append({"step": 0, **v0})

    def maybe_save(v, step, epoch):
        if v and v["macro"] < best["macro"]:
            best.update(v)
            if out and save_fn:
                save_fn(out, {"step": step, "epoch": epoch, "val_nll": v})
            return True
        return False

    step, t0, done = 0, time.time(), False
    run_loss, run_n, parts_run, acc_tok = 0.0, 0, {}, 0
    model.train()
    for epoch in range(tc.epochs):
        for b in feeder.epoch(rng):
            ntok = feeder.tokens(b)
            with autocast(device, tc.bf16):
                z = feeder.forward(model, b, device)
            loss, parts = packed_loss(z, b["labels"], b["names"], tc.loss_weights)
            (loss * ntok / tc.tokens_per_step).backward()
            acc_tok += ntok
            run_loss, run_n = run_loss + float(loss.detach()), run_n + 1
            for k, v in parts.items():
                parts_run[k] = parts_run.get(k, 0.0) + v
            if acc_tok < tc.tokens_per_step:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tc.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            acc_tok, step = 0, step + 1
            if step % tc.log_every == 0:
                emit({"event": "step", "epoch": epoch + 1, "step": step, "loss": run_loss / run_n,
                      "per_decision": {k: v / run_n for k, v in parts_run.items()},
                      "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)})
                history["train_loss"].append(run_loss / run_n)
                run_loss, run_n, parts_run = 0.0, 0, {}
            if tc.eval_every and step % tc.eval_every == 0:
                v = evaluate()
                saved = maybe_save(v, step, epoch + 1)
                emit({"event": "eval", "step": step, "val_nll": v, "saved": saved})
                history["val"].append({"step": step, **(v or {})})
            if tc.max_steps and step >= tc.max_steps:
                done = True
                break
        if acc_tok and not done:              # flush the last partial accumulation
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tc.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            acc_tok, step = 0, step + 1
        if run_n:
            history["train_loss"].append(run_loss / run_n)
            run_loss, run_n, parts_run = 0.0, 0, {}
        v = evaluate()
        saved = maybe_save(v, step, epoch + 1)
        emit({"event": "epoch", "epoch": epoch + 1, "step": step, "val_nll": v, "saved": saved,
              "elapsed_s": round(time.time() - t0)})
        if v:
            history["val"].append({"step": step, **v})
        if done:
            break
    if out and save_fn and not val_packs:
        save_fn(out, {"step": step, "epochs_trained": tc.epochs})
    if log_f:
        log_f.close()
    history["best_val_macro_nll"] = best["macro"]
    history["steps"] = step
    return history
