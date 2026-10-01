"""LM-token readout baselines (design doc 15 §7.3): one prompt per decision group, option
probabilities = softmax over the answer-letter logits at the answer slot (no generation).

    B0  Qwen3-0.6B zero-shot                     (what decision training adds)
    B1  Qwen3-4B-Instruct-2507 zero-shot         (the LLM judge to beat; same prompt as B0)
    B5  Qwen3-0.6B + LoRA trained on the same data to output the letter   (H8: head vs token)

Same prompt layout as Nano-Jev's LLM baseline (nano_jev/scripts/baselines.py): chat template,
thinking off, "Answer with the letter of the correct option only." Segment-scope decisions
are asked once per segment with only that segment (the unpacked view). Groups with more than
26 options can't be lettered and are skipped (counted) — which is part of H8's point.

B5 is trained with cross-entropy restricted to the option letters (softmax over K letters),
not over the whole vocabulary: same objective shape as the head, so H8 compares readouts.
"""

import json
import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import IGNORE

LETTERS = [chr(65 + i) for i in range(26)]
CONFIG_NAME = "lm_readout_config.json"
ADAPTER_DIR = "adapter"
SYSTEM = "You are a precise decision function. Read the input and answer the question."


def _state_text(header: str, segments: list[dict]) -> str:
    parts = [header] if header else []
    for i, s in enumerate(segments):
        title = s.get("title") or ""
        parts.append(f"[{i + 1}] {title}: {s['text']}" if title else f"[{i + 1}] {s['text']}")
    return "\n".join(parts)


def prompt_items(packs: list[dict], max_options: int = 26) -> tuple[list[dict], int]:
    """One item per labelled-or-not group: {pack, name, dec, seg, options, label, content}.
    Returns (items, n skipped for > max_options)."""
    items, skipped = [], 0
    for p in packs:
        st = p["state"]
        for di, d in enumerate(p["decisions"]):
            if len(d["options"]) > max_options:
                skipped += len(d.get("targets", [0])) if d["scope"] == "segment" else 1
                continue
            if d["scope"] == "segment":
                labels = d.get("labels") or [IGNORE] * len(d["targets"])
                views = [(t, [st["segments"][t]], lab) for t, lab in zip(d["targets"], labels)]
            else:
                views = [(-1, st.get("segments", []), d.get("label", IGNORE))]
            for seg, segs, lab in views:
                opts = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(d["options"]))
                content = (f"<input>\n{_state_text(st.get('header', ''), segs)}\n</input>\n\n"
                           f"{d['question']}\n\nOptions:\n{opts}\n\n"
                           "Answer with the letter of the correct option only.")
                items.append({"pack": p["id"], "name": d["name"], "dec": di, "seg": seg,
                              "options": d["options"], "label": lab, "content": content})
    return items, skipped


def render_prompt(tok, content: str, chat: bool = True) -> str:
    if chat and getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": content}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return f"{SYSTEM}\n\n{content}\nAnswer:\n"


class LetterReadout(nn.Module):
    def __init__(self, lm: nn.Module, tok, cfg: dict):
        super().__init__()
        self.cfg = {"chat": True, "max_prompt_tokens": 2048, "readout": "letter", **cfg}
        self.lm, self.tok = lm, tok
        ids = []
        for L in LETTERS:
            enc = tok(L, add_special_tokens=False)["input_ids"]
            if len(enc) != 1:
                raise ValueError(f"letter {L!r} is not one token in this tokenizer")
            ids.append(enc[0])
        self.register_buffer("letter_ids", torch.tensor(ids), persistent=False)

    @classmethod
    def from_base(cls, base: str, lora: dict | None = None, dtype=torch.bfloat16,
                  attn: str = "sdpa", **cfg):
        tok = AutoTokenizer.from_pretrained(base, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, attn_implementation=attn)
        return cls._wrap(lm, tok, lora, {"backbone": base, **cfg})

    @classmethod
    def from_model(cls, lm, tok, lora: dict | None = None, **cfg):
        """An already-built causal LM (tests: tiny random config)."""
        tok.padding_side = "left"
        return cls._wrap(lm, tok, lora, {"backbone": None, **cfg})

    @classmethod
    def _wrap(cls, lm, tok, lora, cfg):
        if lora:
            from peft import get_peft_model

            from .backbones.qwen3 import _lora_config
            lm = get_peft_model(lm, _lora_config(lora))
        return cls(lm, tok, {**cfg, "lora": lora})

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"lora": [p for p in self.lm.parameters() if p.requires_grad]}

    def enable_grad_ckpt(self) -> None:
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(self.lm, "enable_input_require_grads"):
            self.lm.enable_input_require_grads()

    def encode_items(self, items: list[dict]) -> list[list[int]]:
        cap = self.cfg["max_prompt_tokens"]
        out = []
        for it in items:
            ids = self.tok(render_prompt(self.tok, it["content"], self.cfg["chat"]),
                           add_special_tokens=False)["input_ids"]
            if len(ids) > cap:      # keep the end (question, options, answer slot)
                ids = ids[:cap // 4] + ids[len(ids) - (cap - cap // 4):]
            out.append(ids)
        return out

    def collate(self, items: list[dict], ids: list[list[int]]) -> dict:
        T = max(map(len, ids))
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(ids), T), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), T), dtype=torch.long)
        for b, x in enumerate(ids):                        # left padding: answer slot at -1
            input_ids[b, T - len(x):] = torch.tensor(x)
            mask[b, T - len(x):] = 1
        K = max(len(it["options"]) for it in items)
        valid = torch.zeros((len(items), K), dtype=torch.bool)
        for b, it in enumerate(items):
            valid[b, : len(it["options"])] = True
        return {"input_ids": input_ids, "attention_mask": mask, "g_opt_valid": valid,
                "labels": torch.tensor([it["label"] for it in items]),
                "names": [it["name"] for it in items], "items": items}

    def forward(self, input_ids, attention_mask, g_opt_valid, **_) -> torch.Tensor:
        pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos,
                      logits_to_keep=1)
        K = g_opt_valid.shape[1]
        z = out.logits[:, -1, self.letter_ids[:K]].float()
        return z.masked_fill(~g_opt_valid, float("-inf"))

    def save(self, out: str | Path, extra: dict | None = None) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if self.cfg.get("lora"):
            self.lm.save_pretrained(out / ADAPTER_DIR)
        self.tok.save_pretrained(out)
        (out / CONFIG_NAME).write_text(json.dumps({"model": self.cfg, **(extra or {})}, indent=2),
                                       encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device="cpu", dtype=torch.bfloat16):
        path = Path(path)
        saved = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
        cfg = saved["model"]
        tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(cfg["backbone"], dtype=dtype)
        if cfg.get("lora"):
            from peft import PeftModel
            lm = PeftModel.from_pretrained(lm, path / ADAPTER_DIR)
        return cls(lm, tok, cfg).to(device).eval()


def prompt_batches(model: LetterReadout, items: list[dict], tokens_per_batch: int,
                   max_rows: int = 64):
    """Length-bucketed batches of prompt items (B * T_max <= tokens_per_batch)."""
    ids = model.encode_items(items)
    order = sorted(range(len(items)), key=lambda i: len(ids[i]))
    cur: list[int] = []
    for i in order:
        T = max([len(ids[i])] + [len(ids[j]) for j in cur])
        if cur and ((len(cur) + 1) * T > tokens_per_batch or len(cur) >= max_rows):
            yield model.collate([items[j] for j in cur], [ids[j] for j in cur])
            cur = []
        cur.append(i)
    if cur:
        yield model.collate([items[j] for j in cur], [ids[j] for j in cur])


@torch.no_grad()
def score_prompts(model: LetterReadout, packs: list[dict], device, tokens_per_batch: int = 16384,
                  bf16: bool = True) -> list[dict]:
    """Same output format as scoring.score_packs: one dict per group."""
    from .scoring import autocast
    model.eval()
    items, skipped = prompt_items(packs)
    if skipped:
        warnings.warn(f"{skipped} groups have > 26 options: not scored by the letter readout")
    out = []
    for b in prompt_batches(model, items, tokens_per_batch):
        with autocast(device, bf16):
            z = model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()})
        z = z.float().cpu()
        for n, it in enumerate(b["items"]):
            k = len(it["options"])
            out.append({"pack": it["pack"], "name": it["name"], "dec": it["dec"], "seg": it["seg"],
                        "options": it["options"], "logits": z[n, :k], "label": int(it["label"])})
    return out
