"""Tiny-Jev: Qwen3 decoder + LoRA + pair head over causal packed rows (design doc 15 §4).

Training runs a whole packed row in one pass with the causal block mask (`forward`). Inference
can split it: `Session` prefills the state once into a KV cache, then answers any number of
decision batches against it (cropping the decision tokens off again) and appends new chunks
without re-encoding (§4.4). Both paths give the same numbers (tests T-A … T-E, §7.4).
"""

import copy
import json
import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, DynamicCache

from ..heads import build_head
from ..packing import (MARKERS, PackConfig, PackOverflow, Rendered, _encoder, add_markers,
                       allowed, append_decisions, decision_blocks, marker_ids, render_state,
                       segment_ids, state_overhead)
from ..schema import DECISIONS, validate

DEFAULT_BASE = "Qwen/Qwen3-0.6B"
CONFIG_NAME = "tinyjev_config.json"
HEAD_NAME = "head.pt"
MARKERS_NAME = "markers.pt"
ADAPTER_DIR = "adapter"
BACKBONE_DIR = "backbone"
TINY_MARKERS = ("state", "seg", "dec", "q", "qe", "opt", "oe", "ref")
SINK_TOKEN = "<|endoftext|>"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

DEFAULT_LORA = {"r": 16, "alpha": 32, "dropout": 0.05,
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                            "down_proj"]}
DEFAULT_MODEL_CFG = {"backbone": DEFAULT_BASE, "attn": "sdpa", "dtype": "bf16",
                     "option_mode": "isolated", "ref_view": "own", "readout": "end_marker",
                     "head": "pair", "head_dropout": 0.1, "sink_token": True, "mask": "block",
                     "position_restart": True, "lora": DEFAULT_LORA}


class MarkerEmbedding(nn.Module):
    """Small trainable table for the 8 marker tokens, substituted at their positions (§3).
    The 151.9k-row token embedding stays frozen: Adam state for it would cost ~1.2 GB."""

    def __init__(self, marker_ids: list[int], init: torch.Tensor):
        super().__init__()
        self.register_buffer("ids", torch.tensor(marker_ids, dtype=torch.long))
        self.table = nn.Parameter(init.detach().float().clone())        # [n_markers, d], fp32

    def forward(self, input_ids: torch.Tensor, base_embeds: torch.Tensor) -> torch.Tensor:
        hit = input_ids[..., None] == self.ids                          # [B, T, n]
        sub = (hit.to(self.table.dtype) @ self.table).to(base_embeds.dtype)
        return torch.where(hit.any(-1, keepdim=True), sub, base_embeds)


def _lora_config(lora: dict):
    from peft import LoraConfig
    return LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                      target_modules=list(lora["targets"]), bias="none")


def sink_id(tok) -> int | None:
    i = tok.convert_tokens_to_ids(SINK_TOKEN)
    if i is None or i == tok.unk_token_id:
        i = tok.bos_token_id if tok.bos_token_id is not None else tok.pad_token_id
    return i


class TinyJev(nn.Module):
    def __init__(self, bb: nn.Module, M: dict[str, int], marker_init: torch.Tensor,
                 model_cfg: dict | None = None):
        super().__init__()
        self.cfg = {**DEFAULT_MODEL_CFG, **(model_cfg or {})}
        self.base_config = bb.config
        if "sliding_attention" in set(bb.config.layer_types):
            # Our masks would also need the window for those layers; Qwen3-0.6B has none (§2).
            raise ValueError("backbone has sliding-window layers; not supported")
        self.layer_types = sorted(set(bb.config.layer_types))
        self.M = dict(M)
        lora = self.cfg.get("lora")
        if lora:
            from peft import get_peft_model
            bb = get_peft_model(bb, _lora_config(lora))
        else:   # T-A3 full fine-tune: everything but the token embedding trains
            bb.get_input_embeddings().weight.requires_grad_(False)
        self.bb = bb
        self.markers = MarkerEmbedding([M[n] for n in TINY_MARKERS], marker_init)
        d = self.base_config.hidden_size
        self.head = build_head(self.cfg["head"], d, self.cfg["head_dropout"]).float()
        self.sink_id: int | None = None

    # ---------------------------------------------------------------- construction

    @staticmethod
    def _marker_init(bb, tok, M, std_scale=0.02, seed=0) -> torch.Tensor:
        """Mean embedding of each marker's words (§3.3), plus a little noise so markers that
        share words (<opt>/<oe>, <seg>/<ref>) start apart."""
        emb = bb.get_input_embeddings().weight.detach().float()
        g = torch.Generator().manual_seed(seed)
        std = emb.std()
        rows = []
        for n in TINY_MARKERS:
            ids = [i for w in MARKERS[n][1] for i in tok(" " + w, add_special_tokens=False)["input_ids"]]
            ids = [i for i in ids if i < emb.shape[0]]
            v = emb[ids].mean(0) if ids else torch.zeros(emb.shape[1])
            rows.append(v + torch.randn(v.shape, generator=g) * std_scale * std)
        return torch.stack(rows)

    @classmethod
    def _build(cls, bb, tok, cfg):
        M = add_markers(tok, TINY_MARKERS)
        model = cls(bb, M, cls._marker_init(bb, tok, M), cfg)
        model.sink_id = sink_id(tok) if cfg.get("sink_token", True) else None
        return model, tok, M

    @classmethod
    def from_base(cls, base: str = DEFAULT_BASE, model_cfg: dict | None = None, tok=None):
        """New model from a pretrained Qwen3 (no LM head). Adds the markers to `tok`."""
        cfg = {**DEFAULT_MODEL_CFG, **(model_cfg or {}), "backbone": base}
        tok = tok or AutoTokenizer.from_pretrained(base)
        bb = AutoModel.from_pretrained(base, dtype=DTYPES[cfg["dtype"]],
                                       attn_implementation=cfg["attn"])
        return cls._build(bb, tok, cfg)

    @classmethod
    def from_config(cls, config, tok, model_cfg: dict | None = None):
        """Randomly initialised backbone (tests, CI). Saved with its weights."""
        cfg = {**DEFAULT_MODEL_CFG, "dtype": "fp32", **(model_cfg or {}), "backbone": None}
        bb = AutoModel.from_config(config, attn_implementation=cfg["attn"],
                                   dtype=DTYPES[cfg["dtype"]])
        return cls._build(bb, tok, cfg)

    def pack_config(self, max_len: int) -> PackConfig:
        c = self.cfg
        return PackConfig(max_len=max_len, isolated=c["option_mode"] == "isolated",
                          position_restart=c["position_restart"], mask=c["mask"],
                          ref_view=c["ref_view"], causal=True, sink_id=self.sink_id)

    def enable_grad_ckpt(self) -> None:
        self.bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        """{"lora" (or "backbone"), "head", "markers"}: the §4.3 / §6 learning-rate groups."""
        bb = [p for p in self.bb.parameters() if p.requires_grad]
        return {"lora" if self.cfg.get("lora") else "backbone": bb,
                "head": list(self.head.parameters()), "markers": [self.markers.table]}

    def merge_lora(self) -> None:
        """Fold LoRA into the backbone weights (inference speed). Not reversible."""
        if self.cfg.get("lora") and hasattr(self.bb, "merge_and_unload"):
            self.bb = self.bb.merge_and_unload()
            self.cfg = {**self.cfg, "lora_merged": True}

    # ---------------------------------------------------------------- save / load

    def save(self, out: str | Path, tok, extra: dict | None = None) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if self.cfg.get("lora") and not self.cfg.get("lora_merged"):
            self.bb.save_pretrained(out / ADAPTER_DIR)
            if self.cfg.get("backbone") is None:      # random base (tests): keep its weights too
                copy.deepcopy(self.bb).unload().save_pretrained(out / BACKBONE_DIR)
        else:
            self.bb.save_pretrained(out / BACKBONE_DIR)
        tok.save_pretrained(out)
        torch.save(self.head.state_dict(), out / HEAD_NAME)
        torch.save(self.markers.state_dict(), out / MARKERS_NAME)
        (out / CONFIG_NAME).write_text(
            json.dumps({"model": self.cfg, "markers": self.M, "sink_id": self.sink_id,
                        **(extra or {})}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device="cpu", merge: bool = False, dtype: str | None = None):
        """Load a saved Tiny-Jev folder. Returns (model, tok, M, saved config dict)."""
        path = Path(path)
        saved = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
        cfg = dict(saved["model"])
        if dtype:
            cfg["dtype"] = dtype
        tok = AutoTokenizer.from_pretrained(path)
        M = marker_ids(tok, TINY_MARKERS)
        src = path / BACKBONE_DIR if (path / BACKBONE_DIR).exists() else cfg["backbone"]
        bb = AutoModel.from_pretrained(src, dtype=DTYPES[cfg["dtype"]],
                                       attn_implementation=cfg["attn"])
        lora = cfg.get("lora") if not cfg.get("lora_merged") else None
        model = cls(bb, M, torch.zeros(len(TINY_MARKERS), bb.config.hidden_size),
                    {**cfg, "lora": None})
        if lora:
            from peft import PeftModel
            model.bb = PeftModel.from_pretrained(bb, path / ADAPTER_DIR)
            model.cfg["lora"] = lora
            if merge:
                model.merge_lora()
        model.markers.load_state_dict(torch.load(path / MARKERS_NAME, map_location="cpu"))
        model.head.load_state_dict(torch.load(path / HEAD_NAME, map_location="cpu"))
        model.sink_id = saved.get("sink_id")
        return model.to(device).eval(), tok, M, saved

    # ---------------------------------------------------------------- forward

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.bb.get_input_embeddings()
        safe = ids.clamp_max(emb.num_embeddings - 1)       # marker ids may sit past the table
        return self.markers(ids, emb(safe))

    def _mask(self, m: torch.Tensor, dtype) -> dict[str, torch.Tensor]:
        """Bool [B, Tq, Tk] (True = attend) -> per-layer-type 4D masks for Qwen3Model."""
        m = m[:, None]
        if self.cfg["attn"] != "sdpa":   # eager adds the mask
            m = torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(
                ~m, torch.finfo(dtype).min)
        return {t: m for t in self.layer_types}

    def encode(self, ids, pos, mask, cache=None):
        """ids / pos [B, T]; mask bool [B, T, T_cache + T]. Returns (hidden [B, T, d], cache)."""
        x = self.embed(ids)
        dtype = x.dtype
        if torch.is_autocast_enabled(ids.device.type):
            dtype = torch.get_autocast_dtype(ids.device.type)
        out = self.bb(inputs_embeds=x, position_ids=pos, attention_mask=self._mask(mask, dtype),
                      past_key_values=cache, use_cache=cache is not None)
        return out.last_hidden_state, out.past_key_values

    def score(self, h, g_batch, g_anchor, g_opts, g_opt_valid, g_span=None, g_span_valid=None):
        """Head over gathered readouts, in fp32 (§4.2). Returns logits [G, K], -inf at padding."""
        a = h[g_batch, g_anchor].float()                                # [G, d]
        if self.cfg["readout"] == "mean":                               # A4: mean over the span
            sv = g_span_valid.unsqueeze(-1).float()
            o = (h[g_batch[:, None, None], g_span].float() * sv).sum(2) / sv.sum(2).clamp(min=1)
        else:
            o = h[g_batch[:, None], g_opts].float()                     # [G, K, d]
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = self.head(a, o)
        return z.masked_fill(~g_opt_valid, float("-inf"))

    def forward(self, input_ids, position_ids, full_mask, g_batch, g_anchor, g_opts, g_opt_valid,
                g_span=None, g_span_valid=None, **_) -> torch.Tensor:
        """Training path: whole packed rows in one pass. Returns logits [G, Kmax]."""
        h, _ = self.encode(input_ids, position_ids, full_mask)
        return self.score(h, g_batch, g_anchor, g_opts, g_opt_valid, g_span, g_span_valid)


# ------------------------------------------------------------------------------ cached inference

def _tensors(r: Rendered, start: int, device):
    t = lambda x: torch.tensor(x[start:], dtype=torch.long, device=device)[None]  # noqa: E731
    return t(r.ids), t(r.pos)


def allowed_with_state(r: Rendered, q_start: int, cfg: PackConfig, device) -> torch.Tensor:
    """Rows q_start.. of allowed() over the whole (state ⊕ new tokens) row: [1, Tq, T]."""
    T = len(r)
    blk = torch.tensor(r.blk, device=device)
    prt = torch.tensor(r.prt, device=device)
    zeros = torch.zeros(T, dtype=torch.long, device=device)
    ones = torch.ones(T, dtype=torch.bool, device=device)
    return allowed(zeros, blk, prt, ones, cfg.isolated, True, cfg.ref_view, q_start)[None]


class Session:
    """One state, many decision calls; chunks can be appended (the RAG loop, §4.4).

        s = Session(model, tok, cfg, {"header": "query: ...", "segments": [...]})
        s.logits([...decision dicts...])   # [(decision index, segment, options, logits)]
        s.extend([{"title": ..., "text": ...}])

    The state is never truncated silently: segments that don't fit are capped per segment
    (with a warning) at prefill, and `extend` raises PackOverflow past cfg.max_len.
    """

    def __init__(self, model: TinyJev, tok, cfg: PackConfig, state: dict, device=None):
        if not cfg.causal:
            raise ValueError("Session needs a causal PackConfig (model.pack_config)")
        self.model, self.tok, self.M = model, tok, model.M
        self.cfg = copy.copy(cfg)
        self.cfg.drop_untargeted = False          # a session never drops segments
        self.device = device or next(model.parameters()).device
        self.enc = _encoder(tok)
        self.state = {"header": state.get("header", ""),
                      "segments": [dict(s) for s in state.get("segments", [])]}
        budget = self.cfg.max_len - state_overhead(self.cfg)
        self.row = render_state(self.state, tok, self.M, self.cfg, budget, enc=self.enc)
        if self.row.truncated:
            warnings.warn("session state was truncated to fit max_len")
        self.cache = DynamicCache(config=model.base_config)
        self._run(self.row, 0, keep=True)

    @property
    def S(self) -> int:
        return self.row.state_len

    @property
    def n_segments(self) -> int:
        return len(self.state["segments"])

    def cache_len(self) -> int:
        return self.cache.get_seq_length()

    @torch.no_grad()
    def _run(self, r: Rendered, start: int, keep: bool) -> torch.Tensor:
        """Encode r's tokens from `start` on top of the cache (which holds r's first `start`
        tokens). keep=False crops them off again. Returns hidden states [T - start, d]."""
        n = len(r) - start
        if n <= 0:
            return torch.empty(0, self.model.base_config.hidden_size, device=self.device)
        ids, pos = _tensors(r, start, self.device)
        mask = allowed_with_state(r, start, self.cfg, self.device)
        try:
            h, self.cache = self.model.encode(ids, pos, mask, cache=self.cache)
        finally:
            if not keep and self.cache.get_seq_length() > start:
                self.cache.crop(-(self.cache.get_seq_length() - start))   # always negative (§11)
        return h[0]

    def _rendered_decisions(self, decisions: list[dict]) -> Rendered:
        validate({"state": self.state, "decisions": decisions})
        blocks, _ = decision_blocks(decisions, self.enc, self.M, True)
        r = Rendered(ids=list(self.row.ids), pos=list(self.row.pos), blk=list(self.row.blk),
                     prt=list(self.row.prt), state_len=self.S,
                     kept_segments=list(self.row.kept_segments))
        return append_decisions(r, decisions, blocks, self.M, self.cfg)

    @torch.no_grad()
    def logits(self, decisions: list[dict]) -> list[dict]:
        """One entry per group: {dec, name, seg, options, logits (1-D CPU float)}."""
        self.model.eval()
        r = self._rendered_decisions(decisions)
        h = self._run(r, self.S, keep=False)[None]                     # [1, Td, d]
        groups = r.groups
        K = max(len(g.options) for g in groups)
        dev = h.device
        opts = torch.zeros(len(groups), K, dtype=torch.long, device=dev)
        valid = torch.zeros(len(groups), K, dtype=torch.bool, device=dev)
        for n, g in enumerate(groups):
            opts[n, : len(g.options)] = torch.tensor(g.options, device=dev) - self.S
            valid[n, : len(g.options)] = True
        anchors = torch.tensor([g.anchor - self.S for g in groups], device=dev)
        span = span_valid = None
        if self.model.cfg["readout"] == "mean":
            L = max(e - s for g in groups for s, e in g.spans)
            span = torch.zeros(len(groups), K, L, dtype=torch.long, device=dev)
            span_valid = torch.zeros(len(groups), K, L, dtype=torch.bool, device=dev)
            for n, g in enumerate(groups):
                for j, (s, e) in enumerate(g.spans):
                    span[n, j, : e - s] = torch.arange(s - self.S, e - self.S)
                    span_valid[n, j, : e - s] = True
        z = self.model.score(h, torch.zeros(len(groups), dtype=torch.long, device=dev), anchors,
                             opts, valid, span, span_valid).float().cpu()
        return [{"dec": g.dec, "name": g.name, "seg": g.seg,
                 "options": decisions[g.dec]["options"], "logits": z[n, : len(g.options)]}
                for n, g in enumerate(groups)]

    def decide(self, decisions: list[dict], temperatures: dict[str, float] | None = None,
               default_T: float = 1.0) -> dict:
        """{decision name: {option: p}} (global) or [{option: p} per target segment]."""
        temps = temperatures or {}
        out: dict = {}
        for g in self.logits(decisions):
            t = temps.get(g["name"], default_T)
            probs = dict(zip(g["options"], torch.softmax(g["logits"] / t, 0).tolist()))
            if g["seg"] >= 0:
                out.setdefault(g["name"], [None] * self.n_segments)[g["seg"]] = probs
            else:
                out[g["name"]] = probs
        return out

    def extend(self, segments: list[dict]) -> None:
        """Append chunks: they attend to the whole previous state (causal), positions continue,
        nothing already cached is recomputed."""
        segs = [s if isinstance(s, dict) else {"title": s[0], "text": s[1]} for s in segments]
        start = self.S
        for k, seg in enumerate(segs):
            i = self.n_segments + k
            self.row.push(segment_ids(i, seg, self.enc, self.M), blk=0, prt=i + 1)
            self.row.kept_segments.append(i)
        if len(self.row) > self.cfg.max_len:
            del self.row.ids[start:], self.row.pos[start:], self.row.blk[start:], \
                self.row.prt[start:], self.row.kept_segments[self.n_segments:]
            raise PackOverflow(f"state would exceed max_len={self.cfg.max_len}")
        self.row.state_len = len(self.row)
        self.state["segments"] += segs
        self._run(self.row, start, keep=True)


def builtin(name: str, n_seg: int, **fields) -> dict:
    """A decision dict for a builtin decision over all n_seg segments (session helper)."""
    spec = DECISIONS[name]
    d = {"name": name, "kind": spec.kind, "scope": spec.scope,
         "question": spec.question.format(**fields) if fields else spec.question,
         "options": list(spec.options)}
    if spec.scope == "segment":
        d["targets"] = list(range(n_seg))
    return d
