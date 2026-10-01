"""Micro-Jev: ModernBERT encoder + option-scoring head over packed rows (design §4)."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..heads import build_head
from ..packing import MARKERS, MICRO_MARKERS, add_markers, marker_ids

DEFAULT_BASE = "answerdotai/ModernBERT-base"
CONFIG_NAME = "microjev_config.json"
HEAD_NAME = "head.pt"

DEFAULT_MODEL_CFG = {"backbone": DEFAULT_BASE, "attn": "sdpa", "decision_global": True,
                     "option_mode": "isolated", "ref_view": "own", "readout": "marker",
                     "head": "pair", "head_dropout": 0.1, "mask": "block",
                     "position_restart": True}


class MicroJev(nn.Module):
    def __init__(self, enc: nn.Module, model_cfg: dict | None = None):
        super().__init__()
        self.cfg = {**DEFAULT_MODEL_CFG, **(model_cfg or {})}
        self.enc = enc
        self.head = build_head(self.cfg["head"], enc.config.hidden_size, self.cfg["head_dropout"])

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_base(cls, base: str = DEFAULT_BASE, model_cfg: dict | None = None, tok=None):
        """New model from a pretrained backbone. Adds the marker tokens to `tok` (loaded from
        `base` if not given), resizes and initialises their embeddings (§3.3)."""
        cfg = {**DEFAULT_MODEL_CFG, **(model_cfg or {}), "backbone": base}
        tok = tok or AutoTokenizer.from_pretrained(base)
        enc = AutoModel.from_pretrained(base, attn_implementation=cfg["attn"])
        model = cls(enc, cfg)
        M = model.add_markers(tok)
        return model, tok, M

    @classmethod
    def from_config(cls, config, tok, model_cfg: dict | None = None):
        """Randomly initialised backbone (tests, CI)."""
        cfg = {**DEFAULT_MODEL_CFG, **(model_cfg or {})}
        config._attn_implementation = cfg["attn"]
        model = cls(AutoModel.from_config(config, attn_implementation=cfg["attn"]), cfg)
        M = model.add_markers(tok)
        return model, tok, M

    def add_markers(self, tok, names=MICRO_MARKERS) -> dict[str, int]:
        words = {n: MARKERS[n][1] for n in names}
        before = len(tok)
        word_ids = {n: [i for w in ws for i in tok(" " + w, add_special_tokens=False)["input_ids"]]
                    for n, ws in words.items()}
        M = add_markers(tok, names)
        emb = self.enc.get_input_embeddings()
        if len(tok) > emb.num_embeddings:
            self.enc.resize_token_embeddings(len(tok), pad_to_multiple_of=64, mean_resizing=False)
            emb = self.enc.get_input_embeddings()
        if len(tok) > before:   # only initialise markers that were actually new
            with torch.no_grad():
                std = emb.weight[: before].std()
                for n, ids in word_ids.items():
                    if M[n] >= before:
                        v = emb.weight[ids].mean(0) if ids else torch.zeros_like(emb.weight[0])
                        emb.weight[M[n]] = v + torch.randn_like(v) * 0.02 * std
        return M

    # ---------------------------------------------------------------- marker learning rate

    def add_marker_delta(self, M: dict[str, int]) -> nn.Parameter:
        """Trainable offset for the marker rows of the token embedding (§4.5: markers train at
        the head's learning rate, the rest of the embedding at the backbone's). Added through a
        forward hook, so the encoder's state dict is unchanged; `save` folds it in."""
        emb = self.enc.get_input_embeddings()
        ids = sorted(set(M.values()))
        slot = torch.full((emb.num_embeddings,), len(ids), dtype=torch.long)
        slot[ids] = torch.arange(len(ids))
        self.register_buffer("marker_slot", slot.to(emb.weight.device), persistent=False)
        self.marker_ids = ids
        self.marker_delta = nn.Parameter(torch.zeros(len(ids), emb.embedding_dim,
                                                     device=emb.weight.device))

        def hook(_module, inputs, out):
            table = torch.cat([self.marker_delta, self.marker_delta.new_zeros(1, out.shape[-1])])
            return out + table[self.marker_slot[inputs[0]]].to(out.dtype)

        self._delta_hook = emb.register_forward_hook(hook)
        return self.marker_delta

    def folded_state_dict(self) -> dict:
        sd = self.enc.state_dict()
        if getattr(self, "marker_delta", None) is not None:
            key = "embeddings.tok_embeddings.weight"
            w = sd[key].clone()
            w[self.marker_ids] += self.marker_delta.detach().to(w.dtype)
            sd[key] = w
        return sd

    # ---------------------------------------------------------------- save / load

    def save(self, out: str | Path, tok, extra: dict | None = None) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.enc.save_pretrained(out, state_dict=self.folded_state_dict())
        tok.save_pretrained(out)
        torch.save(self.head.state_dict(), out / HEAD_NAME)
        (out / CONFIG_NAME).write_text(json.dumps({"model": self.cfg, **(extra or {})}, indent=2),
                                       encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device="cpu"):
        """Load a saved Micro-Jev folder. Returns (model, tok, M, saved config dict)."""
        path = Path(path)
        saved = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
        cfg = saved["model"]
        tok = AutoTokenizer.from_pretrained(path)
        enc = AutoModel.from_pretrained(path, attn_implementation=cfg.get("attn", "sdpa"))
        model = cls(enc, cfg)
        model.head.load_state_dict(torch.load(path / HEAD_NAME, map_location="cpu"))
        return model.to(device).eval(), tok, marker_ids(tok), saved

    @property
    def half_window(self) -> int:
        return getattr(self.enc.config, "local_attention", 128) // 2

    # ---------------------------------------------------------------- forward

    def _mask(self, m: torch.Tensor, dtype) -> torch.Tensor:
        """Bool [B, T, T] (True = attend) -> what the attention implementation expects."""
        m = m[:, None]
        if self.cfg["attn"] == "sdpa":
            return m
        return torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(
            ~m, torch.finfo(dtype).min)

    def encode(self, input_ids, position_ids, full_mask, local_mask) -> torch.Tensor:
        dtype = self.enc.get_input_embeddings().weight.dtype
        if torch.is_autocast_enabled(input_ids.device.type):
            dtype = torch.get_autocast_dtype(input_ids.device.type)
        out = self.enc(input_ids=input_ids, position_ids=position_ids,
                       attention_mask={"full_attention": self._mask(full_mask, dtype),
                                       "sliding_attention": self._mask(local_mask, dtype)})
        return out.last_hidden_state

    def forward(self, input_ids, position_ids, full_mask, local_mask,
                g_batch, g_anchor, g_opts, g_opt_valid, g_span=None, g_span_valid=None,
                **_) -> torch.Tensor:
        """Returns logits [G, Kmax], -inf at padded options."""
        h = self.encode(input_ids, position_ids, full_mask, local_mask)
        a = h[g_batch, g_anchor]                                       # [G, d]
        if self.cfg["readout"] == "mean":                              # A4: mean over option span
            sv = g_span_valid.unsqueeze(-1).to(h.dtype)                # [G, K, L, 1]
            o = (h[g_batch[:, None, None], g_span] * sv).sum(2) / sv.sum(2).clamp(min=1)
        else:
            o = h[g_batch[:, None], g_opts]                            # [G, K, d]
        z = self.head(a, o).float()
        return z.masked_fill(~g_opt_valid, float("-inf"))
