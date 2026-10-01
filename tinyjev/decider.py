"""Tiny-Jev inference API (design doc 15 §9).

    import tinyjev
    d = tinyjev.load()                                    # HF "sdmlai/tiny-jev" (or a local run)
    s = d.session(query=q, passages=chunks)               # prefill once
    r1 = s.decide(["relevance", "sufficient"])
    s.extend(more_chunks)                                  # incremental, no re-encode
    r2 = s.decide(["relevance", "sufficient", tinyjev.Q("Is the query time-sensitive?", ["yes", "no"])])

    d.decide("Is this ticket about billing?", ["yes", "no"], ticket_text)
    d.policy(tau=0.8)                                      # results get .escalate

Every result is a `Decision`: a dict {option: p} (so Nano-style code keeps working) plus
`.label` (argmax), `.confidence` (max p) and `.escalate` (confidence < τ, when a policy is set).
Built-in decisions use their own temperature; anything else uses the global `T_custom`.
"""

import json
import warnings
from pathlib import Path

import torch

from jevcore.backbones.qwen3 import Session, TinyJev
from jevcore.decider import Q
from jevcore.scoring import score_packs
from jevcore.schema import DECISIONS, make_state, query_header, validate
from jevcore.data.tasks import canonical, passage_options

from . import registry

CUSTOM_T = "custom"
BUILTINS = {**{k: (v.kind, v.scope, v.question, list(v.options)) for k, v in DECISIONS.items()},
            "which_passage": ("choice", "global", canonical("which_passage")[0], None)}


class Decision(dict):
    """{option: probability} with .label, .confidence and .escalate."""

    def __init__(self, probs: dict[str, float], tau: float | None = None):
        super().__init__(probs)
        self.label = max(probs, key=probs.get)
        self.confidence = probs[self.label]
        self.tau = tau
        self.escalate = tau is not None and self.confidence < tau

    def __repr__(self):
        flag = " escalate" if self.escalate else ""
        return f"Decision({dict.__repr__(self)}, label={self.label!r}, confidence={self.confidence:.3f}{flag})"


def _segments(passages) -> list[dict]:
    out = []
    for p in passages:
        if isinstance(p, dict):
            out.append({"title": p.get("title", ""), "text": p["text"]})
        elif isinstance(p, tuple):
            out.append({"title": p[0], "text": p[1]})
        else:
            out.append({"title": "", "text": p})
    return out


class TinyDecider:
    def __init__(self, model: TinyJev, tok, max_len: int, temperatures: dict[str, float],
                 device, config: dict | None = None, path: Path | None = None):
        self.model, self.tok, self.M = model, tok, model.M
        self.cfg = model.pack_config(max_len)
        self.temperatures, self.device = temperatures, device
        self.config, self.path = config or {}, path
        self.taus: dict[str, float] = {}
        self.default_tau: float | None = None

    @classmethod
    def from_pretrained(cls, path: str | None = None, device: str | None = None,
                        calibration: str = "in", max_len: int | None = None,
                        merge: bool = True, dtype: str | None = None) -> "TinyDecider":
        """calibration: "in" (calibration.json) or "ood" (calibration.ood.json)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or ("bf16" if str(device).startswith("cuda") else "fp32")
        local = registry.resolve(path)
        model, tok, _, saved = TinyJev.load(local, device, merge=merge, dtype=dtype)
        name = {"in": "calibration.json", "ood": "calibration.ood.json"}[calibration]
        cal = local / name
        temps = json.loads(cal.read_text(encoding="utf-8")) if cal.exists() else {}
        if not temps:
            warnings.warn(f"{name} not found: using T = 1 (uncalibrated)")
        return cls(model, tok, max_len or saved.get("max_len_eval", 4096), temps, device,
                   saved, local)

    @property
    def version(self) -> str:
        return self.config.get("version", "unknown")

    # ---------------------------------------------------------------- policy

    def policy(self, tau: float | dict[str, float] | None = None) -> "TinyDecider":
        """Escalate when max p < tau. A dict sets per-decision thresholds ("custom" = default
        for everything else). Fit τ on your own validation data: thresholds don't transfer
        across tasks (JEV-as-a-Judge, Cyber-Jev)."""
        if isinstance(tau, dict):
            self.taus = dict(tau)
            self.default_tau = tau.get(CUSTOM_T)
        else:
            self.taus, self.default_tau = {}, tau
        return self

    def _tau(self, name: str) -> float | None:
        return self.taus.get(name, self.default_tau)

    def _T(self, name: str) -> float:
        return self.temperatures.get(name, self.temperatures.get(CUSTOM_T, 1.0))

    def _result(self, name: str, options: list[str], logits: torch.Tensor) -> Decision:
        p = torch.softmax(logits.float() / self._T(name), 0).tolist()
        return Decision(dict(zip(options, p)), self._tau(name))

    # ---------------------------------------------------------------- decisions

    def decisions(self, specs, n_seg: int, claim: str | None = None) -> list[dict]:
        """Strings (built-in names), Q objects or dicts -> decision dicts."""
        out, custom = [], 0
        for d in specs:
            if isinstance(d, str):
                if d not in BUILTINS:
                    raise ValueError(f"unknown built-in decision {d!r}; use tinyjev.Q(...)")
                kind, scope, q, opts = BUILTINS[d]
                if scope == "segment" and n_seg == 0:
                    raise ValueError(f"{d} needs passages")
                if "{claim}" in q:
                    if claim is None:
                        raise ValueError(f"{d} needs claim=")
                    q = q.format(claim=claim)
                dd = {"name": d, "kind": kind, "scope": scope, "question": q,
                      "options": opts if opts else passage_options(n_seg)}
                if scope == "segment":
                    dd["targets"] = list(range(n_seg))
            else:
                if isinstance(d, dict):
                    d = Q(**d)
                name = d.name or f"custom_{custom}"
                custom += d.name is None
                dd = {"name": name, "kind": d.kind, "scope": d.scope, "question": d.question,
                      "options": list(d.options)}
                if d.scope == "segment":
                    dd["targets"] = list(d.targets if d.targets is not None else range(n_seg))
            out.append(dd)
        return out

    def _collect(self, decs: list[dict], groups, n_seg: int) -> dict:
        res: dict = {}
        for g in groups:
            d = decs[g["dec"]]
            r = self._result(d["name"], d["options"], g["logits"])
            if d["scope"] == "segment":
                res.setdefault(d["name"], [None] * n_seg)[g["seg"]] = r
            else:
                res[d["name"]] = r
        return res

    # ---------------------------------------------------------------- sessions

    def session(self, query: str | None = None, passages=(), header: str | None = None) -> "TinySession":
        head = header if header is not None else (query_header(query) if query else "")
        return TinySession(self, {"header": head, "segments": _segments(passages)})

    def run(self, decisions, query: str | None = None, passages=(), header: str | None = None,
            claim: str | None = None) -> dict:
        """One-shot: prefill + decide (a throwaway session)."""
        return self.session(query, passages, header).decide(decisions, claim=claim)

    def decide(self, question: str, options: list[str], state: str = "", passages=(),
               name: str | None = None) -> Decision:
        """One custom decision on a text (and optional passages)."""
        key = name or "custom_0"
        return self.run([Q(question, list(options), name=key)], header=state, passages=passages)[key]

    def decide_many(self, items: list[dict]) -> list[dict]:
        """Batched one-shot calls on many states (training-path forward, no cache):
        items = [{decisions, query?, passages?, header?, claim?}] -> one result dict each."""
        packs = []
        for i, x in enumerate(items):
            segs = _segments(x.get("passages", ()))
            head = x.get("header") if x.get("header") is not None else (
                query_header(x["query"]) if x.get("query") else "")
            p = {"id": f"p{i}", "source": "api", "state": make_state(head, segs),
                 "decisions": self.decisions(x["decisions"], len(segs), x.get("claim"))}
            validate(p)
            packs.append(p)
        scored = score_packs(self.model, self.tok, self.M, packs, self.cfg, self.device)
        by_pack: dict[str, list] = {}
        for s in scored:
            by_pack.setdefault(s["pack"], []).append(s)
        return [self._collect(p["decisions"], by_pack.get(p["id"], []), len(p["state"]["segments"]))
                for p in packs]

    # ---------------------------------------------------------------- Nano-compatible

    def relevance(self, query: str, passages) -> list[Decision]:
        return self.run(["relevance"], query=query, passages=passages)["relevance"]

    def sufficient(self, query: str, passages) -> Decision:
        return self.run(["sufficient"], query=query, passages=passages)["sufficient"]

    def grounded(self, claim: str, context: str) -> Decision:
        return self.run(["grounded"], header=context, claim=claim)["grounded"]


class TinySession:
    """A prefilled state (KV cache). decide() any number of times; extend() with new chunks."""

    def __init__(self, decider: TinyDecider, state: dict):
        self.d = decider
        self.s = Session(decider.model, decider.tok, decider.cfg, state, decider.device)

    @property
    def n_segments(self) -> int:
        return self.s.n_segments

    @property
    def tokens(self) -> int:
        return self.s.S

    def decide(self, decisions, claim: str | None = None) -> dict:
        decs = self.d.decisions(decisions, self.n_segments, claim)
        return self.d._collect(decs, self.s.logits(decs), self.n_segments)

    def extend(self, passages) -> "TinySession":
        self.s.extend(_segments(passages))
        return self
