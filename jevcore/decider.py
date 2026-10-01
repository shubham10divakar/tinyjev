"""Inference API (design §8): one pack, one forward pass, calibrated probabilities per group.

    d = Decider.from_pretrained("runs/micro-jev-dev")
    res = d.run(query=q, passages=chunks,
                decisions=["relevance", "sufficient", Q("Is the query time-sensitive?", ["yes", "no"])])
    res["relevance"]   # [{option: p}, ...] one per passage
    res["sufficient"]  # {option: p}
    res["custom_0"]    # {option: p}

Nano-compatible surface: relevance(q, passages), sufficient(q, passages), grounded(claim, ctx),
decide(question, options, state).
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import registry
from .packing import PackConfig, PackOverflow, render
from .schema import DECISIONS, make_state, query_header, validate
from .scoring import score_packs

CUSTOM_T = "custom"   # temperature key for custom decisions


@dataclass
class Q:
    """A custom decision. scope="segment" asks it once per passage (all passages by default)."""
    question: str
    options: list[str]
    name: str | None = None
    scope: str = "global"
    kind: str = "choice"
    targets: list[int] | None = field(default=None)


class Decider:
    def __init__(self, model, tok, M, cfg: PackConfig, temperatures: dict[str, float], device,
                 config: dict | None = None, path: Path | None = None,
                 tokens_per_batch: int = 32768):
        self.model, self.tok, self.M, self.cfg = model, tok, M, cfg
        self.temperatures, self.device = temperatures, device
        self.config, self.path = config or {}, path
        self.tokens_per_batch = tokens_per_batch

    @property
    def version(self) -> str:
        return self.config.get("version", "unknown")

    @classmethod
    def from_pretrained(cls, path: str | None = None, device: str | None = None,
                        calibration: str = "in", max_len: int | None = None) -> "Decider":
        """calibration: "in" (calibration.json, in-domain) or "ood" (calibration.ood.json)."""
        from .backbones.modernbert import MicroJev
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local = registry.resolve(path)
        model, tok, M, saved = MicroJev.load(local, device)
        cal_name = {"in": "calibration.json", "ood": "calibration.ood.json"}[calibration]
        cal = local / cal_name
        temps = json.loads(cal.read_text(encoding="utf-8")) if cal.exists() else {}
        if not temps:
            warnings.warn(f"{cal_name} not found: using T = 1 (uncalibrated)")
        cfg = PackConfig.from_model_cfg(model.cfg, max_len or saved.get("max_len_eval", 8192))
        cfg.half_window = model.half_window
        return cls(model, tok, M, cfg, temps, device, saved, local)

    # ---------------------------------------------------------------- building packs

    def _decisions(self, decisions, n_seg: int, claim: str | None):
        out, custom = [], 0
        for d in decisions:
            if isinstance(d, str):
                spec = DECISIONS[d]
                if spec.scope == "segment" and n_seg == 0:
                    raise ValueError(f"{d} needs passages")
                q = spec.question
                if "{claim}" in q:
                    if claim is None:
                        raise ValueError(f"{d} needs claim=")
                    q = q.format(claim=claim)
                dd = {"name": d, "kind": spec.kind, "scope": spec.scope, "question": q,
                      "options": list(spec.options)}
                if spec.scope == "segment":
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

    def pack(self, decisions, query: str | None = None, passages=(), header: str | None = None,
             claim: str | None = None, pid: str = "p0") -> dict:
        """passages: strings or (title, text) pairs."""
        segs = [p if isinstance(p, tuple | dict) else ("", p) for p in passages]
        head = header if header is not None else (query_header(query) if query else "")
        ex = {"id": pid, "source": "api", "state": make_state(head, segs),
              "decisions": self._decisions(decisions, len(segs), claim)}
        validate(ex)
        return ex

    def _split(self, pack: dict) -> list[dict]:
        """Split a pack that doesn't fit into several passes over disjoint segment sets."""
        try:
            render(pack, self.tok, self.M, self.cfg)
            return [pack]
        except PackOverflow:
            segs = pack["state"]["segments"]
            if len(segs) <= 1:
                raise
        half = len(segs) // 2
        parts = []
        for lo, hi in ((0, half), (half, len(segs))):
            sub_decs = []
            for d in pack["decisions"]:
                if d["scope"] == "segment":
                    t = [x - lo for x in d["targets"] if lo <= x < hi]
                    if t:   # _offset maps a part's segment index back to the original pack
                        sub_decs.append({**d, "targets": t, "_offset": d.get("_offset", 0) + lo})
                else:
                    sub_decs.append(d)
            sub = {**pack, "id": f"{pack['id']}/{lo}",
                   "state": {"header": pack["state"]["header"], "segments": segs[lo:hi]},
                   "decisions": sub_decs}
            parts += self._split(sub)
        return parts

    # ---------------------------------------------------------------- running

    def _probs(self, name: str, logits: torch.Tensor) -> list[float]:
        t = self.temperatures.get(name, self.temperatures.get(CUSTOM_T, 1.0))
        return torch.softmax(logits / t, 0).tolist()

    def run_packs(self, packs: list[dict]) -> list[dict]:
        """Score full packs (batched). Returns, per pack, {decision name: result}."""
        parts, owner = [], []
        for i, p in enumerate(packs):
            for sub in self._split(p):
                parts.append({**sub, "id": f"{i}|{sub['id']}"})
                owner.append(i)
        for i, p in enumerate(packs):
            if owner.count(i) > 1 and any(d["scope"] == "global" for d in p["decisions"]):
                warnings.warn("pack split into several passes to fit max_len; global decisions "
                              "are averaged over the passes (not exact)")
        scored = score_packs(self.model, self.tok, self.M, parts, self.cfg, self.device,
                             self.tokens_per_batch)
        by_part = {p["id"]: (owner[k], p) for k, p in enumerate(parts)}
        results: list[dict] = [{} for _ in packs]
        sums: list[dict] = [{} for _ in packs]
        for s in scored:
            i, part = by_part[s["pack"]]
            d = part["decisions"][s["dec"]]
            probs = dict(zip(d["options"], self._probs(d["name"], s["logits"])))
            if d["scope"] == "segment":
                n_seg = len(packs[i]["state"]["segments"])
                lst = results[i].setdefault(d["name"], [None] * n_seg)
                lst[s["seg"] + d.get("_offset", 0)] = probs
            else:
                acc = sums[i].setdefault(d["name"], [])
                acc.append(probs)
        for i, acc in enumerate(sums):
            for name, ps in acc.items():
                results[i][name] = {o: sum(p[o] for p in ps) / len(ps) for o in ps[0]}
        return results

    def run(self, decisions, query: str | None = None, passages=(), header: str | None = None,
            claim: str | None = None) -> dict:
        return self.run_packs([self.pack(decisions, query, passages, header, claim)])[0]

    def run_many(self, inputs: list[dict]) -> list[dict]:
        """inputs: [{decisions, query?, passages?, header?, claim?}] -> one result per input."""
        return self.run_packs([self.pack(pid=f"p{i}", **x) for i, x in enumerate(inputs)])

    # ---------------------------------------------------------------- Nano-compatible

    def relevance(self, query: str, passages) -> list[dict[str, float]]:
        return self.run(["relevance"], query=query, passages=passages)["relevance"]

    def sufficient(self, query: str, passages) -> dict[str, float]:
        return self.run(["sufficient"], query=query, passages=passages)["sufficient"]

    def grounded(self, claim: str, context: str) -> dict[str, float]:
        return self.run(["grounded"], header=context, claim=claim)["grounded"]

    def decide(self, question: str, options: list[str], state: str,
               decision: str | None = None) -> dict[str, float]:
        res = self.run([Q(question, options, name=decision or "custom_0")], header=state)
        return res[decision or "custom_0"]
