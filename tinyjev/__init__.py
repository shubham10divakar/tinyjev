"""Tiny-Jev: open 0.6B decision model (Qwen3-0.6B + LoRA) with a reusable state cache.

    import tinyjev
    d = tinyjev.load()                                  # released weights (HF "sdmlai/tiny-jev")
    d = tinyjev.load("runs/tiny-p2-s0")                 # or a local training run
    s = d.session(query=q, passages=chunks)             # encode the state once
    s.decide(["relevance", "sufficient"])
    s.extend(more_chunks)                               # no re-encoding
    d.decide("Route this ticket", ["billing", "tech", "sales", "other"], ticket_text)
"""

__version__ = "0.1.0.dev0"

from jevcore.decider import Q  # noqa: E402

from .decider import Decision, TinyDecider, TinySession  # noqa: E402

__all__ = ["Decision", "Q", "TinyDecider", "TinySession", "load", "__version__"]


def load(path: str | None = None, device: str | None = None, calibration: str = "in",
         max_len: int | None = None, merge: bool = True, dtype: str | None = None) -> TinyDecider:
    return TinyDecider.from_pretrained(path, device, calibration, max_len, merge, dtype)
