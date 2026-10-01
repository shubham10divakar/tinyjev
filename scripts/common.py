"""Shared helpers for the Tiny-Jev scripts (config, model construction, data files, dry-run)."""

import copy
import json
import random
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from jevcore.report import read_jsonl, write_jsonl  # noqa: E402,F401

DEFAULT_CONFIG = ROOT / "configs" / "tiny_0p6b.yaml"

# Ablations (design §8) as overrides of the config: (model, data.aug, extra)
ABLATIONS = {
    "T-A1": {"model": {"backbone": "Qwen/Qwen3-0.6B-Base"}},
    "T-A2-r8": {"model": {"lora": {"r": 8, "alpha": 16}}},
    "T-A2-r64": {"model": {"lora": {"r": 64, "alpha": 128}}},
    "T-A3": {"model": {"lora": None}},
    "T-A4": {"aug": {"single_template": True}},
    "T-A5": {"model": {"option_mode": "siblings"}},
    "T-A6": {"aug": {"opt_subsample": False}},
    "T-A7": {"model": {"echo": True}},
    "T-A9": {"model": {"head": "linear"}},
    "no-sink": {"model": {"sink_token": False}},
}


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str | Path | None = None, ablation: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if ablation:
        if ablation not in ABLATIONS:
            raise SystemExit(f"unknown ablation {ablation}; known: {', '.join(ABLATIONS)}")
        ab = ABLATIONS[ablation]
        for k, v in ab.get("model", {}).items():
            if k == "lora" and v is not None:
                cfg["model"]["lora"] = {**cfg["model"]["lora"], **v}
            else:
                cfg["model"][k] = v
        cfg["data"]["aug"] = {**cfg["data"].get("aug", {}), **ab.get("aug", {})}
        cfg["ablation"] = ablation
    return cfg


def model_cfg(cfg: dict) -> dict:
    m = dict(cfg["model"])
    m.pop("head_dtype", None)
    return m


def build_tiny(cfg: dict):
    from jevcore.backbones.qwen3 import TinyJev
    m = model_cfg(cfg)
    return TinyJev.from_base(m["backbone"], m)


def tiny_random(model_overrides: dict | None = None, seed: int = 0):
    """Tiny random Qwen3 + word-level tokenizer (dry runs, CI). Same as tests/conftest."""
    sys.path.insert(0, str(ROOT / "tests"))
    from conftest import TINY_LORA, tiny_qwen
    o = {"lora": TINY_LORA, **(model_overrides or {})}
    return tiny_qwen(seed=seed, **o)


def dry_run_packs(n: int = 24, seed: int = 0) -> list[dict]:
    """Small offline packs that the tiny tokenizer can read (dry runs)."""
    sys.path.insert(0, str(ROOT / "tests"))
    from conftest import sample_example
    from jevcore.data.synthetic import format_packs
    rng = random.Random(seed)
    out = []
    for i in range(n):
        e = sample_example(n_seg=rng.randint(1, 5))
        e["id"] = f"dry-{i}"
        out.append(e)
    fmt = format_packs(n // 3, seed, ["penguins live in antarctica and cannot fly ."])
    return out + fmt


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_dir(d: str | Path, name: str) -> list[dict]:
    p = Path(d) / f"{name}.jsonl"
    return read_jsonl(p) if p.exists() else []


def deep(o):
    return copy.deepcopy(o)
