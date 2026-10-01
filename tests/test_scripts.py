"""Every script's offline --dry-run works end to end (tiny random model, CPU)."""

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _main(name):
    return importlib.import_module(name).main


def test_train_dry_run(tmp_path):
    hist = _main("train")(["--dry-run", "--out", str(tmp_path / "run"), "--max-steps", "4"])
    assert hist["steps"] == 4 and (tmp_path / "run" / "tinyjev_config.json").exists()


@pytest.mark.parametrize("ablation", ["T-A4", "T-A5", "T-A7", "T-A9"])
def test_train_dry_run_ablations(tmp_path, ablation):
    hist = _main("train")(["--dry-run", "--out", str(tmp_path / "run"), "--max-steps", "2",
                           "--ablation", ablation])
    assert hist["steps"] == 2


def test_train_dry_run_b4(tmp_path):
    hist = _main("train")(["--dry-run", "--out", str(tmp_path / "run"), "--max-steps", "2",
                           "--baseline", "B4"])
    assert hist["steps"] == 2


def test_evaluate_dry_run(tmp_path):
    rep = _main("evaluate")(["--dry-run", "--results", str(tmp_path / "eval")])
    assert {"in_domain", "unseen_templates", "clusters", "by_option_count"} <= set(rep)
    assert (tmp_path / "eval" / "predictions_H1.jsonl").exists()


def test_prepare_dry_run(tmp_path):
    _main("prepare_mixture")(["--dry-run", "--phase", "P2", "--heldout", "--total", "500",
                              "--cap", "20", "--n-val", "4", "--out", str(tmp_path / "p2")])
    for f in ("train", "val_seen", "val_unseen", "test", "H1", "card"):
        assert list((tmp_path / "p2").glob(f"{f}.*")), f


def test_m0_and_latency_dry_run():
    _main("m0_check")(["--dry-run"])
    _main("bench_latency")(["--dry-run", "--n", "2", "--warmup", "1"])
