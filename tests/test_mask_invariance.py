"""Design §6.4 / M0: a decision's probabilities don't depend on what it is packed with.

Runs on a tiny random ModernBERT (CPU, fp32). The same probes on trained weights:
scripts/invariance.py.
"""

import pytest
import torch
from conftest import sample_example, tiny_model

from jevcore.collate import collate
from jevcore.invariance import max_diff, perturbations, probe, probs_of
from jevcore.packing import PackConfig, pack_row, render

TOL = 1e-4   # M0 exit criterion (fp32, random weights)


def _other():
    other = sample_example(n_seg=5, long=True)
    other["id"] = "other"
    return other


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_invariance_random_weights(attn):
    model, tok, M = tiny_model(attn=attn)
    cfg = PackConfig(max_len=512, half_window=model.half_window)
    base = sample_example()
    diffs = probe(model, tok, M, base, _other(), cfg)
    assert all(d <= TOL for d in diffs.values()), diffs
    # sanity: outputs are not trivially uniform
    ref = probs_of(model, tok, M, [base], cfg)
    assert max(max(p.values()) - min(p.values()) for p in ref.values()) > 1e-2


@pytest.mark.parametrize("model_cfg", [{"readout": "mean"}, {"head": "linear"}])
def test_invariance_other_readouts(model_cfg):
    model, tok, M = tiny_model(**model_cfg)
    cfg = PackConfig(max_len=512, half_window=model.half_window)
    diffs = probe(model, tok, M, sample_example(), _other(), cfg)
    assert all(d <= TOL for d in diffs.values()), diffs


def test_no_mask_breaks_invariance():
    """H3 sanity: without the mask (A1) packing changes outputs, so the test above has teeth."""
    model, tok, M = tiny_model(mask="full", position_restart=False)
    cfg = PackConfig(max_len=512, mask="full", position_restart=False,
                     half_window=model.half_window)
    base = sample_example()
    ref = probs_of(model, tok, M, [base], cfg)
    rev = probs_of(model, tok, M, [perturbations(base)["reversed"]], cfg)
    assert max_diff(ref, rev) > 1e-3


def test_no_mask_restart_ablation_A7_breaks_invariance():
    """A7: mask on but no position restart -> decisions depend on what precedes them."""
    model, tok, M = tiny_model(position_restart=False)
    cfg = PackConfig(max_len=512, position_restart=False, half_window=model.half_window)
    base = sample_example()
    ref = probs_of(model, tok, M, [base], cfg)
    rev = probs_of(model, tok, M, [perturbations(base)["reversed"]], cfg)
    assert max_diff(ref, rev) > 1e-3


def test_no_nan_with_padding():
    model, tok, M = tiny_model()
    cfg = PackConfig(max_len=512, half_window=model.half_window)
    short = sample_example(n_seg=1)
    rows = [pack_row([render(short, tok, M, cfg)]), pack_row([render(_other(), tok, M, cfg)])]
    b = collate(rows, tok.pad_token_id, cfg)
    with torch.no_grad():
        h = model.encode(b["input_ids"], b["position_ids"], b["full_mask"], b["local_mask"])
    assert torch.isfinite(h).all()
