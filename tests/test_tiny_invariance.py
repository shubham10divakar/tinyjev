"""Design doc 15 §7.4 / M0: correctness tests T-A … T-E on a tiny random Qwen3 (CPU).

T-A  packed vs alone vs reordered decisions vs permuted options (+ row-packed, padded batch)
T-B  training-path forward (one pass, full mask, batch of mixed rows) vs single-decision forward
T-C  cache path: Session.decide vs the training-path forward
T-D  Session.extend then decide vs a fresh Session on the full state
T-E  cache length after decide equals S (crop worked)

The same probes on real Qwen3-0.6B weights: scripts/m0_check.py.
"""

import copy

import pytest
import torch
from conftest import TINY_LORA, sample_example, tiny_qwen

from jevcore.backbones.qwen3 import Session
from jevcore.collate import collate
from jevcore.invariance import max_diff, perturbations, probe, probs_of
from jevcore.packing import pack_row, render

TOL = 1e-3       # design threshold, fp32
TOL_BF16 = 1e-2


def _other():
    other = sample_example(n_seg=5, long=True)
    other["id"] = "other"
    return other


def _session_probs(model, tok, cfg, ex, state=None):
    s = Session(model, tok, cfg, state or ex["state"])
    return s, {(ex["id"], g["name"], g["seg"]):
               dict(zip(g["options"], torch.softmax(g["logits"], 0).tolist()))
               for g in s.logits(ex["decisions"])}


VARIANTS = [{"attn": "sdpa"}, {"attn": "eager"}, {"attn": "sdpa", "lora": TINY_LORA},
            {"attn": "sdpa", "sink_token": False}, {"attn": "sdpa", "readout": "mean"},
            {"attn": "sdpa", "head": "linear"}]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(f"{k}={v[k]}" if k != "lora" else "lora" for k in v))
def test_TA_invariance(variant):
    model, tok, M = tiny_qwen(**variant)
    cfg = model.pack_config(1024)
    base = sample_example()
    diffs = probe(model, tok, M, base, _other(), cfg)
    assert all(d <= TOL for d in diffs.values()), diffs
    ref = probs_of(model, tok, M, [base], cfg)        # sanity: not trivially uniform
    assert max(max(p.values()) - min(p.values()) for p in ref.values()) > 1e-2


def test_TA_bf16():
    model, tok, M = tiny_qwen(dtype="bf16")
    cfg = model.pack_config(1024)
    diffs = probe(model, tok, M, sample_example(), _other(), cfg)
    assert all(d <= TOL_BF16 for d in diffs.values()), diffs


def test_TA_detects_leaks():
    """Without the block mask (A1) and restart (A7) packing changes outputs: the test has teeth."""
    model, tok, M = tiny_qwen(mask="full", position_restart=False)
    cfg = model.pack_config(1024)
    base = sample_example()
    ref = probs_of(model, tok, M, [base], cfg)
    rev = probs_of(model, tok, M, [perturbations(base)["reversed"]], cfg)
    assert max_diff(ref, rev) > 1e-3


def test_TB_training_forward_matches_single_decisions():
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    cfg = model.pack_config(1024)
    base = sample_example()
    # training batch: the pack row-packed after a longer example, next to a padded row
    rows = [pack_row([render(_other(), tok, M, cfg), render(base, tok, M, cfg)], ["other", base["id"]]),
            pack_row([render(sample_example(n_seg=1) | {"id": "short"}, tok, M, cfg)], ["short"])]
    b = collate(rows, tok.pad_token_id, cfg)
    with torch.no_grad():
        z = model(**b)
    p = torch.softmax(z, -1)
    packed = {}
    for n in range(len(b["names"])):
        if b["g_example"][n] == base["id"]:
            d = base["decisions"][b["g_dec"][n]]
            packed[(base["id"], d["name"], b["g_seg"][n])] = dict(zip(d["options"], p[n].tolist()))
    alone = {}
    for e in perturbations(base)["alone"]:
        alone.update(probs_of(model, tok, M, [e], cfg))
    assert len(packed) == len(alone)
    assert max_diff(packed, alone) <= TOL


@pytest.mark.parametrize("variant", VARIANTS[:4], ids=["sdpa", "eager", "lora", "nosink"])
def test_TC_cache_path_matches_training_path(variant):
    model, tok, M = tiny_qwen(**variant)
    cfg = model.pack_config(1024)
    ex = sample_example()
    ref = probs_of(model, tok, M, [ex], cfg)
    _, cached = _session_probs(model, tok, cfg, ex)
    assert ref.keys() == cached.keys()
    assert max_diff(ref, cached) <= TOL


def test_TC_decisions_one_at_a_time_and_repeated():
    model, tok, M = tiny_qwen()
    cfg = model.pack_config(1024)
    ex = sample_example()
    s, together = _session_probs(model, tok, cfg, ex)
    for _ in range(2):                                  # repeated calls on the same cache
        for d in ex["decisions"]:
            single = {(ex["id"], g["name"], g["seg"]):
                      dict(zip(g["options"], torch.softmax(g["logits"], 0).tolist()))
                      for g in s.logits([d])}
            assert max_diff(together, single) <= TOL


@pytest.mark.parametrize("lora", [None, TINY_LORA])
def test_TD_extend_matches_fresh_session(lora):
    model, tok, M = tiny_qwen(lora=lora)
    cfg = model.pack_config(1024)
    ex = sample_example(n_seg=5)
    first = copy.deepcopy(ex["state"])
    first["segments"] = first["segments"][:2]
    s = Session(model, tok, cfg, first)
    s.extend(ex["state"]["segments"][2:4])
    s.extend(ex["state"]["segments"][4:])
    fresh = Session(model, tok, cfg, ex["state"])
    assert s.row.ids == fresh.row.ids and s.row.pos == fresh.row.pos and s.S == fresh.S
    a = {(g["name"], g["seg"]): g["logits"] for g in s.logits(ex["decisions"])}
    b = {(g["name"], g["seg"]): g["logits"] for g in fresh.logits(ex["decisions"])}
    assert a.keys() == b.keys()
    for k in a:
        pa, pb = torch.softmax(a[k], 0), torch.softmax(b[k], 0)
        assert (pa - pb).abs().max() <= TOL, k
    # and both agree with the one-pass training path
    assert max_diff(probs_of(model, tok, M, [ex], cfg), _session_probs(model, tok, cfg, ex)[1]) <= TOL


def test_TE_crop_restores_cache_length():
    model, tok, M = tiny_qwen()
    cfg = model.pack_config(1024)
    ex = sample_example()
    s = Session(model, tok, cfg, ex["state"])
    assert s.cache_len() == s.S
    s.logits(ex["decisions"])
    assert s.cache_len() == s.S
    s.extend([("Extra", "kiwis are flightless birds .")])
    assert s.cache_len() == s.S == len(s.row)
    s.logits(ex["decisions"][:1])
    assert s.cache_len() == s.S


def test_session_decide_shapes_and_temperature():
    model, tok, M = tiny_qwen()
    cfg = model.pack_config(1024)
    ex = sample_example()
    s = Session(model, tok, cfg, ex["state"])
    res = s.decide(ex["decisions"])
    assert len(res["relevance"]) == 3 and all(abs(sum(p.values()) - 1) < 1e-5 for p in res["relevance"])
    assert set(res["custom_0"]) == {"formal", "informal", "neutral", "positive"}
    hot = s.decide(ex["decisions"], {"custom_0": 100.0})["custom_0"]
    assert max(hot.values()) - min(hot.values()) < max(res["custom_0"].values()) - min(res["custom_0"].values())


def test_extend_overflow_leaves_session_intact():
    from jevcore.packing import PackOverflow
    model, tok, M = tiny_qwen()
    cfg = model.pack_config(120)
    ex = sample_example(n_seg=2)
    s = Session(model, tok, cfg, ex["state"])
    S, n = s.S, s.n_segments
    with pytest.raises(PackOverflow):
        s.extend([("Long", "penguins live in antarctica . " * 40)])
    assert (s.S, s.n_segments, len(s.row), s.cache_len()) == (S, n, S, S)


def test_TC_has_teeth_cache_is_really_used():
    """Corrupting the cached state must change the decisions (else T-C proves nothing)."""
    model, tok, M = tiny_qwen()
    cfg = model.pack_config(1024)
    ex = sample_example()
    s = Session(model, tok, cfg, ex["state"])
    a = s.logits(ex["decisions"])
    for layer in s.cache.layers:
        layer.values.mul_(0.0)
    b = s.logits(ex["decisions"])
    assert max(float((x["logits"] - y["logits"]).abs().max()) for x, y in zip(a, b)) > 1e-3


def test_echo_ablation_T_A7():
    model, tok, M = tiny_qwen(echo=True)
    cfg = model.pack_config(1024)
    ex = sample_example()
    r = render(ex, tok, M, cfg)
    seg0 = [r.ids[i] for i, (b, p) in enumerate(zip(r.blk, r.prt)) if b == 0 and p == 1]
    half = len(seg0) // 2
    assert half and seg0[:half] == seg0[half:]                      # segment 1 rendered twice
    diffs = probe(model, tok, M, ex, _other(), cfg)
    assert all(d <= TOL for d in diffs.values()), diffs
    _, cached = _session_probs(model, tok, cfg, ex)
    assert max_diff(probs_of(model, tok, M, [ex], cfg), cached) <= TOL
    plain_model, _, _ = tiny_qwen()
    assert max_diff(probs_of(plain_model, tok, M, [ex], plain_model.pack_config(1024)),
                    probs_of(model, tok, M, [ex], cfg)) > 1e-4      # echo changes the state
    with pytest.raises(NotImplementedError):
        Session(model, tok, cfg, ex["state"]).extend([("x", "radio city .")])


def test_echo_truncation_fits():
    model, tok, M = tiny_qwen(echo=True)
    cfg = model.pack_config(350)
    ex = sample_example(n_seg=5, long=True)
    assert len(render(ex, tok, M, model.pack_config(2000))) > 350       # would not fit as is
    r = render(ex, tok, M, cfg)
    assert len(r) <= 350 and r.truncated
