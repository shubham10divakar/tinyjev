"""Tiny-Jev trainer and eval helpers on tiny random models (CPU)."""

import random

import pytest
import torch
from conftest import TINY_LORA, sample_example, tiny_qwen, tiny_qwen_config, tiny_tokenizer

from jevcore.backbones.qwen3 import TinyJev
from jevcore.data.synthetic import format_packs
from jevcore.data.tiny_augment import TinyAugConfig
from jevcore.invariance import max_diff, probs_of
from jevcore.tiny_eval import (accuracy_by_option_count, cluster_report, fit_temperatures,
                               nll_by_name, report, stack_ragged, unpacked_view)
from jevcore.tiny_trainer import (PackedFeeder, PromptFeeder, TinyTrainConfig, lr_lambda,
                                  param_groups, train)


def _packs(n=6):
    out = []
    for i in range(n):
        e = sample_example(n_seg=2 + i % 3)
        e["id"] = f"ex{i}"
        out.append(e)
    return out


def _fast_tc(**kw):
    base = dict(lr_lora=5e-3, lr_head=5e-3, lr_markers=5e-3, tokens_per_microbatch=4096,
                tokens_per_step=600, grad_ckpt=False, bf16=False, eval_every=0, log_every=1000,
                constant_lr=True, max_steps=40, epochs=1000)
    return TinyTrainConfig(**{**base, **kw})


def test_lr_schedule():
    f = lr_lambda(100, 0.1, 0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(99) == pytest.approx(0.1, abs=0.01)
    assert all(f(s) >= f(s + 1) for s in range(10, 99))


def test_param_groups():
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    tc = TinyTrainConfig()
    g = {x["name"]: x for x in param_groups(model, tc)}
    assert set(g) == {"lora", "head_decay", "head_plain", "markers"}
    assert g["lora"]["lr"] == 2e-4 and g["markers"]["lr"] == 1e-3 and g["head_decay"]["weight_decay"] == 0.01
    assert g["lora"]["weight_decay"] == 0 and g["markers"]["weight_decay"] == 0


def test_overfit_and_reload(tmp_path):
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    model.train()
    cfg = model.pack_config(1024)
    packs = _packs()
    feeder = PackedFeeder(tok, M, cfg, packs, None, 4096)
    before = nll_by_name(feeder.score(model, packs, "cpu", bf16=False))["macro"]

    def save(out, extra):
        model.save(out, tok, extra)

    h = train(model, feeder, packs, _fast_tc(), tmp_path / "run", "cpu", save)
    after = h["best_val_macro_nll"]
    assert after < 0.5 * before, (before, after)
    m2, tok2, M2, saved = TinyJev.load(tmp_path / "run")
    assert saved["val_nll"]["macro"] == pytest.approx(after, rel=1e-4)
    model.eval()
    # the saved checkpoint is the best one; with constant lr the last step may differ, so
    # compare the reloaded model's val NLL with the recorded best instead of with `model`
    assert nll_by_name(feeder.score(m2, packs, "cpu", bf16=False))["macro"] == pytest.approx(after, rel=1e-3)
    assert (tmp_path / "run" / "train_log.jsonl").exists()


def test_augmented_epoch_runs():
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    packs = _packs() + format_packs(20)
    feeder = PackedFeeder(tok, M, model.pack_config(1024), packs, TinyAugConfig(), 2048)
    h = train(model, feeder, packs[:4], _fast_tc(max_steps=5, eval_every=2), None, "cpu")
    assert h["steps"] == 5 and len(h["val"]) >= 3


def test_b4_unpacked_feeder_and_scoring():
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    cfg = model.pack_config(1024)
    packs = _packs(3)
    parts, origin = unpacked_view(packs)
    assert all(len(p["decisions"]) == 1 for p in parts)
    assert all(len(p["state"]["segments"]) <= 1 for p in parts if p["decisions"][0]["scope"] == "segment")
    packed = PackedFeeder(tok, M, cfg, packs, None, 4096)
    unp = PackedFeeder(tok, M, cfg, packs, None, 4096, unpacked=True)
    a = {(g["pack"], g["dec"], g["seg"]) for g in packed.score(model, packs, "cpu", False)}
    b = {(g["pack"], g["dec"], g["seg"]) for g in unp.score(model, packs, "cpu", False)}
    assert a == b                                   # same groups, mapped back
    # global decisions see the same state either way -> same numbers; relevance differs
    sa = {(g["pack"], g["dec"], g["seg"]): g["logits"] for g in packed.score(model, packs, "cpu", False)}
    sb = {(g["pack"], g["dec"], g["seg"]): g["logits"] for g in unp.score(model, packs, "cpu", False)}
    for k in sa:
        if k[2] == -1:
            assert torch.allclose(sa[k], sb[k], atol=1e-4)
    h = train(model, unp, [], _fast_tc(max_steps=3), None, "cpu")
    assert h["steps"] == 3


def test_b5_prompt_feeder_trains():
    from transformers import AutoModelForCausalLM

    from jevcore.lm_readout import LETTERS, LetterReadout
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    tok.add_tokens(LETTERS)
    lm = AutoModelForCausalLM.from_config(tiny_qwen_config(tok))
    m = LetterReadout.from_model(lm, tok, lora=TINY_LORA, chat=False)
    packs = _packs(4)
    feeder = PromptFeeder(m, packs, None, 4096)
    before = nll_by_name(feeder.score(m, packs, "cpu", False))["macro"]
    h = train(m, feeder, packs, _fast_tc(max_steps=30, lr_lora=2e-2), None, "cpu")
    assert h["best_val_macro_nll"] < before


def test_eval_helpers():
    g = lambda name, logits, label: {"name": name, "logits": torch.tensor(logits), "label": label}  # noqa: E731
    scored = [g("a", [2.0, 0.0], 0), g("a", [0.0, 1.0, 3.0], 2), g("a", [1.0, 0.0], 1),
              g("b", [0.0, 2.0], 1), g("b", [0.5] * 77, 3), g("b", [0.0, 0.0], -100)]
    z, y = stack_ragged([s for s in scored if s["name"] == "a"])
    assert z.shape == (3, 3) and z[0, 2] < -1e3 and y.tolist() == [0, 2, 1]
    temps = fit_temperatures(scored, scored)
    assert set(temps) == {"a", "b", "custom"} and all(t > 0 for t in temps.values())
    rep = report(scored, temps)
    assert rep["a"]["calibrated"]["n"] == 3 and rep["b"]["calibrated"]["n"] == 2
    cl = cluster_report({"H1": scored[:3], "H2": scored[3:]}, temps)
    assert set(cl) == {"H1", "H2", "macro"} and cl["H1"]["accuracy"] == pytest.approx(2 / 3)
    bins = accuracy_by_option_count(scored)
    assert bins["2-2"]["n"] == 3 and bins["41-100"]["n"] == 1
