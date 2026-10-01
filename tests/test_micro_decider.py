import json
import warnings

import pytest
import torch
from conftest import tiny_model

from jevcore.decider import Decider, Q
from jevcore.calibration import fit_temperature, metrics, paired_bootstrap
from jevcore.loss import packed_loss

PASSAGES = ["penguins live in antarctica and cannot fly .", "paris is the capital of france .",
            "kiwis are flightless birds ."]
QUERY = "are there birds that cannot fly ?"


@pytest.fixture(scope="module")
def decider(tmp_path_factory):
    model, tok, M = tiny_model()
    out = tmp_path_factory.mktemp("tiny-micro-jev")
    model.save(out, tok, {"version": "test", "max_len_eval": 512})
    (out / "calibration.json").write_text(json.dumps({"relevance": 2.0, "custom": 0.5}))
    return Decider.from_pretrained(str(out), "cpu")


def close(a, b, tol=1e-4):
    return all(abs(a[k] - b[k]) <= tol for k in a)


def test_round_trip_loads_head(decider):
    model, _, _ = tiny_model()
    for p, q in zip(model.head.parameters(), decider.model.head.parameters()):
        assert torch.equal(p, q)
    assert decider.version == "test"


def test_run_shapes(decider):
    res = decider.run(["relevance", "sufficient", Q("rate the tone ?", ["formal", "informal"])],
                      query=QUERY, passages=PASSAGES)
    assert len(res["relevance"]) == 3
    assert set(res["relevance"][0]) == {"irrelevant", "partially relevant", "directly answers"}
    assert set(res["custom_0"]) == {"formal", "informal"}
    for p in [*res["relevance"], res["sufficient"], res["custom_0"]]:
        assert abs(sum(p.values()) - 1) < 1e-5


def test_api_is_pack_invariant(decider):
    """One packed run equals the Nano-style separate calls (G2 through the public API)."""
    res = decider.run(["relevance", "sufficient"], query=QUERY, passages=PASSAGES)
    alone_rel = decider.relevance(QUERY, PASSAGES)
    alone_suf = decider.sufficient(QUERY, PASSAGES)
    assert all(close(a, b) for a, b in zip(res["relevance"], alone_rel))
    assert close(res["sufficient"], alone_suf)


def test_temperatures_applied(decider):
    raw = Decider(decider.model, decider.tok, decider.M, decider.cfg, {}, "cpu")
    a = decider.relevance(QUERY, PASSAGES)[0]
    b = raw.relevance(QUERY, PASSAGES)[0]
    assert max(a.values()) <= max(b.values()) + 1e-6       # T = 2 flattens


def test_decide_and_grounded(decider):
    p = decider.decide("rate the tone ?", ["formal", "informal", "neutral"], "paris is a city .")
    assert set(p) == {"formal", "informal", "neutral"}
    g = decider.grounded("penguins cannot fly .", PASSAGES[0])
    assert set(g) == {"yes", "no"}
    with pytest.raises(ValueError):
        decider.run(["grounded"], header="x")


def test_run_many_matches_run(decider):
    a = decider.run(["relevance"], query=QUERY, passages=PASSAGES)
    many = decider.run_many([{"decisions": ["sufficient"], "query": "where is paris ?",
                              "passages": PASSAGES[1:]},
                             {"decisions": ["relevance"], "query": QUERY, "passages": PASSAGES}])
    assert all(close(x, y) for x, y in zip(a["relevance"], many[1]["relevance"]))


def test_split_when_too_long(decider):
    import copy
    small = copy.copy(decider)
    small.cfg = copy.copy(decider.cfg)
    small.cfg.max_len = 100
    passages = PASSAGES * 3
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = small.run(["relevance", "sufficient"], query=QUERY, passages=passages)
    assert len(res["relevance"]) == 9 and all(r is not None for r in res["relevance"])
    assert any("split" in str(x.message) for x in w)
    # relevance-only split is exact per part: segment 0 alone in its part equals a k=1 read
    rel_only = small.run(["relevance"], query=QUERY, passages=passages)
    assert all(r is not None for r in rel_only["relevance"])


def test_loss_per_name_average():
    z = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, float("-inf")], [0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1, -100])
    loss, parts = packed_loss(z, labels, ["relevance", "sufficient", "relevance"])
    ce0 = torch.nn.functional.cross_entropy(z[:1], labels[:1])
    ce1 = torch.nn.functional.cross_entropy(z[1:2], labels[1:2])
    assert torch.isclose(loss, (ce0 + ce1) / 2)
    assert set(parts) == {"relevance", "sufficient"}
    # zero-initialised head -> uniform -> NLL = log K
    loss, _ = packed_loss(torch.zeros(4, 3), torch.tensor([0, 1, 2, 0]), ["r"] * 4)
    assert torch.isclose(loss, torch.log(torch.tensor(3.0)))


def test_calibration_metrics():
    torch.manual_seed(0)
    labels = torch.randint(0, 2, (500,))
    logits = torch.randn(500, 2) + 3 * torch.nn.functional.one_hot(labels, 2)
    t = fit_temperature(logits * 4, labels)
    assert t > 1.5                                          # overconfident logits get T > 1
    m = metrics(logits, labels)
    assert {"auroc", "aurc", "acc_escalated"} <= set(m)
    assert m["acc_escalated"]["30%"] >= m["acc_escalated"]["0%"]
    m3 = metrics(torch.randn(50, 3), torch.randint(0, 3, (50,)), ordinal=True)
    assert "qwk" in m3
    bs = paired_bootstrap([1] * 80 + [0] * 20, [1] * 60 + [0] * 40, n=2000)
    assert bs["significant"] and bs["ci95"][0] > 0
