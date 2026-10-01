"""Public API (design doc 15 §9) on a saved tiny random model."""

import json
import warnings

import pytest
import torch
from conftest import TINY_LORA, tiny_qwen

import tinyjev

PASSAGES = ["penguins live in antarctica and cannot fly .", "paris is the capital of france .",
            ("Eiffel", "the eiffel tower was built in 1889 .")]


@pytest.fixture(scope="module")
def d(tmp_path_factory):
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    out = tmp_path_factory.mktemp("tiny") / "run"
    model.save(out, tok, {"version": "test", "max_len_eval": 1024})
    (out / "calibration.json").write_text(json.dumps({"relevance": 1.5, "custom": 2.0}))
    return tinyjev.load(str(out), device="cpu")


def test_load(d):
    assert d.version == "test" and d.temperatures["custom"] == 2.0
    assert d.cfg.causal and d.cfg.max_len == 1024


def test_session_decide_extend(d):
    s = d.session(query="are there birds that cannot fly ?", passages=PASSAGES[:2])
    r1 = s.decide(["relevance", "sufficient", "which_passage"])
    assert len(r1["relevance"]) == 2 and set(r1["which_passage"]) == {"[1]", "[2]", "none of them"}
    n0 = s.tokens
    s.extend(PASSAGES[2:])
    r2 = s.decide(["relevance", tinyjev.Q("is the query time sensitive ?", ["yes", "no"])])
    assert len(r2["relevance"]) == 3 and s.tokens > n0
    fresh = d.session(query="are there birds that cannot fly ?", passages=PASSAGES).decide(["relevance"])
    for a, b in zip(r2["relevance"], fresh["relevance"]):
        assert max(abs(a[k] - b[k]) for k in a) < 1e-4


def test_decide_result_and_policy(d):
    r = d.decide("rate the tone ?", ["formal", "informal", "neutral"], "paris is the capital of france .")
    assert isinstance(r, tinyjev.Decision) and abs(sum(r.values()) - 1) < 1e-5
    assert r.label == max(r, key=r.get) and r.confidence == r[r.label] and not r.escalate
    d.policy(tau=1.0)
    assert d.decide("rate the tone ?", ["formal", "informal"], "x").escalate
    d.policy(tau={"custom_0": 0.0})
    assert not d.decide("rate the tone ?", ["formal", "informal"], "x").escalate
    d.policy(None)


def test_temperatures_used(d):
    sess = d.session(query="q ?", passages=PASSAGES[:1])
    raw = sess.s.logits(d.decisions(["relevance"], 1))[0]["logits"]
    got = sess.decide(["relevance"])["relevance"][0]
    want = torch.softmax(raw / 1.5, 0).tolist()
    assert max(abs(a - b) for a, b in zip(got.values(), want)) < 1e-6


def test_decide_many_matches_sessions(d):
    items = [{"decisions": ["relevance", "sufficient"], "query": "where is paris ?", "passages": PASSAGES},
             {"decisions": [tinyjev.Q("rate the tone ?", ["formal", "informal"])], "header": "paris ."}]
    many = d.decide_many(items)
    one = d.run(["relevance", "sufficient"], query="where is paris ?", passages=PASSAGES)
    assert max(abs(many[0]["sufficient"][k] - one["sufficient"][k]) for k in one["sufficient"]) < 1e-4
    assert set(many[1]["custom_0"]) == {"formal", "informal"}


def test_nano_compatible_and_errors(d):
    assert len(d.relevance("q ?", PASSAGES)) == 3
    assert set(d.sufficient("q ?", PASSAGES)) == {"yes", "no"}
    assert set(d.grounded("penguins cannot fly .", "penguins live in antarctica .")) == {"yes", "no"}
    with pytest.raises(ValueError):
        d.run(["nonexistent"], query="q")
    with pytest.raises(ValueError):
        d.run(["relevance"], query="q")          # no passages
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.session(query="q", passages=PASSAGES)  # no truncation warning at this size
