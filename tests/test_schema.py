import pytest
from conftest import sample_example

from jevcore.schema import DECISIONS, from_nano_row, labeled_groups, validate


def test_sample_is_valid():
    validate(sample_example())


@pytest.mark.parametrize("breaker,msg", [
    (lambda e: e.update(decisions=[]), "at least one decision"),
    (lambda e: e["decisions"][0].update(targets=[0, 9]), "target out of range"),
    (lambda e: e["decisions"][0].update(labels=[0]), "differ in length"),
    (lambda e: e["decisions"][0].update(labels=[3, 0, 0]), "label out of range"),
    (lambda e: e["decisions"][1].update(label=2), "label out of range"),
    (lambda e: e["decisions"][1].update(options=["yes"]), "at least 2 options"),
    (lambda e: e["decisions"][1].update(scope="row"), "scope"),
    (lambda e: e["decisions"][0].update(targets=[0, 0, 1]), "duplicate"),
])
def test_validate_rejects(breaker, msg):
    ex = sample_example()
    breaker(ex)
    with pytest.raises(ValueError, match=msg):
        validate(ex)


def test_unlabeled_is_valid():
    ex = sample_example()
    ex["decisions"][1]["label"] = -100
    del ex["decisions"][0]["labels"]
    validate(ex)


def test_from_nano_row():
    row = {"decision": "grounded", "question": "Is this claim supported by the context? Claim: x",
           "options": ["yes", "no"], "state": "some premise", "label": 1, "source": "mnli"}
    ex = from_nano_row(row)
    validate(ex)
    assert ex["state"] == {"header": "some premise", "segments": []}
    assert ex["decisions"][0]["scope"] == "global"
    assert list(labeled_groups(ex)) == [("grounded", -1, 1)]


def test_groups_order():
    groups = list(labeled_groups(sample_example()))
    assert [g[0] for g in groups] == ["relevance"] * 3 + ["sufficient", "grounded", "custom_0"]
    assert groups[0] == ("relevance", 0, 2)


def test_builtin_scopes():
    assert DECISIONS["relevance"].scope == "segment"
    assert all(d.scope == "global" for n, d in DECISIONS.items() if n != "relevance")
