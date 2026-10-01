import pytest
import torch
from conftest import sample_example

from jevcore.packing import (REF_BASE, PackConfig, PackOverflow, add_markers, allowed, local_from_global,
                             pack_row, render, truncate_state)


@pytest.fixture
def M(tok):
    return add_markers(tok)


def masks_for(rows, cfg=PackConfig()):
    from jevcore.collate import collate
    return collate(rows, 0, cfg)


def test_render_layout(tok, M):
    ex = sample_example()
    r = render(ex, tok, M, PackConfig(max_len=512))
    S = r.state_len
    assert r.ids[0] == tok.cls_token_id and r.ids[1] == M["state"]
    assert r.ids[S - 1] == tok.sep_token_id
    assert r.pos[:S] == list(range(S))
    assert all(b == 0 for b in r.blk[:S]) and all(b > 0 for b in r.blk[S:])
    # 3 relevance groups + 3 global groups
    assert [g.name for g in r.groups] == ["relevance"] * 3 + ["sufficient", "grounded", "custom_0"]
    for g in r.groups:
        assert r.ids[g.anchor] == (M["ref"] if g.name == "relevance" else M["q"])
        assert all(r.ids[o] == M["opt"] for o in g.options)
        # every decision block restarts at S; isolated options all start at S + Lq
        q_start = r.blk.index(r.blk[g.anchor])
        assert r.pos[q_start] == S
        assert len({r.pos[o] for o in g.options}) == 1
        if g.name == "relevance":
            assert r.pos[g.anchor] == r.pos[g.options[0]]
            assert r.prt[g.anchor] == REF_BASE + g.seg
    assert [g.seg for g in r.groups[:3]] == [0, 1, 2]
    assert [g.label for g in r.groups] == [2, 0, 0, 0, 0, 2]


def test_siblings_positions_are_sequential(tok, M):
    r = render(sample_example(), tok, M, PackConfig(max_len=512, isolated=False))
    g = r.groups[-1]
    starts = [r.pos[o] for o in g.options]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_no_restart_is_sequential(tok, M):
    r = render(sample_example(), tok, M, PackConfig(max_len=512, position_restart=False))
    assert r.pos == list(range(len(r)))


def _row_tensors(row):
    t = lambda x: torch.tensor(x)  # noqa: E731
    return t(row.ex), t(row.blk), t(row.prt), torch.ones(len(row), dtype=torch.bool)


def test_mask_rules(tok, M):
    r = render(sample_example(), tok, M, PackConfig(max_len=512))
    row = pack_row([r])
    ex, blk, prt, valid = _row_tensors(row)
    m = allowed(ex, blk, prt, valid)
    S = r.state_len
    st = blk == 0
    # the state never sees decisions; state tokens see the whole state
    assert not m[:S, S:].any() and m[:S, :S].all()
    # decision tokens never see other decisions
    for d in blk.unique()[1:]:
        q_in = blk == d
        others = (blk != d) & ~st
        assert not m[q_in][:, others].any()
    # options never see sibling options; question span sees no options
    g = r.groups[-1]
    o_prt = [prt[o].item() for o in g.options]
    for j, pj in enumerate(o_prt):
        rows_j = (blk == blk[g.anchor]) & (prt == pj)
        for pk in o_prt[:j] + o_prt[j + 1:]:
            assert not m[rows_j][:, (blk == blk[g.anchor]) & (prt == pk)].any()
    q_span = (blk == blk[g.anchor]) & (prt == 0)
    assert not m[q_span][:, (blk == blk[g.anchor]) & (prt > 0)].any()
    # <ref> for segment i sees the header + segment i, not other segments
    for gr in r.groups[:3]:
        row_ref = m[gr.anchor]
        assert row_ref[(blk == 0) & (prt == 0)].all()
        assert row_ref[(blk == 0) & (prt == gr.seg + 1)].all()
        assert not row_ref[(blk == 0) & (prt > 0) & (prt != gr.seg + 1)].any()


def test_siblings_mask(tok, M):
    r = render(sample_example(), tok, M, PackConfig(max_len=512, isolated=False))
    ex, blk, prt, valid = _row_tensors(pack_row([r]))
    m = allowed(ex, blk, prt, valid, isolated=False)
    g = r.groups[-1]
    a, b = g.options[0], g.options[1]
    assert m[a, b] and m[b, a]
    assert not m[g.anchor, a]          # question span still sees no options


def test_padding_and_examples(tok, M):
    cfg = PackConfig(max_len=512)
    r1, r2 = render(sample_example(), tok, M, cfg), render(sample_example(2), tok, M, cfg)
    row = pack_row([r1, r2], ["a", "b"])
    assert row.examples == ["a", "b"]
    n_blocks = lambda r: len(set(r.blk) - {0})  # noqa: E731
    assert len(set(b for b in row.blk if b)) == n_blocks(r1) + n_blocks(r2)
    assert row.pos[len(r1)] == 0                                   # positions restart per example
    from jevcore.collate import collate
    batch = collate([row, pack_row([r2])], tok.pad_token_id, cfg)
    full = batch["full_mask"]
    n1 = len(r1)
    assert not full[0, :n1, n1:len(row)].any() and not full[0, n1:len(row), :n1].any()
    # padding rows attend only to themselves (no all-masked rows -> no NaN in SDPA)
    pad = ~batch["valid"][1]
    assert pad.any()
    assert (full[1][pad].sum(-1) == 1).all() and full.any(-1).all()
    assert batch["g_anchor"].shape[0] == len(r1.groups) + 2 * len(r2.groups)


def test_local_window():
    blk = torch.tensor([0] * 40 + [1] * 5)
    m = torch.ones(45, 45, dtype=torch.bool)
    loc = local_from_global(m, blk, half_window=8)
    assert loc[0, 8] and not loc[0, 9]           # state keeps the window
    assert loc[44, 0]                            # decision tokens stay global
    assert not local_from_global(m, blk, 8, decision_global=False)[44, 0]


def test_truncation_never_touches_decisions(tok, M):
    ex = sample_example(n_seg=3, long=True)
    full = render(ex, tok, M, PackConfig(max_len=4096))
    small = render(ex, tok, M, PackConfig(max_len=len(full) - 40))
    assert small.truncated and len(small) <= len(full) - 40
    assert len(small) - small.state_len == len(full) - full.state_len


def test_truncate_equal_cap_and_drop():
    header = list(range(300))
    segs = [[1] * 100, [2] * 50, [3] * 10]
    h, out, trunc = truncate_state(header, segs, {0, 1, 2}, 256 + 120)
    assert len(h) == 256 and trunc
    assert sum(len(s) for _, s in out) <= 120
    assert [len(s) for _, s in out] == [60, 50, 10]
    # untargeted segments are dropped before giving up
    h, out, _ = truncate_state([], [[1] * 100] * 4, {1}, 40)
    assert [i for i, _ in out] == [1]
    with pytest.raises(PackOverflow):
        truncate_state([], [[1] * 100] * 4, {0, 1, 2}, 40)


def test_overflow_when_decisions_too_long(tok, M):
    with pytest.raises(PackOverflow):
        render(sample_example(), tok, M, PackConfig(max_len=20))
