"""Causal (Tiny-Jev) rendering and masks: design §3, §4.4."""

import torch
from conftest import sample_example

from jevcore.packing import (MARKERS, PackConfig, add_markers, allowed, append_decisions,
                             decision_blocks, pack_row, render, render_state, state_overhead,
                             _encoder)

TINY = tuple(MARKERS)       # all 8 markers, incl. <qe> / <oe>


def _setup(tok, sink=True):
    M = add_markers(tok, TINY)
    cfg = PackConfig(max_len=512, causal=True, sink_id=tok.pad_token_id if sink else None)
    return M, cfg


def _arrays(r, e=0):
    t = lambda x: torch.tensor(x)  # noqa: E731
    return t([e] * len(r)), t(r.blk), t(r.prt), torch.ones(len(r), dtype=torch.bool)


def test_causal_render_layout(tok):
    M, cfg = _setup(tok)
    r = render(sample_example(), tok, M, cfg)
    assert r.ids[0] == cfg.sink_id and r.ids[1] == M["state"]
    assert tok.cls_token_id not in r.ids and tok.sep_token_id not in r.ids
    for g in r.groups:
        assert r.ids[g.anchor] in (M["qe"], M["ref"])
        assert all(r.ids[o] == M["oe"] for o in g.options)
        # end-marker readout: the option's last token, the span ends right after it
        assert [e - 1 for _, e in g.spans] == g.options
    # every option of a decision restarts at the same position (S + Lq)
    g = r.groups[-1]
    assert len({r.pos[s] for s, _ in g.spans}) == 1


def test_overhead_matches_render(tok):
    for sink in (True, False):
        M, cfg = _setup(tok, sink)
        st = render_state({"header": "", "segments": []}, tok, M, cfg, budget=100)
        assert len(st) == state_overhead(cfg)


def test_render_is_state_plus_decisions(tok):
    M, cfg = _setup(tok)
    ex = sample_example()
    full = render(ex, tok, M, cfg)
    enc = _encoder(tok)
    blocks, _ = decision_blocks(ex["decisions"], enc, M, True)
    st = render_state(ex["state"], tok, M, cfg, budget=10_000)
    two = append_decisions(st, ex["decisions"], blocks, M, cfg)
    assert (two.ids, two.pos, two.blk, two.prt) == (full.ids, full.pos, full.blk, full.prt)
    assert [g.anchor for g in two.groups] == [g.anchor for g in full.groups]


def test_allowed_row_slice_equals_full_rows(tok):
    M, cfg = _setup(tok)
    r = pack_row([render(sample_example(), tok, M, cfg)])
    ex, blk, prt, valid = _arrays(r)
    full = allowed(ex, blk, prt, valid, causal=True)
    S = sum(1 for b in r.blk if b == 0)
    part = allowed(ex, blk, prt, valid, causal=True, q_start=S)
    assert part.shape == (len(r) - S, len(r))
    assert torch.equal(part, full[S:])


def test_causal_mask_properties(tok):
    M, cfg = _setup(tok)
    r = pack_row([render(sample_example(), tok, M, cfg)])
    ex, blk, prt, valid = _arrays(r)
    m = allowed(ex, blk, prt, valid, causal=True)
    assert not torch.triu(m, 1).any()                       # nothing attends forward
    S = sum(1 for b in r.blk if b == 0)
    assert m[:S, :S].equal(torch.tril(torch.ones(S, S, dtype=torch.bool)))   # plain causal state
    g = r.groups[-1]                                         # options don't see each other
    (s0, e0), (s1, e1) = g.spans[:2]
    assert not m[s1:e1, s0:e0].any()
