import random

import torch
from conftest import sample_example

from jevcore.collate import collate, epoch_batches, eval_batches, make_rows, token_batches
from jevcore.packing import PackConfig, add_markers, render


def _examples(n):
    out = []
    for i in range(n):
        e = sample_example(n_seg=1 + i % 5, long=i % 3 == 0)
        e["id"] = f"e{i}"
        out.append(e)
    return out


def test_make_rows_respects_max_len(tok):
    M = add_markers(tok)
    cfg = PackConfig(max_len=400)
    rendered = [render(e, tok, M, cfg) for e in _examples(20)]
    rows = make_rows(rendered, cfg.max_len, ids=[f"e{i}" for i in range(20)])
    assert all(len(r) <= cfg.max_len for r in rows)
    assert len(rows) < 20                                   # short examples share rows
    assert sorted(x for r in rows for x in r.examples) == sorted(f"e{i}" for i in range(20))
    assert len(make_rows(rendered, cfg.max_len, row_packing=False)) == 20


def test_token_batches_budget(tok):
    M = add_markers(tok)
    cfg = PackConfig(max_len=400)
    rows = make_rows([render(e, tok, M, cfg) for e in _examples(30)], 400, row_packing=False)
    for b in token_batches(rows, 1200):
        assert len(b) == 1 or len(b) * max(map(len, b)) <= 1200


def test_epoch_covers_all_groups(tok):
    M = add_markers(tok)
    cfg = PackConfig(max_len=512)
    exs = _examples(12)
    n_groups = sum(len(render(e, tok, M, cfg).groups) for e in exs)
    got = sum(len(b["names"]) for b in epoch_batches(exs, tok, M, cfg, random.Random(0), 2048,
                                                    mega=5))
    assert got == n_groups
    got_eval = [b for b in eval_batches(exs, tok, M, cfg, 2048)]
    assert sum(len(b["names"]) for b in got_eval) == n_groups
    seen = {(b["g_example"][i], b["g_dec"][i], b["g_seg"][i]) for b in got_eval
            for i in range(len(b["names"]))}
    assert len(seen) == n_groups


def test_collate_spans(tok):
    M = add_markers(tok)
    cfg = PackConfig(max_len=512)
    r = render(sample_example(), tok, M, cfg)
    from jevcore.packing import pack_row
    b = collate([pack_row([r])], tok.pad_token_id, cfg, with_spans=True)
    g = r.groups[-1]
    first = b["g_span"][-1, 0][b["g_span_valid"][-1, 0]]
    assert first.tolist() == list(range(*g.spans[0]))
    assert b["input_ids"][0, first[0]] == M["opt"]
    assert b["g_opt_valid"].sum(-1).tolist() == [3, 3, 3, 2, 2, 4]
    assert torch.equal(b["labels"], torch.tensor([2, 0, 0, 0, 0, 2]))
