"""Batching (design §5.5): row packing, length bucketing, group flattening, batch masks."""

import random
import warnings
from collections.abc import Callable, Iterator

import torch

from .packing import PackConfig, PackOverflow, Rendered, Row, pack_row, render, row_masks


def make_rows(rendered: list[Rendered], max_len: int, row_packing: bool = True,
              ids=None) -> list[Row]:
    """Put rendered examples into rows. With row_packing, short examples share a row
    (first-fit decreasing); `ex` keeps them apart in the mask, positions restart per example."""
    ids = list(range(len(rendered))) if ids is None else list(ids)
    if not row_packing:
        return [pack_row([r], [i]) for r, i in zip(rendered, ids)]
    order = sorted(range(len(rendered)), key=lambda k: -len(rendered[k]))
    bins: list[tuple[int, list[int]]] = []          # (used tokens, member positions)
    for k in order:
        n = len(rendered[k])
        for b, (used, members) in enumerate(bins):
            if used + n <= max_len:
                bins[b] = (used + n, members + [k])
                break
        else:
            bins.append((n, [k]))
    return [pack_row([rendered[k] for k in members], [ids[k] for k in members])
            for _, members in bins]


def collate(rows: list[Row], pad_id: int, cfg: PackConfig, with_spans: bool = False,
            pad_to: int = 8) -> dict:
    """Pad rows into one batch with bool masks [B, T, T] and flattened groups."""
    B = len(rows)
    T = max(len(r) for r in rows)
    T = -(-T // pad_to) * pad_to
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    pos = torch.zeros((B, T), dtype=torch.long)
    ex = torch.full((B, T), -1, dtype=torch.long)
    blk = torch.zeros((B, T), dtype=torch.long)
    prt = torch.zeros((B, T), dtype=torch.long)
    valid = torch.zeros((B, T), dtype=torch.bool)
    for b, r in enumerate(rows):
        n = len(r)
        ids[b, :n] = torch.tensor(r.ids)
        pos[b, :n] = torch.tensor(r.pos)
        ex[b, :n] = torch.tensor(r.ex)
        blk[b, :n] = torch.tensor(r.blk)
        prt[b, :n] = torch.tensor(r.prt)
        valid[b, :n] = True
    full = torch.empty((B, T, T), dtype=torch.bool)
    local = torch.empty((B, T, T), dtype=torch.bool)
    for b in range(B):
        full[b], local[b] = row_masks(ex[b], blk[b], prt[b], valid[b], cfg)

    groups = [(b, g) for b, r in enumerate(rows) for g in r.groups]
    G = len(groups)
    K = max((len(g.options) for _, g in groups), default=1)
    g_opts = torch.zeros((G, K), dtype=torch.long)
    g_valid = torch.zeros((G, K), dtype=torch.bool)
    for n, (_, g) in enumerate(groups):
        g_opts[n, : len(g.options)] = torch.tensor(g.options)
        g_valid[n, : len(g.options)] = True
    batch = {
        "input_ids": ids, "position_ids": pos, "full_mask": full, "local_mask": local,
        "valid": valid,
        "g_batch": torch.tensor([b for b, _ in groups], dtype=torch.long),
        "g_anchor": torch.tensor([g.anchor for _, g in groups], dtype=torch.long),
        "g_opts": g_opts, "g_opt_valid": g_valid,
        "labels": torch.tensor([g.label for _, g in groups], dtype=torch.long),
        "names": [g.name for _, g in groups],
        # bookkeeping to map logits back to (example, decision, segment)
        "g_example": [rows[b].examples[g.ex] for b, g in groups],
        "g_dec": [g.dec for _, g in groups],
        "g_seg": [g.seg for _, g in groups],
    }
    if with_spans:   # A4 mean readout
        L = max((e - s for _, g in groups for s, e in g.spans), default=1)
        span = torch.zeros((G, K, L), dtype=torch.long)
        span_valid = torch.zeros((G, K, L), dtype=torch.bool)
        for n, (_, g) in enumerate(groups):
            for j, (s, e) in enumerate(g.spans):
                span[n, j, : e - s] = torch.arange(s, e)
                span_valid[n, j, : e - s] = True
        batch["g_span"], batch["g_span_valid"] = span, span_valid
    return batch


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def token_batches(rows: list[Row], tokens_per_batch: int, max_rows: int = 64) -> list[list[Row]]:
    """Group rows of similar length so that B * T_max <= tokens_per_batch."""
    rows = sorted(rows, key=len)
    batches, cur = [], []
    for r in rows:
        T = max([len(r)] + [len(x) for x in cur])
        if cur and ((len(cur) + 1) * T > tokens_per_batch or len(cur) >= max_rows):
            batches.append(cur)
            cur = []
        cur.append(r)
    if cur:
        batches.append(cur)
    return batches


def render_all(examples: list[dict], tok, M, cfg: PackConfig, ids=None, warn=True):
    """Render examples, skipping ones that can't fit. Returns (rendered, kept ids)."""
    ids = list(range(len(examples))) if ids is None else list(ids)
    out, kept, skipped = [], [], 0
    for i, ex in zip(ids, examples):
        try:
            out.append(render(ex, tok, M, cfg))
            kept.append(i)
        except PackOverflow:
            skipped += 1
    if skipped and warn:
        warnings.warn(f"skipped {skipped} examples that do not fit in {cfg.max_len} tokens")
    return out, kept


def epoch_batches(examples: list[dict], tok, M, cfg: PackConfig, rng: random.Random,
                  tokens_per_batch: int, augment: Callable[[dict, random.Random], dict] | None = None,
                  mega: int = 512, row_packing: bool = True,
                  with_spans: bool = False) -> Iterator[dict]:
    """One training epoch: shuffle, augment, render and pack in mega-batches, bucket by length,
    shuffle the batches inside each mega-batch."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    for s in range(0, len(order), mega):
        chunk = order[s: s + mega]
        exs = [augment(examples[i], rng) if augment else examples[i] for i in chunk]
        rendered, kept = render_all(exs, tok, M, cfg, ids=chunk, warn=False)
        rows = make_rows(rendered, cfg.max_len, row_packing, ids=kept)
        batches = token_batches(rows, tokens_per_batch)
        rng.shuffle(batches)
        for b in batches:
            yield collate(b, tok.pad_token_id, cfg, with_spans)


def eval_batches(examples: list[dict], tok, M, cfg: PackConfig, tokens_per_batch: int,
                 row_packing: bool = True, with_spans: bool = False) -> Iterator[dict]:
    """Deterministic batches over all examples (no augmentation)."""
    rendered, kept = render_all(examples, tok, M, cfg)
    rows = make_rows(rendered, cfg.max_len, row_packing, ids=kept)
    for b in token_batches(rows, tokens_per_batch):
        yield collate(b, tok.pad_token_id, cfg, with_spans)
