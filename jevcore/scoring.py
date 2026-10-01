"""Run a model over packs and return per-group logits (shared by train / evaluate / Decider)."""

import time

import torch

from .collate import collate, make_rows, render_all, to_device, token_batches
from .packing import PackConfig


def autocast(device, enabled: bool = True):
    dev = torch.device(device).type
    return torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=enabled and dev == "cuda")


@torch.no_grad()
def score_packs(model, tok, M, packs: list[dict], cfg: PackConfig, device,
                tokens_per_batch: int = 16384, row_packing: bool = True, bf16: bool = True,
                timings: list | None = None) -> list[dict]:
    """Returns one dict per group:
        {pack, name, dec, seg, options, logits (1-D CPU float, length K), label}
    in no particular order. Packs that don't fit in cfg.max_len are skipped with a warning.
    If `timings` is a list, the forward time (s) of each batch is appended to it.
    """
    model.eval()
    ids = [p["id"] for p in packs]
    by_id = dict(zip(ids, packs))
    rendered, kept = render_all(packs, tok, M, cfg, ids=ids)
    rows = make_rows(rendered, cfg.max_len, row_packing, ids=kept)
    with_spans = model.cfg.get("readout") == "mean"
    out = []
    for rows_b in token_batches(rows, tokens_per_batch):
        b = collate(rows_b, tok.pad_token_id, cfg, with_spans)
        t0 = time.perf_counter()
        with autocast(device, bf16):
            z = model(**to_device(b, device))
        if timings is not None:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
        z = z.float().cpu()
        for n in range(len(b["names"])):
            pack = by_id[b["g_example"][n]]
            d = pack["decisions"][b["g_dec"][n]]
            k = len(d["options"])
            out.append({"pack": pack["id"], "name": d["name"], "dec": b["g_dec"][n],
                        "seg": b["g_seg"][n], "options": d["options"], "logits": z[n, :k],
                        "label": int(b["labels"][n])})
    return out
