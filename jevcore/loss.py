"""Training loss (design §5.3), shared with Tiny-Jev.

Cross-entropy per group, averaged per decision name, then a weighted mean over the names
present, so ten relevance groups in a pack don't drown out one `sufficient`.
No label smoothing and no class weights: both distort calibration.
"""

import torch
import torch.nn.functional as F

from .schema import IGNORE


def packed_loss(z: torch.Tensor, labels: torch.Tensor, names: list[str],
                weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """z [G, Kmax] (-inf at padded options), labels [G]. Returns (loss, {name: mean CE})."""
    labels = labels.to(z.device)
    ce = F.cross_entropy(z, labels.clamp(min=0), reduction="none")
    keep = labels != IGNORE
    total, wsum, parts = z.new_zeros(()), 0.0, {}
    for n in dict.fromkeys(names):
        sel = torch.tensor([x == n for x in names], device=z.device) & keep
        if not sel.any():
            continue
        w = (weights or {}).get(n, 1.0)
        m = ce[sel].mean()
        total = total + w * m
        wsum += w
        parts[n] = float(m.detach())
    if wsum == 0:
        return z.sum() * 0.0, parts     # keeps the graph alive for an all-unlabelled batch
    return total / wsum, parts
