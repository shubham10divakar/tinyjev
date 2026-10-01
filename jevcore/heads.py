"""Option-scoring heads (design §4.4), shared with Tiny-Jev."""

import torch
from torch import nn


class PairScorer(nn.Module):
    """logit(anchor, option) = MLP(Wa·a + Wo·o + (Wa·a) ⊙ (Wo·o)).

    The last layer starts at zero, so the model starts at uniform probabilities (NLL = log K).
    """

    def __init__(self, d: int, h: int | None = None, p: float = 0.1):
        super().__init__()
        h = h or d
        self.wa, self.wo = nn.Linear(d, h), nn.Linear(d, h)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, a: torch.Tensor, o: torch.Tensor) -> torch.Tensor:  # a [G, d], o [G, K, d]
        ha, ho = self.wa(a)[:, None], self.wo(o)
        return self.mlp(ha + ho + ha * ho).squeeze(-1)


class LinearScorer(nn.Module):
    """Ablation A5: Nano-style linear(o); the anchor is ignored."""

    def __init__(self, d: int, p: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.lin = nn.Linear(d, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, a: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return self.lin(self.drop(o)).squeeze(-1)


def build_head(kind: str, d: int, p: float = 0.1) -> nn.Module:
    if kind == "pair":
        return PairScorer(d, p=p)
    if kind == "linear":
        return LinearScorer(d, p=p)
    raise ValueError(f"unknown head {kind!r}")
