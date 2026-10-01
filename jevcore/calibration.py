"""Temperature scaling and metrics (design §5.6, §6.2, §6.6).

fit_temperature / ece / the base metrics are copied from nanojev/calibration.py; the rest is
new for Micro-Jev: AUROC (binary decisions), quadratic weighted kappa (ordinal relevance),
selective prediction (AURC, accuracy with x% escalated) and the paired bootstrap.
"""

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single temperature T in [0.05, 20] minimising NLL of softmax(logits / T).

    Falls back to T = 1 if the fit does not improve calibration-set NLL.
    """
    logits, labels = logits.float(), labels.long()
    nll = lambda t: torch.nn.functional.cross_entropy(logits / t, labels)  # noqa: E731
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(log_t.clamp(-3.0, 3.0).exp())
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().clamp(-3.0, 3.0).exp())
    with torch.no_grad():
        return t if nll(t) < nll(1.0) else 1.0


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-label confidence."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def risk_coverage(probs: np.ndarray, labels: np.ndarray):
    """Sort by confidence (high first); return (coverage, risk) arrays."""
    conf = probs.max(1)
    order = np.argsort(-conf, kind="stable")
    wrong = (probs.argmax(1) != labels)[order].astype(float)
    n = np.arange(1, len(labels) + 1)
    return n / len(labels), np.cumsum(wrong) / n


def aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(probs, labels)
    return float(risk.mean())


def acc_escalated(probs: np.ndarray, labels: np.ndarray, frac: float) -> float:
    """Accuracy on the items kept after escalating the `frac` least confident ones."""
    conf = probs.max(1)
    keep = np.argsort(-conf, kind="stable")[: max(1, int(round(len(labels) * (1 - frac))))]
    return float((probs[keep].argmax(1) == labels[keep]).mean())


def metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0,
            ordinal: bool = False) -> dict:
    probs = torch.softmax(logits.float() / temperature, dim=1).numpy()
    y = labels.numpy()
    pred = probs.argmax(1)
    onehot = np.eye(probs.shape[1])[y]
    out = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "ece": ece(probs, y),
        "aurc": aurc(probs, y),
        "acc_escalated": {f"{int(f * 100)}%": acc_escalated(probs, y, f)
                          for f in (0.0, 0.1, 0.2, 0.3)},
    }
    if probs.shape[1] == 2 and len(set(y.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y == 0, probs[:, 0]))   # option 0 = "yes"-like
    if ordinal:
        out["qwk"] = float(cohen_kappa_score(y, pred, weights="quadratic"))
    return out


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, n: int = 10_000,
                     seed: int = 0) -> dict:
    """Accuracy difference a - b on identical items, with a 95% bootstrap CI (§6.6)."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}
