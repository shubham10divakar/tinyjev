import torch
from conftest import sample_example, tiny_model

from jevcore.backbones.modernbert import MicroJev
from jevcore.packing import PackConfig
from jevcore.scoring import score_packs
from jevcore.trainer import TrainConfig, param_groups, train


def _packs(n=8):
    out = []
    for i in range(n):
        e = sample_example(n_seg=1 + i % 4)
        e["id"] = f"p{i}"
        e["state"]["header"] += f" {i}"          # distinct inputs, so the labels are learnable
        rel = e["decisions"][0]
        rel["labels"] = [(i + k) % 3 for k in range(len(rel["targets"]))]
        e["decisions"][1]["label"] = i % 2
        out.append(e)
    return out


def test_param_groups_split():
    model, tok, M = tiny_model()
    model.add_marker_delta(M)
    groups = {g["name"]: g for g in param_groups(model, TrainConfig())}
    n_all = sum(1 for p in model.parameters() if p.requires_grad)
    assert sum(len(g["params"]) for g in groups.values()) == n_all
    assert any(p is model.marker_delta for p in groups["head_plain"]["params"])
    assert groups["head_decay"]["lr"] == 5e-4 and groups["bb_decay"]["lr"] == 5e-5
    emb = model.enc.get_input_embeddings().weight
    assert any(p is emb for p in groups["bb_plain"]["params"])


def test_overfit_and_reload(tmp_path):
    model, tok, M = tiny_model()
    # back to the real zero-init last layer: training must start at uniform (NLL = log K)
    torch.nn.init.zeros_(model.head.mlp[-1].weight)
    torch.nn.init.zeros_(model.head.mlp[-1].bias)
    cfg = PackConfig(max_len=512, half_window=model.half_window)
    packs = _packs()
    tc = TrainConfig(lr_backbone=1e-3, lr_head=3e-3, epochs=60, packs_per_step=8,
                     tokens_per_microbatch=4096, bf16=False, grad_ckpt=False, warmup=0.0,
                     constant_lr=True, log_every=1000)
    hist = train(model, tok, M, packs, packs, cfg, tc, tmp_path / "run", "cpu", aug=None,
                 log=lambda *_: None)
    first, last = hist["dev"][0]["macro"], hist["dev"][-1]["macro"]
    assert abs(hist["dev"][0]["sufficient"] - torch.log(torch.tensor(2.0))) < 1e-4
    assert last < 0.1 < first, (first, last)
    assert torch.any(model.marker_delta != 0)                 # markers trained via the delta

    # reload: marker delta folded into the embedding, same logits as in memory
    model.save(tmp_path / "final", tok, {"epochs_trained": tc.epochs})
    loaded, tok2, M2, saved = MicroJev.load(tmp_path / "final")
    assert M2 == M and saved["epochs_trained"] == tc.epochs
    a = score_packs(model, tok, M, packs, cfg, "cpu", bf16=False)
    b = score_packs(loaded, tok2, M2, packs, cfg, "cpu", bf16=False)
    diff = max((x["logits"] - y["logits"]).abs().max().item() for x, y in zip(a, b))
    assert diff < 1e-4
    assert (tmp_path / "run" / "train_log.jsonl").exists()
