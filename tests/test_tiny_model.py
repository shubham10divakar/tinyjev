"""TinyJev model plumbing: markers, trainable parts (§4.3), save / load / merge, gradients."""

import pytest
import torch
from conftest import TINY_LORA, sample_example, tiny_qwen

from jevcore.backbones.qwen3 import TINY_MARKERS, MarkerEmbedding, Session, TinyJev
from jevcore.collate import collate
from jevcore.invariance import max_diff, probs_of
from jevcore.loss import packed_loss
from jevcore.packing import pack_row, render


def test_marker_embedding_substitutes_only_markers():
    me = MarkerEmbedding([10, 11], torch.arange(6.0).view(2, 3))
    ids = torch.tensor([[1, 10, 2, 11]])
    base = torch.full((1, 4, 3), -1.0)
    out = me(ids, base)
    assert out[0, 0].tolist() == [-1, -1, -1] and out[0, 2].tolist() == [-1, -1, -1]
    assert out[0, 1].tolist() == [0, 1, 2] and out[0, 3].tolist() == [3, 4, 5]


def test_markers_and_sink():
    model, tok, M = tiny_qwen()
    assert set(M) == set(TINY_MARKERS)
    assert model.markers.table.shape == (8, 64)
    assert model.sink_id == tok.pad_token_id      # no <|endoftext|> in the test vocab
    r = render(sample_example(), tok, M, model.pack_config(1024))
    assert r.ids[0] == model.sink_id


def test_only_lora_head_markers_train():
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    groups = model.trainable_groups()
    assert set(groups) == {"lora", "head", "markers"}
    assert all("lora_" in n for n, p in model.bb.named_parameters() if p.requires_grad)
    assert not model.bb.get_input_embeddings().weight.requires_grad
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train == sum(p.numel() for ps in groups.values() for p in ps)
    assert all(p.dtype == torch.float32 for p in model.head.parameters())


def test_full_finetune_freezes_embedding_only():
    model, tok, M = tiny_qwen(lora=None)
    assert "backbone" in model.trainable_groups()
    assert not model.bb.get_input_embeddings().weight.requires_grad
    assert model.bb.layers[0].self_attn.q_proj.weight.requires_grad


def test_training_step_gradients_reach_all_groups():
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    model.train()
    cfg = model.pack_config(1024)
    rows = [pack_row([render(sample_example(), tok, M, cfg)]),
            pack_row([render(sample_example(n_seg=1) | {"id": "s"}, tok, M, cfg)])]
    b = collate(rows, tok.pad_token_id, cfg)
    z = model(**b)
    loss, parts = packed_loss(z, b["labels"], b["names"])
    assert torch.isfinite(loss) and set(parts) >= {"relevance", "sufficient"}
    loss.backward()
    for name, ps in model.trainable_groups().items():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ps), name


def test_grad_ckpt_matches_plain():
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    model.train()
    cfg = model.pack_config(1024)
    b = collate([pack_row([render(sample_example(), tok, M, cfg)])], tok.pad_token_id, cfg)

    def grads():
        torch.manual_seed(0)                            # same head-dropout mask in both runs
        model.zero_grad()
        loss, _ = packed_loss(model(**b), b["labels"], b["names"])
        loss.backward()
        return loss.item(), model.markers.table.grad.clone()

    l0, g0 = grads()
    model.enable_grad_ckpt()
    l1, g1 = grads()
    assert abs(l0 - l1) < 1e-5 and torch.allclose(g0, g1, atol=1e-5)


@pytest.mark.parametrize("lora", [TINY_LORA, None])
def test_save_load_round_trip(tmp_path, lora):
    model, tok, M = tiny_qwen(lora=lora)
    with torch.no_grad():
        model.markers.table.add_(0.5)               # trained-looking markers must be restored
    cfg = model.pack_config(1024)
    ex = sample_example()
    ref = probs_of(model, tok, M, [ex], cfg)
    model.save(tmp_path / "m", tok, {"version": "test"})
    m2, tok2, M2, saved = TinyJev.load(tmp_path / "m")
    assert M2 == M and saved["version"] == "test" and m2.sink_id == model.sink_id
    assert max_diff(ref, probs_of(m2, tok2, M2, [ex], m2.pack_config(1024))) <= 1e-5


def test_merge_lora_keeps_outputs(tmp_path):
    model, tok, M = tiny_qwen(lora=TINY_LORA)
    cfg = model.pack_config(1024)
    ex = sample_example()
    ref = probs_of(model, tok, M, [ex], cfg)
    model.save(tmp_path / "m", tok)
    merged, tok2, M2, _ = TinyJev.load(tmp_path / "m", merge=True)
    assert not any("lora_" in n for n, _ in merged.bb.named_parameters())
    assert max_diff(ref, probs_of(merged, tok2, M2, [ex], cfg)) <= 1e-4
    s = Session(merged, tok2, cfg, ex["state"])          # cache path on the merged model
    got = {(ex["id"], g["name"], g["seg"]): dict(zip(g["options"], torch.softmax(g["logits"], 0).tolist()))
           for g in s.logits(ex["decisions"])}
    assert max_diff(ref, got) <= 1e-4
