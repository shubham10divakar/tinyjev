"""LM-token readout baselines (B0 / B1 / B5) on a tiny random Qwen3 causal LM."""

import pytest
import torch
from conftest import TINY_LORA, sample_example, tiny_qwen_config, tiny_tokenizer

from jevcore.lm_readout import LETTERS, LetterReadout, prompt_items, score_prompts
from jevcore.loss import packed_loss


def _model(lora=None):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    tok.add_tokens(LETTERS)
    lm = AutoModelForCausalLM.from_config(tiny_qwen_config(tok), attn_implementation="sdpa")
    return LetterReadout.from_model(lm, tok, lora=lora, chat=False)


def test_prompt_items_unpacks_segments_and_skips_large():
    ex = sample_example()
    big = {"id": "big", "state": {"header": "x", "segments": []},
           "decisions": [{"name": "intent", "kind": "choice", "scope": "global", "question": "q ?",
                          "options": [f"o{i}" for i in range(40)], "label": 3}]}
    items, skipped = prompt_items([ex, big])
    assert skipped == 1
    rel = [it for it in items if it["name"] == "relevance"]
    assert len(rel) == 3 and all(it["content"].count("[1]") == 1 and "[2]" not in it["content"] for it in rel)
    assert "D. positive" in next(it for it in items if it["name"] == "custom_0")["content"]


def test_scores_shape_and_padding_invariance():
    m = _model().eval()
    ex = sample_example()
    batched = {(s["name"], s["seg"]): s["logits"] for s in score_prompts(m, [ex], "cpu", bf16=False)}
    alone = {}
    for it_ex in [ex]:
        items, _ = prompt_items([it_ex])
        for it in items:                       # one prompt per batch: no padding at all
            ids = m.encode_items([it])
            with torch.no_grad():
                z = m(**m.collate([it], ids))
            alone[(it["name"], it["seg"])] = z[0, : len(it["options"])]
    assert batched.keys() == alone.keys()
    for k in batched:
        assert torch.allclose(batched[k], alone[k], atol=1e-4), k
        assert len(batched[k]) == len(next(d for d in ex["decisions"] if d["name"] == k[0])["options"])


def test_b5_training_step_reaches_lora():
    m = _model(TINY_LORA)
    m.train()
    items, _ = prompt_items([sample_example()])
    b = m.collate(items, m.encode_items(items))
    z = m(**b)
    loss, _ = packed_loss(z, b["labels"], b["names"])
    loss.backward()
    grads = [p.grad for p in m.trainable_groups()["lora"]]
    assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)


class _SplitTok:
    """Tokenizes every letter into two ids, so the single-token check must fail."""
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [1, 2]}


def test_rejects_multi_token_letters():
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_config(tiny_qwen_config(tiny_tokenizer()))
    with pytest.raises(ValueError, match="not one token"):
        LetterReadout(lm, _SplitTok(), {})
