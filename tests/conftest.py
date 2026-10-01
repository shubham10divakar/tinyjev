"""Offline fixtures: a small word-level tokenizer and a tiny random ModernBERT.

No network: the real ModernBERT-base checks are marked `network` (TINYJEV_NETWORK_TESTS=1).
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevcore.schema import decision, make_state, query_header  # noqa: E402

CORPUS = """
query passage context decide question option relevant answer the a an of to is in on and for
which magazine was started first arthur's first women radio city published monthly in
how relevant this to do passages contain enough information answer yes no irrelevant
partially directly answers claim supported by the context is it true false sufficient
are there birds that cannot fly penguins ostriches emus kiwis flightless live antarctica
paris capital france tower eiffel built 1889 river seine city population million
time sensitive tone formal informal positive negative neutral rate quality low medium high
""".split()


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials)}
    for w in CORPUS + [str(i) for i in range(30)] + list(":?.,'[]()-"):
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(),
                                                pre_tokenizers.Punctuation()])
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def tiny_config(tok, local_attention=16, layers=3):
    from transformers import ModernBertConfig
    return ModernBertConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, global_attn_every_n_layers=3, local_attention=local_attention,
        max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
        cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id,
        bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id,
        # Larger than ModernBERT's 0.02 so random attention is peaked and options differ;
        # at 0.02 every <opt> marker gets ~the same hidden state and the tests see no signal.
        initializer_range=0.3)


def tiny_model(attn="sdpa", seed=0, **model_cfg):
    from jevcore.backbones.modernbert import MicroJev
    torch.manual_seed(seed)
    tok = tiny_tokenizer()
    model, tok, M = MicroJev.from_config(tiny_config(tok), tok, {"attn": attn, **model_cfg})
    # The head's last layer starts at zero (uniform output); randomise it so tests see signal.
    for p in model.head.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.5)
    return model.eval(), tok, M


@pytest.fixture
def tok():
    return tiny_tokenizer()


def sample_example(n_seg=3, long=False):
    words = ("penguins live in antarctica and cannot fly . " * (12 if long else 2)).strip()
    segs = [("Penguins", words), ("Paris", "paris is the capital of france ."),
            ("Eiffel", "the eiffel tower was built in 1889 on the river seine ."),
            ("Kiwis", "kiwis are flightless birds ."), ("City", "radio city .")][:n_seg]
    return {
        "id": "ex0", "source": "test",
        "state": make_state(query_header("are there birds that cannot fly ?"), segs),
        "decisions": [
            decision("relevance", targets=list(range(n_seg)), labels=[2] + [0] * (n_seg - 1)),
            decision("sufficient", label=0),
            decision("grounded", label=0, claim="penguins cannot fly ."),
            {"name": "custom_0", "kind": "choice", "scope": "global",
             "question": "rate the tone ?", "options": ["formal", "informal", "neutral", "positive"],
             "label": 2},
        ],
    }


def network_enabled():
    return os.environ.get("TINYJEV_NETWORK_TESTS") == "1"
