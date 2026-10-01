# Tiny-Jev — working notes

Newest first. Decisions, deviations from the design doc, and things to check later.

## 2026-10-01 — model, Session and M0 tests on random weights (steps 1–6)

M0 on a tiny random Qwen3 (3 layers, d 64, GQA 4/2 heads, CPU, fp32), `tests/test_tiny_invariance.py`:

| Test | Result (max \|Δp\|) |
|---|---|
| T-A packed vs alone / reversed / options permuted / extra decision / row-packed / padded batch | ~1e-6 (sdpa, eager, LoRA, no sink, mean readout, linear head); bf16 ≤ 1e-2 |
| T-A teeth: no block mask + no restart | > 1e-3 (the probes detect leaks) |
| T-B training forward (mixed batch, row-packed) vs each decision alone | ~1e-6 |
| T-C `Session.decide` vs one-pass training forward | 0.0 (cache path bit-identical on CPU) |
| T-C teeth: zeroing the cached values changes the logits | yes |
| T-D `extend` ×2 vs fresh session on the full state | ~1e-6, identical ids / positions |
| T-E cache length after decide / extend | = S exactly |

Decisions / deviations:
- **Rendering split** (`packing.py`): `render` = `render_state` + `append_decisions`. The Session
  prefills exactly `render_state`'s tokens, so the cache and training paths share one renderer.
  `allowed(..., q_start)` returns only the new tokens' rows: `allowed_with_state` costs
  O(Td·(S+Td)), not O((S+Td)²).
- **Sink token:** `<|endoftext|>` (falls back to BOS / pad) before `<state>`, on by default
  (`sink_token: true`). Overhead of the causal state is now 1 + sink (was a fixed 2).
- **Session never drops segments** (`drop_untargeted=False`): it caps per-segment length with a
  warning if the state is too long, and `extend` raises `PackOverflow` and leaves the session
  untouched.
- **Markers:** ids are added to the tokenizer but the embedding is *not* resized; `embed()`
  clamps ids into the table and `MarkerEmbedding` (fp32, 8 × d) substitutes the vectors. Init =
  mean of the marker words' embeddings + 0.02·std noise (so `<opt>`/`<oe>` and `<seg>`/`<ref>`,
  which share words, start apart).
- **Head in fp32 with autocast disabled** inside `score()`, so bf16 autocast in training doesn't
  downcast it.
- **A1 (`mask: full`) on the causal model** = tril of the no-isolation mask (`row_masks`).
- Sliding-window layers are rejected at construction (Qwen3-0.6B has none; confirm in M0).
- Save: LoRA adapter (`adapter/`) + `markers.pt` + `head.pt` + tokenizer + `tinyjev_config.json`;
  the base comes from the Hub id. `load(merge=True)` folds LoRA in for inference (outputs
  within 1e-4 in tests). Random test bases are saved too (`backbone/`, via an unloaded copy).
- Bugs caught by tests: saving the LoRA-injected base wrote `base_layer` keys (fixed by
  `deepcopy(...).unload()`).

## 2026-10-01 — start

- Repo: `code_repo/tinyjev/` with its own `git init`, remote
  `github.com/shubham10divakar/tinyjev`. Added `tinyjev/` to `code_repo/.git/info/exclude` so
  the outer nano-jev repo ignores it (same as microjev / cyber_jev).
- GPU busy: implement only. Tests use a tiny random Qwen3 config on CPU.
- Environment: shared `code_repo/.venv` has transformers 5.17.0 (the version the design checked)
  and torch 2.11 cu128. **Installed `peft` 0.21.1** (uv) for LoRA; nothing else changed.
- Checked in the installed transformers 5.17.0 source (`models/qwen3/modeling_qwen3.py`,
  `integrations/sdpa_attention.py`, `cache_utils.py`):
  - `Qwen3Model.forward` uses a dict `attention_mask` as-is (keys = layer types); it only builds
    its own causal mask when the mask is *not* a dict. ✔ §2
  - SDPA passes our mask straight to `F.scaled_dot_product_attention`; a bool mask means
    True = attend, and `is_causal` is off whenever a mask is given. Eager *adds* the mask, so
    eager needs a float mask (same as Micro-Jev's `_mask`).
  - `position_ids` drive RoPE directly, also with a cache (the cache only appends keys / values).
  - `DynamicLayer.crop(n)`: negative n removes the last |n| tokens; a positive n is the legacy
    "absolute length" form, deprecated, removed in 5.18. Always call `crop(-Td)` (risk in §11).
- **Shared code:** Micro-Jev has no remote and isn't a package yet, so `jevcore` is **vendored**
  (copied from microjev @ `7c630a8`; microjev is being developed in parallel, so re-sync deliberately and note it here) instead of imported. Tiny-specific changes are made in
  separate commits so they can be upstreamed later. When `jevcore` becomes its own package,
  delete the copy here.
