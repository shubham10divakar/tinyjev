# Tiny-Jev — working notes

Newest first. Decisions, deviations from the design doc, and things to check later.

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
  (copied from microjev @ `bd40d42`) instead of imported. Tiny-specific changes are made in
  separate commits so they can be upstreamed later. When `jevcore` becomes its own package,
  delete the copy here.
