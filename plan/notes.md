# Tiny-Jev — working notes

Newest first. Decisions, deviations from the design doc, and things to check later.

## 2026-10-01 — API, baselines, trainer, eval, scripts (steps 10–13)

- **API** (`tinyjev`): `load()`, `d.session(query, passages)` → `decide([...])` / `extend()`,
  one-shot `d.decide(question, options, text)`, `d.run`, batched `d.decide_many` (no cache),
  Nano-compatible `relevance / sufficient / grounded`. Results are `Decision` objects: a plain
  `{option: p}` dict plus `.label`, `.confidence`, `.escalate`. `d.policy(tau)` takes a float
  or per-decision dict. `which_passage` is a built-in; `next_action` is not (needs P3): use `Q`.
- **B0 / B1 / B5 letter readout** (`lm_readout.py`): same prompt layout as Nano-Jev's LLM
  baseline (chat template, thinking off, letter logits at the answer slot, `logits_to_keep=1`).
  Left padding **with explicit position ids** (test: batched = alone). B1 is the HF
  `Qwen/Qwen3-4B-Instruct-2507` (as in Nano), not Ollama. Groups with > 26 options are
  skipped by the letter readout and counted. **B5 loss = CE over the K option letters**, not the
  full vocabulary, so H8 compares readouts with the same objective shape.
- **B4** = the same TinyJev and recipe on `unpacked_view` (one group per sequence; relevance
  sees header + its own passage), at train *and* eval time (`--baseline B4`, `--unpacked`).
- **T-A7 echo** = segments rendered twice in the state (`PackConfig.echo`); invariance and
  cache tests pass with it; `Session.extend` raises with echo (would need re-echoing).
- **Trainer** (`tiny_trainer.py`): param groups LoRA 2e-4 / head 1e-3 (wd 0.01 on matrices) /
  markers 1e-3; cosine with 3% warmup to 10%; accumulation by **tokens** (each micro-batch's
  loss weighted by its token share of `tokens_per_step`); eval every `eval_every` steps on the
  validation packs; best checkpoint by macro NLL. Total steps estimated from the mean rendered
  length of 300 sampled packs. Overfit test: macro val NLL halves in 40 steps on 6 packs.
- **Eval** (`tiny_eval.py`): ragged stacking (pad logit −1e4, finite so T fitting is NaN-free);
  T per training decision from `val_seen`, `T_custom` from `val_unseen`; held-out clusters and
  JevBench always use `T_custom`. P2 val draws are split into calib (`val_seen` / `val_unseen`)
  and in-domain `test` halves; evaluate adds the unseen-template view of `test`.
- **Scripts** — all with `--dry-run`, covered by `tests/test_scripts.py`:
  `m0_check` (config facts vs §2 table, probes on real weights with randomised head / LoRA-B,
  `--throughput`), `prepare_mixture` (`--check` for M1), `train` (ablations by name, `--drop-family`,
  `--overfit`, B4 / B5), `evaluate` (Tiny run, B0/B1 via `--lm`, B5 via `--b5`), `bench_latency`
  (S10 + loop), `cascade` (τ on val only + per-set oracle τ), `jevbench`.
- Windows console is cp1252: scripts reconfigure stdout to UTF-8 (`scripts/common.py`).
- Dry-run CPU latency on the tiny model (meaningless for the 3060, but the plumbing works):
  warm decide ≈ 2× faster than prefill + decide at S = 216 tokens.

## 2026-10-01 — data: tasks, mixture, synthetic, JevBench tooling (steps 7–9)

- **Templates live in `jevcore/tasks.yaml`, not `templates.yaml`** (design App. A names the
  latter). Micro-Jev's `templates.yaml` and tests stay untouched; Tiny's file holds every task
  (26: RAG ×4, verification ×3, yes/no, topic ×6, MCQ ×3, paraphrase, security ×3, format,
  held-out ×5) with 8 train + 2 held-out questions and [canonical, alternative] + 1 held-out
  option wording. LLM-drafted; **review by hand before M2 freezes them.**
- **No negated questions.** Two drafts ("Is anything missing…", "Is the language
  acceptable?") would flip yes / no labels when templates are swapped; replaced.
- **Dataset registry** (`data/tasks.py`): 24 training sources + 9 held-out, with HF id,
  splits, licence and `commercial` (True / False / None = unclear). `--release-only` keeps True
  only. Non-commercial or unclear: ag_news, yahoo, sciq, trec, mrpc, cyber (3 research-only
  sources), yelp, sst2, imdb, tweet_eval. **All ids / fields are from memory; M1 must run
  `prepare_mixture.py --check`.**
- Deviations from §5.2:
  - **NQ:** FlashRAG's NQ has questions only (no passages), so we use
    `sentence-transformers/natural-questions` (query, gold passage); negatives are other rows'
    passages (easy negatives; note in the data card). Val = last 2000 rows.
  - **FEVER** needs evidence text: `copenlu/fever_gold_evidence` assumed; confirm.
  - **which_passage** (RAG "which passage answers?") is a global choice over "[1]".."[n]" +
    "none of them". It keeps `answer_seg`, so augmentation re-derives options and label after
    shuffling / dropping segments. Added to Hotpot / 2Wiki / NQ packs when exactly one passage
    is "directly answers"; SQuAD 2.0 builds it over 3–5 contexts of the same article.
  - **Cap counts packs, not groups** (simpler to sample); RAG packs carry several groups.
  - **MNLI** is used as 3-way `nli`; binary `grounded` stays in P1 (phase A) only.
  - Families short of their quota are reported, not topped up from other families.
- Augmentation (`data/tiny_augment.py`): question always drawn from the 8 train templates,
  option wording swapped with p = 0.3, option subsampling above 4 options (gold kept),
  multi-template packs (p = 0.3: 1–2 extra copies of a single-decision pack, each with a
  different template), option order always shuffled. `single_template=True` = T-A4.
- `heldout_view(pack, q, o)` = unseen-template val / test. `build_p2` returns `val_seen`
  (canonical, per-decision T) and `val_unseen` (held-out templates, for `T_custom` and the
  every-1k-steps eval).
- Format family (`data/synthetic.py`): 7 features (URL, email, date, code, phone, money,
  hashtag); 0–2 inserted into a real text, 1–3 questions per pack; labels from regex detectors
  on the final text (so a base text that already has a date is labelled correctly).
- **JevBench-mini**: tooling only (`jevbench/README.md`, `scripts/jevbench.py`
  validate / agreement / freeze, hash check in `load_frozen`). `draft_items.jsonl` = 20
  Claude-written examples of the format, **not** benchmark items. The 300 items are yours to
  write (M2).
- **`.gitignore` bug:** `data*/` (copied from microjev) also ignored `jevcore/data/`, so the
  package wasn't committed. Anchored to the root (`/data*/`). **The microjev repo has the same
  bug: its `jevcore/data/` is not tracked there.** Not fixed in microjev (separate repo,
  worked on in parallel).

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
