# Tiny-Jev — implementation plan

Spec: [`../15_tiny_jev_design.md`](../15_tiny_jev_design.md) (§ numbers below refer to it). It
builds on Micro-Jev's design doc 14 and its code (`code_repo/microjev`, `jevcore`).
Aims: [`00_aims.md`](00_aims.md). Running log: [`notes.md`](notes.md).

**Scope of this round (2026-10-01): implement only.** GPU is busy, so no training, no
evaluation runs, no model or dataset downloads. Everything is tested on CPU with a tiny randomly
initialised Qwen3 config, so the tests need no network.

## Layout (`code_repo/tinyjev/`, own git repo, pushed to github.com/shubham10divakar/tinyjev)

```
tinyjev/
  15_tiny_jev_design.md
  plan/                    00_aims.md  01_implementation_plan.md  notes.md
  jevcore/                 shared code, vendored from microjev @ 7c630a8 (see notes)
    schema.py packing.py collate.py heads.py loss.py calibration.py report.py scoring.py ...
    backbones/modernbert.py  (Micro-Jev, unchanged)
    backbones/qwen3.py       MarkerEmbedding, TinyJev, Session          <- new
    lm_readout.py            letter-readout baselines B0 / B5            <- new
    templates.yaml           Micro-Jev's RAG templates (unchanged)
    tasks.yaml               every Tiny task: 8 train / 2 held-out questions + wordings  <- new
    data/tasks.py            dataset registry: HF id, licence, family, converter  <- new
    data/mixture.py          P2 sampling (caps, family quotas, n^0.5)    <- new
    data/synthetic.py        format-robustness family                    <- new
    data/tiny_augment.py     option subsampling, multi-template packs, unseen-template view  <- new
    data/jevbench.py         JevBench-mini load / validate / freeze / kappa  <- new
  tinyjev/__init__.py      public API: load(), session(), decide(), policy()
  jevbench/                JevBench-mini format, validator, draft items (H6)
  scripts/                 m0_check prepare_mixture train evaluate bench_latency baselines cascade invariance
  tests/                   T-A … T-E on a tiny random Qwen3 + data / API tests
  configs/                 tiny_0p6b.yaml (App. A), tiny_1p7b.yaml (B6)
```

## Steps (commit + push after each)

| # | Step | Design § | Status |
|---|---|---|---|
| 0 | git init, exclude from outer repo, aims + plan + notes | §1, §10 | done |
| 1 | Scaffold: pyproject (pin transformers 5.17.0, peft), .gitignore, LICENSE, README stub, configs | §2, App. A | done |
| 2 | Vendor `jevcore` from microjev unchanged (own commit, so later diffs are visible) | — | done |
| 3 | `packing.py` for causal: sink token, split render into state / decision parts, `allowed()` for a row slice (cache masks) | §3, §4.4 | done |
| 4 | `backbones/qwen3.py`: `MarkerEmbedding`, `TinyJev` (LoRA via peft, fp32 head, marker init, save / load / merge) | §4.2, §4.3 | done |
| 5 | `Session`: prefill, `decide` (+ crop), `extend` | §4.4 | done |
| 6 | Tests T-A … T-E on a tiny random Qwen3 (fp32, CPU; sdpa + eager; with and without LoRA) | §7.4, M0 | done |
| 7 | Task registry + templates (10 per task, 8 train / 2 held out; options canonical + 2 alternatives) | §5.2, §5.3 | done |
| 8 | Augmentation (option subsampling, multi-template packs), mixture sampler, synthetic format family, held-out builders H1–H5 | §5.2–5.4 | done |
| 9 | JevBench-mini: format, validator, freeze hash, draft items for you to rewrite | §5.4 H6 | done |
| 10 | `tinyjev` API: `load`, `session`, `decide`, `policy(tau)` → confidence / escalate | §9 | todo |
| 11 | LM-readout baselines B0 / B5, unpacked B4 | §7.3 | todo |
| 12 | Scripts: m0_check, prepare_mixture, train (LoRA, lr groups, token accumulation, `--overfit`, `--dry-run`), evaluate (+ `T_custom`), bench_latency, cascade, invariance | §6, §7 | todo |
| 13 | README how-to-run for M0–M5; update notes | — | todo |

## Later (needs GPU / network; not this round)

- **M0 proper:** `scripts/m0_check.py` on real `Qwen/Qwen3-0.6B` — config facts (§2), bool-mask
  path, T-A…T-E on real weights, tokens/s with LoRA + checkpointing at 2048.
- **M0 reading:** `jaredpalmer/kev` and `TianyuCodings/NanoJev` head / mask design (needs network).
- **M1:** confirm every HF id / field in `data/tasks.py` (`scripts/prepare_mixture.py --check`),
  build P1 = Micro-Jev phase A, train seeds 0–2.
- **M2:** write and freeze JevBench-mini (300 items, by you), data card with licences.
- M3–M6 as in §10.
