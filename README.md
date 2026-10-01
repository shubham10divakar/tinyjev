# Tiny-Jev

> Jev-style decision model. Independent; not affiliated with TypeSafe AI or the NanoJev
> GitHub project.

**Status: implemented and tested on random weights, not trained yet.** No weights released.

Tiny-Jev is the general member of the Jev line (Nano-Jev → Micro-Jev → Tiny-Jev): Qwen3-0.6B +
LoRA, returning **calibrated probabilities over any options you give it**, including decision
questions it was never trained on. In a RAG loop it **encodes the state once** (KV cache), then
answers any number of decision batches and takes new chunks without re-encoding. It never
generates text.

| | |
|---|---|
| Design | [`15_tiny_jev_design.md`](15_tiny_jev_design.md) |
| Aims, expectations, kill criteria | [`plan/00_aims.md`](plan/00_aims.md) |
| Plan and status | [`plan/01_implementation_plan.md`](plan/01_implementation_plan.md) |
| Working notes (decisions, deviations) | [`plan/notes.md`](plan/notes.md) |

## How it works (short)

```
[sink] <state> query: … <seg> [1] … <seg> [2] …          ← prefilled once into the KV cache
<dec> <q> {question} <qe> <opt> {option 1} <oe> <opt> {option 2} <oe> … [<ref> per passage]
```

- Causal **block mask** + **position restart**: each decision (and each option) computes
  exactly what it would compute alone, so packing, decision order and option order don't change
  the answer (tests T-A, T-B).
- Readout at the end markers: `<qe>` (or `<ref>` for a per-passage question) and each `<oe>`
  go through a pair head → softmax over the options.
- `Session` keeps the state's KV cache; `decide` appends the decision tokens, reads them, and
  crops them off again; `extend` appends new chunks (tests T-C, T-D, T-E).
- Only LoRA (r 16), the pair head (fp32) and an 8-row marker table train.

## Use (once weights exist)

```python
import tinyjev

d = tinyjev.load("runs/tiny-p2-s0")            # or tinyjev.load() for released weights
s = d.session(query="When was UCL founded?", passages=chunks)
r1 = s.decide(["relevance", "sufficient"])
s.extend(more_chunks)                          # no re-encoding of what's cached
r2 = s.decide(["relevance", "sufficient",
               tinyjev.Q("What should we do next?", ["answer", "retrieve more", "abstain"])])

d.policy(tau=0.8)
r = d.decide("Route this ticket", ["billing", "tech", "sales", "other"], ticket_text)
r.label, r.confidence, r.escalate              # r is also a plain {option: p} dict
```

## Setup

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt            # transformers pinned to 5.17.0
pytest -q                                     # 159 tests, CPU, no network, ~25 s
```

## Running the milestones (design §10)

Every script has `--dry-run` (tiny random model, offline, CPU, seconds) to check it runs.

| M | Command | Notes |
|---|---|---|
| M0 | `python scripts/m0_check.py` then `--dtype bf16`, then `--throughput --p2-tokens N` | Real Qwen3-0.6B: config facts, T-A…T-E, tokens/s on the 3060 → `results/m0.json` |
| M1 | `python scripts/prepare_mixture.py --check` | Confirms every HF id / field (they're from memory) |
| M1 | `python scripts/prepare_mixture.py --phase P1 --out data/p1` | = Micro-Jev phase A |
| M1 | `python scripts/train.py --data data/p1 --out runs/tiny-p1-s0 --seed 0` (seeds 0–2) | + `--baseline B4` |
| M2 | `python scripts/prepare_mixture.py --phase P2 --total 200000 --tokenizer Qwen/Qwen3-0.6B --out data/p2` | Data card in `data/p2/card.json` |
| M2 | `python scripts/prepare_mixture.py --heldout --out data/heldout` | H1–H5 |
| M2 | write `jevbench/items.jsonl`, then `python scripts/jevbench.py freeze` | H6, **before** P2 training |
| M2 | `python scripts/train.py --data data/p2 --out runs/overfit --overfit 200` | Loss should go to ~0 |
| M3 | `python scripts/train.py --data data/p2 --out runs/tiny-p2-s0` | |
| M3 | `python scripts/evaluate.py --run runs/tiny-p2-s0 --data data/p2 --heldout data/heldout --jevbench jevbench/items.jsonl` | Writes `calibration.json` into the run |
| M3 | `python scripts/evaluate.py --lm Qwen/Qwen3-0.6B --name b0 …` / `--lm Qwen/Qwen3-4B-Instruct-2507 --name b1 …` | B0, B1 |
| M3 | `python scripts/train.py --data data/p2 --out runs/b5-p2 --baseline B5`, then `evaluate.py --b5 runs/b5-p2 …` | B5 |
| M4 | `python scripts/train.py --data data/p2 --out runs/p2-A4 --ablation T-A4` (also T-A1, T-A5, T-A6, T-A7, T-A9, T-A2-r8 / r64, T-A3; `--drop-family X` for T-A8) | |
| M5 | `python scripts/bench_latency.py --run runs/tiny-p2-s0` | S10 + RAG-loop scenario |
| M5 | `python scripts/cascade.py --student results/tiny-p2-s0 --teacher results/b1 --test H4 --test H6` | τ fitted on val only |

## Layout

```
jevcore/              shared with Micro-Jev (vendored from microjev @ 7c630a8, see notes)
  backbones/qwen3.py  TinyJev, MarkerEmbedding, Session          (Tiny)
  packing.py          rendering + masks (causal path, sink, echo)
  tasks.yaml          26 decision tasks: 8 train / 2 held-out questions + option wordings
  data/tasks.py       dataset registry (HF ids, licences, converters)
  data/mixture.py     P2 sampler (caps, family quotas, n^0.5), held-out sets
  data/tiny_augment.py, data/synthetic.py, data/jevbench.py
  tiny_trainer.py     training loop (packed, B4 unpacked, B5 letter readout)
  tiny_eval.py        ragged metrics, per-decision T + T_custom, cluster reports
  lm_readout.py       letter-readout baselines B0 / B1 / B5
tinyjev/              public API
scripts/              m0_check, prepare_mixture, train, evaluate, bench_latency, cascade, jevbench
jevbench/             JevBench-mini format, rules, draft examples
```

## Licence

Code: Apache-2.0. Datasets keep their own licences: `data/tasks.py` lists each one, and
`--release-only` builds a mixture from commercially usable sources only.
