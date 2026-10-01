# Tiny-Jev — aims and expectations

_Written 2026-10-01, before any training. Revisit after M1 (P1 parity) and again after M3 (first
zero-shot table)._

## Where it sits

End goal (`code_repo/end goal.txt`): an open, offline "System One" decision model. Unstructured
state in, typed probabilistic decisions out, never free text.

| | Nano-Jev v1.0 | Micro-Jev | **Tiny-Jev** |
|---|---|---|---|
| Backbone | MiniLM-L12, 33M | ModernBERT-base, 149M | **Qwen3-0.6B + LoRA** |
| Job | Calibrated RAG decisions, one pass per pair | Fast exact RAG controller: whole decision set in one pass | **General decisions it was never trained on**, plus the RAG loop with a reusable state cache |
| Weak point it attacks | — | Speed, context length | **Zero-shot generality** and custom option sets (Nano v1.0 is weak at them) |

One-line aim: **Tiny-Jev is the first Jev model meant to answer *new* decision questions.** You
write a question and options it never saw, and get calibrated probabilities in tens of
milliseconds. Inside a RAG loop it encodes the state once, then answers any number of decision
batches and takes new chunks without re-encoding.

Who it's for: the app developer who would otherwise prompt a 4B LLM for "is this ticket about
billing?" / "route this to A/B/C/D" / "is this chunk relevant?", and wants the same answer as
a probability, ~10× cheaper, with an honest confidence to escalate on.

## What we expect (three levels)

Reference points: B0 = Qwen3-0.6B zero-shot with letter readout (what training adds), B1 =
Qwen3-4B zero-shot through Ollama (the LLM judge we want to replace), Micro-Jev (encoder at
matched data), Nano v1.0 (test 0.815 / 0.845 / 0.844, held-out 0.635 / 0.670 / 0.675 on
rel / suff / grd).

### Must (otherwise the design or the code is wrong; stop and fix before going on)

| | Expectation | How we know |
|---|---|---|
| M-1 | **Exactness:** packed = alone = reordered = options-permuted, cache path = training path, `extend` = fresh session. max \|Δp\| ≤ 1e-3 fp32 (random weights, CI), ≤ 1e-2 bf16 (trained) | Tests T-A…T-E (§7.4), before any training |
| M-2 | Trained Tiny-Jev beats B0 by **≥ 10 pts macro accuracy** on the held-out clusters H1–H3 | Otherwise decision tuning adds nothing over the raw LM |
| M-3 | P1 in-domain RAG accuracy ≥ Nano v1.0 on all three decisions | 18× bigger backbone, same data |
| M-4 | Warm-cache `decide` at S10 is **≥ 3× faster** than prefill + decide | The point of the cache (T6) |
| M-5 | No invalid output, ever: every call returns a distribution over exactly the given options | By construction (head readout, no generation) |

### Target (a result worth the paper)

| | Expectation |
|---|---|
| T-1 | **JevBench-mini (H6) macro accuracy within 5 pts of B1 (Qwen3-4B)** at ≥ 10× lower latency on the 3060 |
| T-2 | Held-out clusters H1–H3 macro accuracy ≥ B1 − 5 pts; sentiment (H1) ≥ 0.88 on SST-2 |
| T-3 | H6 (decoder prior helps OOD): held-out MuSiQue / VitaminC ≥ Micro-Jev + 3 pts at matched P1 data |
| T-4 | H5 (template diversity): 8 templates + option rewording beat 1 template by ≥ 5 pts on held-out clusters (T-A4) |
| T-5 | Calibration on unseen tasks with one global `T_custom`: ECE ≤ 0.08 on H1–H6 (Nano held-out was ~0.17 before OOD temperature) |
| T-6 | Cascade on H6: keep ≥ 98% of B1's accuracy while escalating ≤ 40% of decisions |
| T-7 | Many options: CLINC150 (150 options) ≥ 0.85, Banking77 ≥ 0.85, evaluated with *all* options |
| T-8 | H8 (head vs LM-token readout): head ECE lower than B5 on held-out, and the gap grows with option count |

### Stretch

- Match or beat B1 on H6 outright.
- One Tiny-Jev that also matches Cyber-Jev in-domain on all three security decisions (security
  family) and improves on Cyber-Jev's held-out transfer (H5).
- Qwen3-1.7B (B6) adds ≥ 3 pts on H6 at < 2× latency.
- P3 distillation: end-to-end RAG answer accuracy equal to the LLM-only loop at ≤ 30% of its
  LLM calls.

## Honest caveats (to say in the paper, not hide)

- **Public held-out clusters are contaminated.** Qwen3 pre-training very likely saw SST-2, IMDB,
  COPA, Winogrande. Their numbers are secondary; **H6 (private, new) is the headline**.
- **Winogrande / COPA (H2) are hard at 0.6B.** Expect Winogrande near 0.55–0.60 even after
  training. H2 is there to show the limit, not to claim a win.
- **Calibration will not transfer perfectly.** JEV-as-a-Judge and Cyber-Jev both found that
  thresholds don't move across tasks. Ship per-cluster cascade defaults and say so.
- **Causal state asymmetry.** Segment i only sees segments before it, so per-chunk relevance may
  trail Micro-Jev in domain. Report the segment-order sensitivity as a metric, don't claim
  invariance to it.
- **The speed claim is against prompted LLMs and against Tiny-unpacked (B4)**, not against
  Nano-Jev (33M is faster per call). Against Micro-Jev expect Tiny to be slower per pass and
  faster only in the multi-call loop (cache).
- **Licences.** The research mixture keeps some non-commercial sets (Yahoo Answers, SciQ, AG
  News unclear). A released checkpoint must be retrained on the release mixture, and the
  data card has to say which one it is.

## Kill / pivot criteria

- **M-1 fails** after debugging (cache or 4D-mask path doesn't honour our masks): pin a
  transformers version or write a minimal custom attention. No training before M-1 is green.
- **After P1 seed 0:** Tiny below Micro-Jev in domain on 2+ decisions by > 2 pts **and** not
  better held out → the causal design costs more than the prior gives. Try T-A7 (state echo)
  before P2.
- **After P2:** held-out macro ≤ B0 + 3 pts → the model learned dataset artefacts, not the
  question. Stop scaling; look at T-A4, T-A8 and the format-robustness family first.
- **Throughput:** if one P2 epoch on the 3060 is > 24 h even with caps at 10k, rent a GPU for
  the P2 run rather than cutting families.
