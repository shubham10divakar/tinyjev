# 15 — Tiny-Jev: general packed decision model on Qwen3-0.6B + LoRA (design)

_Status: design, 2026-10-01. Depends on [14_micro_jev_design.md](14_micro_jev_design.md):
same packed format (§3), pair head (§4.4), loss (§5.3) and evaluation protocol (§6).
Build it after Micro-Jev M0–M2, so the shared code is already tested._

---

## 0. Summary

Micro-Jev is a fast specialist. Tiny-Jev is the step towards the real goal: **an open, offline
System-One model that answers decision questions it was never trained on**, with calibrated
probabilities, and that reuses one encoding of the state across a whole RAG loop.

What changes from Micro-Jev:

| | Micro-Jev | Tiny-Jev |
|---|---|---|
| Backbone | ModernBERT-base, bidirectional, 149M, full fine-tune | **Qwen3-0.6B, causal, LoRA** |
| Readout | `<q>` / `<opt>` marker (sees the whole span) | **`<qe>` / `<oe>` end markers** (causal: the last token has seen the span) |
| Mask | Block isolation | Block isolation **AND causal** |
| State reuse | Recomputed every call | **KV-cache: prefill the state once, then ask any number of decision batches**; append new chunks incrementally |
| Training data | RAG decisions (+ cyber) | **Broad decision-instruction mixture** (~25 datasets, many templates); whole task clusters held out to test zero-shot |
| Target hardware | CPU or GPU | GPU first (3060); CPU later |

The scientific question: **does a 0.6B decoder, trained with many question templates and option
wordings in the packed format, generalize to unseen decision tasks well enough to replace a
prompted 4B LLM at a fraction of the latency, with honest confidence for escalation?**

## 1. Goals, non-goals, hypotheses

### Goals

| # | Goal | Measured by |
|---|---|---|
| T1 | Zero-shot decisions on held-out task clusters | Accuracy / NLL on H1–H3 and JevBench-mini (§5.4) |
| T2 | RAG decisions ≥ Micro-Jev; held-out better | MuSiQue / VitaminC |
| T3 | Exact invariance (packing, order) and cache equivalence | §7.4 tests |
| T4 | Calibrated confidence usable for a cascade | ECE with global T on unseen tasks; AURC; cascade curve |
| T5 | Up to 255 options per decision | CLINC150 (150 options), Banking77 (77) |
| T6 | Incremental RAG loop: decide → add chunks → decide again without re-encoding | Latency of the second call vs the first |

### Non-goals

- Free-text generation and chain-of-thought.
- CPU-first deployment (reconsider after M5).
- RLCD-style RL for calibration. It's listed as future work (§12).

### Hypotheses

- **H5. Template diversity drives zero-shot.** Training with 8 templates per task plus option
  rewording beats 1 template by ≥ 5 pts on held-out clusters. _Test:_ ablation T-A4.
- **H6. The decoder prior helps OOD.** At matched RAG data (phase P1), Tiny-Jev beats Micro-Jev on
  held-out MuSiQue / VitaminC by ≥ 3 pts. _Test:_ P1 comparison.
- **H7. Packing is free here too.** Tiny packed ≡ Tiny unpacked in accuracy, and much faster with
  the cache. _Test:_ §7.4, baseline B4.
- **H8. Head readout beats LM-token readout** on calibration and option-count scaling (77–150
  options). _Test:_ B5.

## 2. Backbone

Check these against the downloaded `config.json` in M0.

| Property | Expected value |
|---|---|
| Checkpoint | `Qwen/Qwen3-0.6B` (post-trained) — default. `Qwen/Qwen3-0.6B-Base` — ablation T-A1 |
| Parameters | ~0.6B total (~0.44B without embeddings) |
| Layers / hidden / heads | 28 / 1024 / 16 query heads, 8 KV heads (GQA), head_dim 128 |
| MLP | intermediate 3072, SwiGLU |
| Context | 32,768 |
| Vocab / embeddings | ~151.9k rows, tied input/output embeddings |
| Licence | Apache-2.0 |
| HF API (transformers 5.17.0, checked in your `.venv`) | `Qwen3Model.forward(input_ids / inputs_embeds, attention_mask, position_ids, past_key_values, use_cache)`. `attention_mask` may be a **dict keyed by layer type** (`"full_attention"`, plus `"sliding_attention"` only if the config has such layers). A **4D mask is used as-is**. `DynamicCache.crop(-n)` removes the last n tokens |

Implementation notes:

- Load with `AutoModel` (gives `Qwen3Model`, **no LM head**, saves ~155M output-projection compute),
  in `bf16`, with `attn_implementation="sdpa"`.
- Load the causal-LM class only for baselines B0 and B5.
- Don't use the chat template. Decisions are not chat turns, and the markers do the work.

## 3. Packed format: what differs from doc 14 §3

Same JSON schema (14 §3.2), same bookkeeping arrays (14 §3.5), same `allowed()` called with
`causal=True` (14 §3.8). The rendered row:

```
<state> {header} <seg> [1] {title}: {text} <seg> [2] … 
<dec> <q> {question} <qe> <o> {option 1} <oe> <o> {option 2} <oe> … [<ref> × |targets|]
<dec> …
```

Readouts:

| Readout | Token | What it has seen (mask + causality) |
|---|---|---|
| Global anchor | `<qe>` | State + question |
| Option j | `<oe>` of option j | State + question + option j text (isolated mode) |
| Segment anchor | `<ref>` for segment i | Header + segment i + question |

Other differences:

- **No `[CLS]` / `[SEP]`.** Optionally start with `<|endoftext|>` as a BOS-like sink token in the
  state header (attention-sink stability). Ablation: with vs without.
- **Position restart** is identical to Micro:
  - decision block at `S`;
  - each option at `S + Lq`;
  - `<ref>` at `S + Lq`.

  With the causal mask, option j attends to tokens at positions < its own, all of which belong to
  the state or its question. So the computation is **exactly** that of "state + question + option j"
  run alone.
- **Causal asymmetry of the state:** segment i only sees segments < i. Mitigations:
  - shuffle segment order in training;
  - the invariance tests do *not* claim segment-order invariance (report it as a metric);
  - ablation T-A7 tries "state twice" (an echo of the segments).
- **New tokens:** `<state> <seg> <dec> <q> <qe> <o> <oe> <ref>`.
  - Add them to the tokenizer.
  - **Don't train the 151.9k-row embedding matrix.** Use a small separate `MarkerEmbedding`
    (8 × 1024) and substitute its vectors at marker positions (§4.2). The Adam state for the full
    embedding would cost ~1.2 GB for 8 rows.

## 4. Model

### 4.1 Overview

```
            ┌──────────── prefill (once per state) ────────────┐
state ids ─►│ Qwen3Model + LoRA, causal mask, pos 0..S−1         │──► KV cache (S tokens)
            └───────────────────────────────────────────────────┘
decision ids ─► Qwen3Model + LoRA, past_key_values = cache, 4D mask [B,1,Td,S+Td],
                position_ids restarted at S ──► H_dec [B, Td, 1024]
                                      │
          a = H[<qe>] or H[<ref_i>],   o_j = H[<oe>_j]
                                      │
                   PairScorer(1024) ─► z ─► softmax(z / T) ─► {option: p}
            (then cache.crop(−Td) so the cache holds only the state again)
```

**Training** doesn't use the cache. It runs the whole row in one pass with the full causal block
mask from `allowed(..., causal=True)`. **Inference** splits it into prefill + decision pass. Both
give the same numbers; test T-C in §7.4 checks this.

### 4.2 Wrapper

```python
class MarkerEmbedding(nn.Module):
    def __init__(self, marker_ids: list[int], d: int, init: torch.Tensor):
        super().__init__()
        self.register_buffer("ids", torch.tensor(marker_ids))
        self.table = nn.Parameter(init.clone())                    # [n_markers, d]

    def forward(self, input_ids, base_embeds):                     # base_embeds [B, T, d]
        hit = input_ids[..., None] == self.ids                     # [B, T, n]
        sub = (hit.to(self.table.dtype) @ self.table)              # [B, T, d]
        return torch.where(hit.any(-1, keepdim=True), sub, base_embeds)


class TinyJev(nn.Module):
    def __init__(self, base="Qwen/Qwen3-0.6B", marker_ids=None, marker_init=None, lora=None):
        super().__init__()
        bb = AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.bb = get_peft_model(bb, LoraConfig(**lora)) if lora else bb
        d = bb.config.hidden_size
        self.markers = MarkerEmbedding(marker_ids, d, marker_init)
        self.head = PairScorer(d)                                   # doc 14 §4.4
        self.layer_types = set(bb.config.layer_types)

    def embed(self, ids):
        safe = ids.clamp_max(self.bb.get_input_embeddings().num_embeddings - 1)
        return self.markers(ids, self.bb.get_input_embeddings()(safe))

    def encode(self, ids, pos, mask4d, cache=None):
        m = {t: mask4d for t in self.layer_types}                   # all "full_attention" expected
        out = self.bb(inputs_embeds=self.embed(ids), position_ids=pos, attention_mask=m,
                      past_key_values=cache, use_cache=cache is not None)
        return out.last_hidden_state, out.past_key_values

    def forward(self, ids, pos, mask4d, g_batch, g_anchor, g_opts, g_opt_valid):
        h, _ = self.encode(ids, pos, mask4d)
        a, o = h[g_batch, g_anchor].float(), h[g_batch[:, None], g_opts].float()
        return self.head(a, o).masked_fill(~g_opt_valid, float("-inf"))
```

- Run the head in fp32: `.float()` before the head, and keep the head's weights fp32.
- Mask dtype: bool, True = attend. **Check in M0** that SDPA in 5.17 takes bool 4D masks for Qwen3.
  If not, convert with `torch.where(m, 0, finfo.min)` to the activation dtype.

### 4.3 LoRA and trainable parameters

| Part | Setting | Trainable |
|---|---|---|
| LoRA | r = 16, α = 32, dropout 0.05, targets `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | ≈ 10M |
| Head | PairScorer(1024), fp32 | ≈ 3.1M |
| Markers | 8 × 1024 | 8k |
| Backbone | Frozen (bf16) | — |

Full fine-tuning is ablation T-A3. It won't fit a 12 GB card with Adam, so run it on a rented GPU
only if LoRA plateaus.

### 4.4 Cached inference (`jevcore/backbones/qwen3.py: Session`)

```python
class Session:
    """One state, many decision calls; chunks can be appended (RAG loop)."""
    def __init__(self, tj: TinyJev, tok, header: str, segments: list[dict]):
        self.tj, self.tok = tj, tok
        self.row = render_state(header, segments, tok, causal=True)    # ids, pos, prt (blk = 0)
        ids, pos = self.row.tensors()
        mask = causal_state_mask(len(ids))                             # [1,1,S,S]
        _, self.cache = tj.encode(ids, pos, mask, cache=DynamicCache(config=tj.bb.config))
        self.S = len(ids)

    @torch.no_grad()
    def decide(self, decisions: list[dict]) -> dict:
        dec = render_decisions(decisions, self.tok, S=self.S, causal=True)  # blk ≥ 1, pos from S
        full = allowed_with_state(self.row, dec, causal=True)              # [Td, S+Td] rows of allowed()
        h, _ = self.tj.encode(dec.ids, dec.pos, full[None, None], cache=self.cache)
        self.cache.crop(-len(dec.ids))                                     # drop decision tokens
        z = self.tj.head(*dec.gather(h))                                   # anchors / options
        return dec.to_probs(z, temperatures)

    def extend(self, segments: list[dict]):
        """Append chunks: new tokens attend to the whole previous state (causal), positions continue."""
        add = render_segments(segments, self.tok, start_index=self.row.n_segments, pos_start=self.S)
        mask = causal_append_mask(self.S, len(add.ids))                    # [1,1,Ta,S+Ta]
        _, self.cache = self.tj.encode(add.ids, add.pos, mask, cache=self.cache)
        self.row.extend(add); self.S += len(add.ids)
```

`allowed_with_state` is `allowed()` evaluated on the concatenated (state ⊕ decision) arrays,
keeping only the decision query rows: shape `[Td, S + Td]`.

- **Batching** several states: left-align each state, pad decision blocks, and build the masks per
  row.
- **First version:** batch only decisions within one state. That's the RAG-loop case.

## 5. Training data: the decision-instruction mixture

### 5.1 Phases

| Phase | Data | Why |
|---|---|---|
| **P1 parity** | Exactly Micro-Jev phase A (doc 14 §5.1) | Architecture comparison at matched data (Nano vs Micro vs Tiny) → H6 |
| **P2 general** | §5.2 mixture | Zero-shot ability → T1, H5 |
| **P3 distill + loop** (optional) | Teacher-LLM soft labels on your own RAG traces; `next_action` from rollouts (plan 03) | RAG-loop control in your actual pipeline |

### 5.2 P2 mixture

Rules:

- Cap of **20k groups per dataset**.
- Within a family, sample datasets with probability ∝ n^0.5.
- Family quotas as listed.
- **Check every licence before a weights release; drop non-commercial sources** from the release
  mixture. A research-only run may keep them, but document it.

| Family (quota) | Datasets (HF ids to confirm in M1) | Decision forms |
|---|---|---|
| RAG control (35%) | HotpotQA distractor, 2WikiMultihopQA, SQuAD 2.0, Natural Questions (short-answer, via FlashRAG preprocessing) | relevance (segment), sufficient, "which passage answers?" (choice over passages) |
| Verification / NLI (12%) | MultiNLI, SNLI, FEVER | grounded / entailed / contradicts (3-way choice) |
| Yes/no QA (5%) | BoolQ | noul with the passage as state |
| Topic / intent (15%) | AG News, DBpedia-14, Yahoo Answers Topics, TREC, Banking77 (77 options), CLINC150 (150 options) | choice; option subsampling (§5.3) |
| Multiple-choice reasoning (15%) | ARC-Easy/Challenge, OpenBookQA, CommonsenseQA, SciQ, HellaSwag | choice; options = answer texts |
| Paraphrase (5%) | PAWS, MRPC | noul "same meaning?" with both texts as two segments |
| Security (10%) | Cyber-Jev train sets (http_attack, prompt_injection, phishing_url) | noul |
| Format robustness (3%) | Synthetic: decisions about the state's own form ("Does the text contain a URL / a date / code?") generated by rules | noul; teaches that the question, not the data source, decides the answer |

### 5.3 Templates and augmentation (applies to every family)

- **Question templates:**
  - 10 per task: 8 for training, 2 held out for the unseen-template test.
  - Write them once (by hand, or LLM-drafted then edited) and freeze them in `templates.yaml`.
  - About a third are phrased with the label semantics in the question ("Is this review positive?"),
    and the rest neutrally ("What is the sentiment?").
- **Option wordings:** canonical + 2 alternatives per label (positive / favourable / good …), one
  held out.
- **Option subsampling** (n > 4 classes): train with m ~ U{2..n} options, always including the gold
  one. Evaluate with all options.
- **Option-order shuffle:** always. **Decision subset:** p = 0.8. **Segment shuffle:** always.
- **Multi-decision packs** where natural: RAG packs; for classification, ask 2–3 different
  templates of the same task in one pack (cheap and consistent).
- **Row packing** of short examples (doc 14 §5.5).

### 5.4 Held-out evaluation (never trained on, including their templates)

| Id | Cluster | Datasets | Tests |
|---|---|---|---|
| H1 | Sentiment | SST-2, IMDB, Yelp polarity | Unseen task type with a common label space |
| H2 | Commonsense causal / coreference | COPA, Winogrande | Unseen reasoning format |
| H3 | Toxicity / offensive | tweet_eval (offensive, hate) | Unseen safety-style task (close to cyber, different domain) |
| H4 | RAG OOD | MuSiQue, VitaminC (as in Nano / Micro) | Comparison across the whole line |
| H5 | Security OOD | Cyber-Jev `data_heldout` | Domain transfer |
| H6 | **JevBench-mini** | 300 hand-written realistic app decisions (email urgency, ticket routing, PII present?, refund eligible?, is the code change risky?, …), ~20 decision types, novel option sets, labelled by you; second annotator on 100 for agreement | The real use case |

**Freeze H6 before training starts**, and never tune on it.

## 6. Training recipe

| | P1 | P2 |
|---|---|---|
| Objective | Grouped CE (doc 14 §5.3), no smoothing | same |
| LR | LoRA 2e-4; head 1e-3; markers 1e-3 | same |
| Optimizer | AdamW (0.9, 0.95), weight decay 0 on LoRA and markers, 0.01 on head | same |
| Schedule | 3% warmup, cosine to 10% | same |
| Length | max 2048 (RAG packs), 4096 allowed | max 2048 |
| Batch | ~8k tokens per micro-batch, accumulate to ~64k tokens per step | same |
| Epochs | 3, best by dev NLL | 1 pass over the capped mixture; eval every 1k steps on the unseen-template val |
| Precision / memory | bf16 weights, fp32 head and LoRA, gradient checkpointing, SDPA | same |
| Seeds | 0, 1, 2 | 0 (+ 1 repeat for the headline) |

- **Compute:** measure tokens/s on the 3060 in M0 before fixing the P2 size. If one P2 epoch is
  over ~12 h, lower the per-dataset cap to 10k first. Keep the families.
- **Calibration:**
  1. Per built-in decision: temperature on its calib split.
  2. **Global `T_custom`** for anything else, fitted on the unseen-template val set (not the
     held-out clusters).
  3. Report ECE on H1–H6 with `T_custom` (honest OOD calibration).
  4. Also report **AURC** and the cascade curve.
- **P3 (optional):**
  - Teacher = a larger local model (Qwen3-8B/14B scored via option log-likelihoods) or an API LLM.
  - Loss = CE on gold where labels exist + λ·KL(teacher ‖ student), with λ = 0.5, on unlabeled traces.
  - `next_action` labels come from rollouts (plan 03).

## 7. Evaluation

### 7.1 Sets

Everything in doc 14 §6.1, plus H1–H6 and the unseen-template test.

### 7.2 Metrics

Same as doc 14 §6.2, plus:

- macro-average accuracy and NLL over held-out clusters;
- accuracy vs number of options (T5);
- second-call latency (T6).

### 7.3 Baselines

| Id | System | Isolates |
|---|---|---|
| B0 | Qwen3-0.6B **zero-shot**, LM-token readout (A/B/C letters, softmax over letter logits) | What training adds |
| B1 | Qwen3-4B zero-shot (existing Ollama setup), same prompt | The LLM judge to beat or match |
| B2 | Nano-Jev v1.0 | RAG reference |
| B3 | Micro-Jev (doc 14) | Encoder vs decoder at matched data (P1) |
| B4 | **Tiny-Jev unpacked:** same recipe, one decision per sequence | H7 — packing cost / benefit |
| B5 | **Tiny-Jev LM-readout:** same data, LoRA, trained to output the option letter | H8 — head vs token readout |
| B6 | Qwen3-1.7B Tiny-Jev (same recipe) | Scale, if compute allows |

### 7.4 Correctness tests (CI on a tiny random Qwen3 config + once on trained weights)

| Test | Comparison | Pass threshold (max \|Δp\|) |
|---|---|---|
| T-A | Packed vs alone vs reordered decisions vs permuted options | ≤ 1e-3 fp32, ≤ 1e-2 bf16 |
| T-B | Training-path forward (one pass, full mask) vs single-decision forward | ≤ 1e-3 fp32 |
| T-C | **Cache path:** `Session.decide` vs the training-path forward | ≤ 1e-3 fp32 |
| T-D | `Session.extend` then `decide` vs a fresh `Session` built on the full state | ≤ 1e-3 fp32 |
| T-E | Cache length after `decide` equals S (crop worked) | exact |

### 7.5 Latency (3060, batch 1, median of 200 after warm-up)

- **S10** as in doc 14 §6.5: prefill + decide, and decide-only on a warm cache.
- **Loop scenario:**
  1. decide (relevance × 10 + sufficient);
  2. extend by 5 chunks;
  3. decide (relevance × 5 + sufficient + next_action).

  Compare against B4 and B1.

### 7.6 Cascade

- Reuse `nano_jev/scripts/cascade.py`: accept when max p ≥ τ, otherwise escalate to B1 (or a
  larger LLM).
- Plot accuracy vs % escalated on H4 and H6. Report the τ that keeps ≥ 98% of the teacher's
  accuracy, and how much was escalated.
- **Fit τ on validation data only.** JEV-as-a-Judge found thresholds don't transfer across tasks,
  so report per-cluster τ as well.

## 8. Ablations (priority order)

| Id | Change | Question |
|---|---|---|
| T-A4 | 1 template + canonical options only | H5: what template diversity buys |
| B4 / B5 | (baselines above) | H7, H8 |
| T-A1 | `Qwen3-0.6B-Base` init | Does post-training help decisions? |
| T-A2 | LoRA r ∈ {8, 16, 64} | Capacity |
| T-A5 | Siblings options (causal: later options see earlier) | Accuracy vs order bias |
| T-A6 | No option subsampling | Option-count robustness (T5) |
| T-A7 | State echo (segments twice) | Fix for causal segment asymmetry |
| T-A8 | Drop one family from training (leave-one-family-out, 2–3 runs) | Which families transfer to H6 |
| T-A9 | `linear(o)` head | Pair head value |
| T-A3 | Full fine-tune (rented GPU) | LoRA ceiling |

## 9. Inference API

Same surface as Micro-Jev, plus sessions:

```python
import tinyjev
d = tinyjev.load()                                   # HF "sdmlai/tiny-jev", tag v0.1, GPU if available

s = d.session(query=q, passages=chunks)              # prefill once
r1 = s.decide(["relevance", "sufficient"])
s.extend(more_chunks)                                 # incremental, no re-encode
r2 = s.decide(["relevance", "sufficient", "next_action"])

d.decide("Is this ticket about billing?", ["yes", "no"], ticket_text)   # one-shot
d.decide("Route this ticket", ["billing", "tech", "sales", "other"], ticket_text)
```

Outputs carry `confidence = max p` and an `escalate` flag when a policy τ is configured:

```python
d.policy(tau=0.8)
```

## 10. Milestones and exit criteria

| M | Work | Exit criterion |
|---|---|---|
| M0 (1–2 evenings) | Read `jaredpalmer/kev` and `TianyuCodings/NanoJev` code and note their head / mask design (adapt here if better); load Qwen3-0.6B; check config facts (§2); bool-mask path; `MarkerEmbedding`; tests T-A to T-E on a tiny random config; tokens/s on the 3060 with LoRA + checkpointing at 2048 | All tests green; throughput number recorded; decision on mask dtype |
| M1 | P1 on Micro-Jev's phase-A data; seeds 0–2; compare B2, B3, B4 | H6 / H7 verdicts with CIs |
| M2 | Mixture builders, `templates.yaml`, held-out clusters, **JevBench-mini written and frozen** | Data card: per-dataset counts, licences, token totals |
| M3 | P2 run; evaluate H1–H6; B0, B1, B5 | T1 verdict; zero-shot table |
| M4 | Ablations T-A4, T-A1, T-A5, T-A6 (then the rest) | Ablation table |
| M5 | Calibration (`T_custom`), cascade curves, latency, Session API, packaging | `pip install tiny-jev`; model card with OOD calibration and cascade guidance |
| M6 (optional) | P3 distillation + `next_action` inside your RAG loop | End-to-end RAG accuracy vs LLM-only control at lower cost |

## 11. Risks

| Risk | Mitigation |
|---|---|
| Zero-shot gain is small (the model learns dataset artefacts, not the question) | Template / option variation; format-robustness family; leave-one-family-out; H6 written by hand |
| Causal segment asymmetry hurts per-chunk relevance | Shuffling; T-A7 echo; compare with Micro on k-sweep |
| Bool 4D mask or cache API differs in 5.17 | M0 tests; float-mask fallback; pin version |
| `crop` semantics change (deprecation of positive arguments noted in 5.17) | Always call with a negative count; test T-E |
| 3060 throughput too low for P2 | Lower caps; shorter max length for classification; rent a GPU for one P2 run |
| Licence contamination in the release | Licence column in the data card; release mixture excludes non-commercial sources |
| Calibration does not transfer to new tasks (as JEV-as-a-Judge and Cyber-Jev found) | Global `T_custom` fitted on unseen templates; report per-cluster ECE; ship cascade defaults per cluster, never one τ |
| Benchmark contamination (Qwen3 pretraining saw public test sets) | H6 (private, new) is the headline zero-shot number; public clusters are secondary |

## 12. Paper framing and future work

**Proposed title:** _Tiny-Jev: A 0.6B Open System-One Decision Model with Cached, Order-Invariant
Packed Decisions_.

**Contributions:**

1. The packed format on a causal LM with exact cache equivalence (state encoded once, decisions
   asked incrementally).
2. Decision-instruction tuning with template and option variation; zero-shot on held-out clusters
   and a hand-built application benchmark.
3. Head vs token readout and packed vs unpacked, controlled.
4. Calibration and cascade on unseen tasks.

**Future work:**

- RLCD-style training on a proper scoring rule over the model's own confidence.
- Qwen3-1.7B / 4B scale-up.
- Speculative "System One → System Two" routing inside the RAG loop.

## Appendix A — `configs/tiny_0p6b.yaml`

```yaml
model: {backbone: Qwen/Qwen3-0.6B, attn: sdpa, dtype: bf16, option_mode: isolated, ref_view: own,
        readout: end_marker, head: pair, head_dtype: fp32, sink_token: true,
        lora: {r: 16, alpha: 32, dropout: 0.05,
               targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]}}
data:  {phase: P2, max_len: 2048, cap_per_dataset: 20000, family_sampling_alpha: 0.5, row_packing: true,
        templates: jevcore/templates.yaml, heldout_clusters: [H1, H2, H3, H4, H5, H6],
        aug: {opt_subsample: true, optperm: true, subset_p: 0.8, shuffle_segments: true,
              verb_p: 0.3, qpara: all_train_templates}}
train: {lr_lora: 2.0e-4, lr_head: 1.0e-3, lr_markers: 1.0e-3, betas: [0.9, 0.95],
        warmup: 0.03, schedule: cosine, min_lr_ratio: 0.1, tokens_per_microbatch: 8192,
        tokens_per_step: 65536, grad_ckpt: true, epochs: 1, eval_every: 1000, seed: 0}
calib: {per_decision: true, custom_T_from: unseen_template_val}
```
