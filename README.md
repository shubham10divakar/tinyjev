# Tiny-Jev

> Jev-style decision model. Independent; not affiliated with TypeSafe AI or the NanoJev
> GitHub project.

**Status: implementation in progress, not trained yet.** No weights have been released.

Tiny-Jev is the general member of the Jev line (Nano-Jev → Micro-Jev → Tiny-Jev): a
Qwen3-0.6B + LoRA model that returns **calibrated probabilities over any options you give it**,
including decision questions it was never trained on. In a RAG loop it encodes the state once
(KV cache), then answers any number of decision batches and takes new chunks without
re-encoding.

- Design: [`15_tiny_jev_design.md`](15_tiny_jev_design.md)
- Aims and expectations: [`plan/00_aims.md`](plan/00_aims.md)
- Plan and status: [`plan/01_implementation_plan.md`](plan/01_implementation_plan.md)
- Working notes: [`plan/notes.md`](plan/notes.md)

## Licence

Code: Apache-2.0.
