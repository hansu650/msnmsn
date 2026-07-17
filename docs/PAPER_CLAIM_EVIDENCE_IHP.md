# IHP Claim--Evidence Matrix

| Claim | Evidence | Allowed wording |
|---|---|---|
| Released masks are zero based while the released projector queries shifted membership. | Immutable vendor audit plus deterministic topology tests. | State the executable mismatch exactly; avoid intent claims. |
| Shifted membership creates cross-row aliases and one uncovered terminal cell. | Exhaustive 14x14 incidence enumeration; corrected branch covers 196/196. | Structural fact. |
| IHP changes projection only. | Shared score manifests, cache hashes, CLIP hash, matcher, memory and evaluator. | Frozen, training-free, parameter-free projection correction. |
| IHP improves full-benchmark mean F1. | 0.662142 versus same-cache 0.635409. | Mean improvement; not statistically significant under the registered hierarchical CI. |
| IHP improves AUPRC and VUS-PR. | Deltas +0.021811/+0.010200; CI lower bounds +0.006207/+0.002064. | Positive paired full-benchmark improvement. |
| IHP is competitive with paper-reported ViT4TS. | 0.662142 versus 0.612; higher on 9/11 subdatasets. | Descriptive external comparison only. |
| IHP adds no frozen-model inference. | Same released token cache and CLIP forward count for REL_U and IDX_U. | No extra backbone pass; CPU projection overhead only. |

Forbidden claims: SOTA, paired superiority over paper numbers, significant F1
improvement, effective Coordinate Envelope, VLM reasoning, or universality
beyond the audited frozen visual projector.
