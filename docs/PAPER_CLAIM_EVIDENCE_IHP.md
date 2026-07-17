# IHP Claim--Evidence Matrix

| Claim | Evidence | Allowed wording |
|---|---|---|
| Released masks are zero based while the released projector queries shifted membership. | Immutable vendor audit plus deterministic topology tests. | State the executable mismatch exactly; avoid intent claims. |
| Every valid pooled-scale query is shifted; 13 wrap across rows and the terminal query is unsupported. | Exhaustive 14x14 incidence enumeration over identical masks in all 492 caches; corrected branch covers 196/196. | Structural fact; do not imply that only 13 queries are affected. |
| IHP changes projection only. | Shared score manifests, cache hashes, CLIP hash, matcher, memory and evaluator. | Frozen, training-free, parameter-free projection correction. |
| IHP improves full-benchmark mean F1. | 0.662142 versus same-cache 0.635409. | Mean improvement; not statistically significant under the registered hierarchical CI. |
| IHP has higher observed AUPRC and VUS-PR. | Deltas +0.021811/+0.010200; unadjusted interval lower bounds +0.006207/+0.002064. | Observed paired improvement; intervals exclude zero but are not adjusted for arm selection. |
| IHP is competitive with paper-reported ViT4TS. | 0.662142 versus 0.612; higher on 9/11 subdatasets. | Descriptive external comparison only. |
| IHP adds no frozen-model inference. | Same released token cache and CLIP forward count for REL_U and IDX_U. | No extra backbone pass; do not infer arm-isolated latency or memory. |

Selection boundary: IHP was a prespecified component arm promoted after the
registered composite candidate failed on the same 492-series evaluation.
Bootstrap intervals quantify sampling variability but are not
selection-adjusted confirmatory inference. The released full-series
preprocessing and all-window memory also make the screen offline and
transductive.

Forbidden claims: SOTA, paired superiority over paper numbers, confirmatory
statistical significance after arm selection, effective Coordinate Envelope,
VLM reasoning, streaming readiness, or universality beyond the audited frozen
visual projector.
