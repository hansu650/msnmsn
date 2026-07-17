# IHP Full Evidence Freeze

## Status

`PAPER_EVIDENCE_READY_WITH_POST_SELECTION_DISCLOSURE`

The Coordinate-Envelope headline is closed. The final paper-facing method is
the already frozen `IDX_U` arm, named **Index-Consistent Harmonic Projection
(IHP)**. IHP was a prespecified component arm, but it was promoted only after
the registered composite candidate failed on the same 492-series evaluation.
No IHP score, formula, parameter, data row, or checkpoint was changed after
that result; nevertheless, uncertainty intervals are not selection-adjusted.

## Fixed scope

- Baseline system: frozen ViT4TS screening stage from VLM4TS (AAAI 2026).
- Data: 492 official series, 11 subdatasets, NAB/NASA/Yahoo.
- Candidate: `IDX_U` / IHP.
- Same-cache control: `REL_U`.
- External reference: paper-reported ViT4TS Table 1 only; no paired claim.
- Weights, matcher, memory, preprocessing, and evaluator: fixed. F1-max uses
  the same evaluator but selects its maximizing operating point per arm.
- Protocol: label-free but offline/transductive because preprocessing and
  median memory use the complete evaluated series.

## Mechanism and two equations

For scale `s`, pooled token `k`, and the vendor-provided zero-based membership
set `M_{s,k}`, IHP constructs

`A_s(i,k) = 1[i in M_{s,k}], i = 0,...,195`.

For finite token cost `d_{s,k}`, the scale projection is the
validity-normalized harmonic mean

`p_s(i) = n_s(i) / sum_{k:A_s(i,k)=1} 1/d_{s,k}`,

with the same zero-safe convention as the frozen implementation: if any
incident cost is exactly zero, the harmonic projection is zero.

where `n_s(i)=sum_k A_s(i,k)`. Scale projections are fused using the same
released equal-scale average. Literal incidence yields `196/196` coverage;
all 195 valid released `i+1` queries are displaced by one flattened cell, 13
wrap across row boundaries, and the terminal query is unsupported.

## Full results

| Arm | Equal-11 F1-max | AUPRC | VUS-PR |
|---|---:|---:|---:|
| REL_U | 0.635409 | 0.297401 | 0.687294 |
| IHP (`IDX_U`) | 0.662142 | 0.319212 | 0.697495 |
| Delta | +0.026733 | +0.021811 | +0.010200 |

Unadjusted hierarchical paired bootstrap, 10,000 replicates, resampling 11
subdatasets then paired files:

| Metric | 95% CI |
|---|---:|
| F1-max | [-0.002289, 0.068128] |
| AUPRC | [0.006207, 0.039806] |
| VUS-PR | [0.002064, 0.019693] |

The F1 mean improves but its interval crosses zero. The AUPRC and VUS-PR
intervals exclude zero, but none is interpreted as confirmatory post-selection
inference.

## External paper comparison

IHP equal-11 F1-max is `0.662142`, versus paper-reported ViT4TS `0.612`, and
is higher in 9/11 Table-1 subdatasets. This is descriptive because it is not a
same-execution paired comparison. VLM4TS `0.659` is not the primary comparator:
it adds language-model verification absent from the frozen ViT4TS/IHP path.

## Claim limits

- Do not claim the max envelope works.
- Do not claim confirmatory statistical significance after arm selection.
- Do not call external paper numbers a reproduction.
- Do not claim a new backbone, training method, or language-model system.
- Do claim exact index consistency, complete grid coverage, zero additional
  model inference, full fixed-manifest coverage, and the observed paired
  secondary-metric gains with their unadjusted intervals.
