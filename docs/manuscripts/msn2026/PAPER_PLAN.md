# IHP MSN 2026 Paper Plan

> ResearchPilot G.7 revision: 2026-07-17
> Target: IEEE MSN 2026 Regular Paper, Big Data and AI
> Format: anonymous IEEE conference LaTeX, English, at most eight pages

## Frozen title

**Auditing and Repairing Multiscale Coordinate Projection in Frozen Visual
Time-Series Anomaly Detection**

## One-sentence thesis

The released ViT4TS screen queries already zero-based multiscale memberships
with shifted indices; a minimal literal-incidence repair followed by the
unchanged harmonic projector restores complete support and improves paired
telemetry anomaly ranking without training, labels, or another encoder pass.

## Contribution structure

1. **Mechanism audit.** Formalize the executable mask-to-grid contract and
   certify that all 195 valid pooled-scale queries are displaced by one
   flattened cell, 13 wrap across rows, and the terminal query is unsupported.
2. **Minimal repair.** Implement IHP by interpreting supplied memberships
   literally before applying the inherited harmonic projector. The reducer is
   not claimed as a second innovation.
3. **Frozen-system evidence.** Evaluate 492 NAB/NASA/Yahoo series with a
   same-cache control, label-free certificate, unadjusted paired intervals,
   complete subdataset reporting, and separated paper-reported context.

## Paper architecture

### I. Introduction

- Motivate timestamp-localized telemetry screening.
- Explain why a coordinate-contract failure survives a strong frozen encoder.
- Preview the minimal repair and lead with same-cache evidence.
- State three verifiable contributions with bounded scope.

### II. Related Work

- Recent numerical and pretrained TSAD.
- Visual and multimodal TSAD.
- Training-free localization and visual-grid consistency.
- Telemetry benchmarking and MSN-specific monitoring context.

### III. Index-Consistent Harmonic Projection

- Frozen ViT4TS interface and notation.
- Released shifted support graph.
- Literal incidence repair: $A_s(i,k)=\mathbb{1}[i\in M_{s,k}]$.
- Inherited support-normalized harmonic projection on corrected support.
- Frozen renderer, encoder, all-window memory, matching, fusion, and stitcher.

### IV. Experimental Protocol

- Complete 492-series, eleven-subdataset taxonomy.
- Same-cache causal control before external paper context.
- F1-max, AUPRC, VUS-PR, equal-subdataset aggregation, hierarchical bootstrap.
- Post-selection disclosure and non-confirmatory interval interpretation.
- Vendor/data revisions, windowing, preprocessing, memory, evaluator, runtime
  environment, label isolation, and offline/transductive limitation.

### V. Results and Analysis

- **Table I:** same-cache three-metric projection ablation and intervals.
- **Table II:** paper-reported ViT4TS plus local REL-U/IHP for all 11 groups.
- **Fig. 2:** unconnected categorical markers plus paired metric deltas.
- **Table III:** 195 displaced valid queries, 13 row wraps, one hole, 196/196
  repaired support.
- Same-cache identity and persisted-storage boundary; no arm-isolated latency,
  VRAM, or host-RAM claim.

### VI. Limitations and Conclusion

- Same-cache evidence first; external comparison secondary.
- Scope limited to one frozen screen, grid, pooling layout, and offline protocol.
- Explicit post-selection and transductive deployment limitations.

## Claim boundaries

Allowed:

- exact executable mismatch and complete label-free repair;
- zero new trainable parameters and zero incremental frozen-encoder calls;
- observed paired metric gains and whether unadjusted intervals exclude zero;
- descriptive comparison with explicitly paper-reported ViT4TS values.

Forbidden:

- two independently novel modules, a new harmonic reducer, or a new network;
- SOTA or paired superiority over external paper values;
- confirmatory statistical significance after component-arm selection;
- streaming readiness, downstream VLM reasoning, universal grid applicability,
  or unmeasured arm-specific runtime and memory gains.
