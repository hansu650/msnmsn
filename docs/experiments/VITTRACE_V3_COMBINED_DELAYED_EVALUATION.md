# ViTTrace v3 Combined Delayed-Label Evaluation

## Purpose

ViTTrace v3 produces scores in independent cache-only, encoder-control,
dynamic-line, and spectrogram stages.  These stages must not load labels and
must remain independently resumable.  The combined evaluator supplies the
single label boundary used after every registered stage is complete.

## Frozen contract

- Workload: the exact 492-series manifest and its bound source-data hashes.
- Corrected primary mask: one arm-independent 488-series mask for F1-max,
  AUPRC, and VUS-PR; the four no-positive series are excluded from all three
  detection metrics and retain separate false-positive-burden fields.
- Aggregation: per-series, 11 subgroups, three families, equal-11 primary, and
  file-weighted supplementary views.
- Uncertainty: paired hierarchical bootstrap, subgroup then paired series,
  shared indices for every contrast and metric, seed 2027, 10,000 draws.
- Immutability: no model, renderer, score formula, score file, cache, or frozen
  stage configuration is changed by evaluation.

## Fail-closed sequence

1. Load the combined stage registry and verify its config/manifest hashes.
2. Verify all 492 source-data hashes without importing label modules.
3. For every registered stage, require its exact 492-series status grid,
   retained-failure absence, score and manifest hashes, length/dtype/finite
   invariants, and stage-specific provenance.
4. If any stage is incomplete or invalid, write a durable `BLOCKED` report and
   exit before importing the ground-truth loader.
5. After all stages are `READY`, assert score/data immutability, load labels
   once, freeze and verify the common 488-series mask, and evaluate every arm.
6. Materialize both combined and per-stage metrics plus the stage evaluation
   index used by downstream table/figure packaging.

## Incremental stage registry

Every arm has a globally unique output ID and a stage-local `source_arm`.
Dynamic line variants therefore map their local `REL/IHP/FULL` transactions to
unique output IDs such as `WIN_120_REL` or `BB_B32_FULL`.  Adding a window,
stride, backbone, or representation stage appends a stage and its contrasts;
it does not reinterpret prior transactions.  Until the appended stage reaches
492/492, the combined evaluator returns `BLOCKED` while completed stages remain
individually auditable.

## Output interface

The evaluation directory contains:

- `combined_preflight_status.json`
- `per_series_metrics.csv`
- `valid_series_mask.csv`
- `arm_metadata.csv`
- `stage_evaluation_index.json`
- `stages/<stage_id>/per_series_metrics.csv`
- `stages/<stage_id>/arm_metadata.csv`
- `stages/<stage_id>/_EVALUATION_COMPLETE.json`
- `_COMBINED_EVALUATION_COMPLETE.json`
- `subgroup11_metrics.csv`, `family3_metrics.csv`, `equal11_metrics.csv`,
  `fileweighted_metrics.csv`, and `bootstrap_ci.csv` after aggregation.

`stage_evaluation_index.json` uses null metric/marker paths for blocked stages,
so downstream delivery code cannot silently treat an unfinished stage as
paper evidence.
