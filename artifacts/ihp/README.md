# IHP compact evidence

This directory contains the final paper-facing evidence for Index-Consistent
Harmonic Projection (IHP). The files are compact derivatives of a completed
492-series experiment over all 11 VLM4TS Table-1 subdatasets.

- `ihp_evidence_summary.json`: claim boundaries and headline results.
- `ihp_11_subdataset_metrics.csv`: same-cache IHP and released-control metrics.
- `ihp_equal11_metrics.csv`: equal-subdataset macro metrics.
- `ihp_hierarchical_bootstrap.csv`: paired hierarchical confidence intervals.
- `ihp_external_vit4ts_comparison.csv`: explicitly marked external,
  paper-reported ViT4TS F1-max comparison.
- `ihp_structure_certificate.json`: label-free mask coverage certificate.

Raw datasets, model weights, token caches, per-series arrays, and failed-route
artifacts are intentionally excluded from Git.
