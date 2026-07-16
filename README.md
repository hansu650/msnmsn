# MSN 2026 Research Workspace

This repository contains the compact, reproducible research assets for an IEEE MSN 2026 submission under the provisional **Big Data and AI** track.

Current status: ResearchPilot Phase E implementation. The 2026 baseline is **PaAno (ICLR 2026)**, and the frozen K0 tests execution fidelity, objective activity, checkpoint semantics, and temporal-overlap shortcuts in patch-based telemetry anomaly detection.

Large datasets, downloaded PDFs, model weights, caches, and Conda environments are intentionally excluded.

## Layout

```text
docs/       research decisions, provenance, experiment reports, manuscript material
code/       route-specific K0 implementation and launch scripts
artifacts/  compact result tables and machine-readable decisions
```

The PaAno paper-reported TSB-AD-U/M values are external references. The project does not reproduce the complete baseline benchmark; it runs only the matched controls needed for the causal K0.
