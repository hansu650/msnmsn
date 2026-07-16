# MSN 2026 Research Workspace

This repository contains the compact, reproducible research assets for an IEEE MSN 2026 submission under the provisional **Big Data and AI** track.

Current status: the ResearchPilot Phase E PaAno K0 is complete with terminal outcome **STOP_NO_PERFORMANCE_HEADROOM**. The mechanism was established, but all preregistered accuracy-repair branches failed; no method is frozen.

Large datasets, downloaded PDFs, model weights, caches, and Conda environments are intentionally excluded.

## Layout

```text
docs/       research decisions, provenance, experiment reports, manuscript material
code/       route-specific K0 implementation and launch scripts
artifacts/  compact result tables and machine-readable decisions
```

The PaAno paper-reported TSB-AD-U/M values are external references. The project does not reproduce the complete baseline benchmark; it runs only the matched controls needed for the causal K0.

Compact results are in [`docs/experiments/PAANO_K0_RESULTS.md`](docs/experiments/PAANO_K0_RESULTS.md) and `artifacts/paano_k0/`.
