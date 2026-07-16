# MSN 2026 Research Workspace

This repository contains the compact, reproducible research assets for an IEEE MSN 2026 submission under the provisional **Big Data and AI** track.

Current status: the ResearchPilot Phase F full benchmark is running. The original six-file K0 remains preserved with terminal outcome **STOP_NO_PERFORMANCE_HEADROOM**; under the user-confirmed full-benchmark override, the already registered `PAPERNEG_NONOVERLAP-LAST` arm and two component-removal controls are being evaluated on the complete 350-series TSB-AD-U and 180-series TSB-AD-M Eval lists. No new rescue module or family-specific selection is permitted.

Large datasets, downloaded PDFs, model weights, caches, and Conda environments are intentionally excluded.

## Layout

```text
docs/       research decisions, provenance, experiment reports, manuscript material
code/       route-specific K0 implementation and launch scripts
artifacts/  compact result tables and machine-readable decisions
```

The external PaAno references are the exact Table 15 default full-Eval VUS-PR values: U `0.5296` and M `0.4263` (rounded headlines `0.53/0.43`). They are labeled paper-reported rather than local reproductions. The project does not rerun the paper's full baseline suite.

## Full benchmark workflow

```powershell
powershell -ExecutionPolicy Bypass -File .\code\scripts\05_run_full_main.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\06_run_full_ablations.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\08_finalize_full.ps1
```

The score runners are label-free. Finalization first requires all 1,590 registered LAST score artifacts, performs a global hash/provenance preflight, then reads labels in the evaluator, aggregates all tracks and arms, renders the English numeric report, and runs the complete test suite.

Compact results are in [`docs/experiments/PAANO_K0_RESULTS.md`](docs/experiments/PAANO_K0_RESULTS.md) and `artifacts/paano_k0/`.
Full-benchmark compact results will be written to `artifacts/paano_full/` and `docs/experiments/PAANO_FULL_MAIN_RESULTS.md` after complete evaluation.
