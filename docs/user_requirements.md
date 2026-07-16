# User Requirements - MSN 2026 PaAno Route
> Updated: 2026-07-16 | Workflow: ResearchPilot | Status: PHASE_E_CODING - GO_K0_MECHANISM_ONLY

## Research Direction Constraints

- Target an IEEE MSN 2026 Regular Paper, provisionally under **Big Data and AI**, using CCF-A-level standards for novelty, evidence, fairness, and reproducibility.
- Frame the work around network, server, IoT, or industrial telemetry anomaly detection.
- Use a **2026 accepted paper** as the primary baseline. Prefer official, mature, well-starred code that runs on one RTX 4090; stars are an adoption signal rather than scientific evidence.
- Current primary baseline: **PaAno (ICLR 2026)**, official repository `jinnnju/PaAno`, frozen locally at commit `d4c67116190efa4592dc6a8a157ced0def68b6af`.
- Earlier CCF-A papers may supply mechanisms, technical inspiration, controls, secondary baselines, and reusable improvement principles.
- Prefer a falsifiable root-cause chain followed by the smallest matched intervention. Do not freeze a method before real-data K0 establishes both mechanism and performance headroom.
- Do not revive stopped PrefixCal/DualCal, TSPulse calibration, CANDI, GuardACAS, or TimeRCD routes.
- Avoid threshold refresh, post-hoc calibration, pooling replacement, family-specific selection, alpha blending, new large backbones, or multi-GPU pretraining as the main novelty.

## Confirmed Research Direction and Questions

**Direction:** diagnose whether PaAno's paper-code negative-selection mismatch, non-comparable checkpoint rule, and near-duplicate temporal positives produce a weakly active or mis-executed training objective, and retain method design only if a nontrivial mechanism causes measurable real-data anomaly-detection headroom.

- **RQ1 (core):** How much does PaAno's learned objective contribute to anomaly-discriminative normal representations beyond its patch architecture and memory scorer across heterogeneous telemetry families?
- **RQ2 (mechanism):** Do the negative-selection execution mismatch, scheduled-loss checkpoint rule, and extreme raw overlap suppress useful triplet activity or bias the learned representation?
- **RQ3 (boundary):** Under which anomaly families and temporal regimes does restoring meaningful objective activity improve detection without losing useful local-shift invariance or PaAno's efficiency?

No method name or intervention is frozen. A failed K0 closes the route rather than triggering an unregistered rescue module.

## Evidence and Execution Rules

- ResearchPilot skills lead every stage. Preserve paper-code cross-validation, independent literature support, cross-family real-data evidence, and same-code causal controls.
- The user grants standing authorization to advance through normal ResearchPilot stages without routine confirmation. Pause only after a scientific experiment fails, a new hypothesis is required, scope/compute must materially expand, or an external submission/publishing action is imminent.
- Move quickly and minimize nonessential audits. Routine download, environment, implementation, and technically equivalent retry decisions are autonomous.
- First K0 uses six deterministic real telemetry files spanning both tracks: NAB-U, IOPS-U, Exathlon-M, SMD-M, SMAP-M, and SWaT-M.
- No online algorithm/scorer may consume test labels. Labels are evaluator-only after scores are written.
- Do not reproduce the baseline's full benchmark. Use PaAno paper-reported headline values as external comparison; run official PaAno only on the small identical inputs needed for same-code causal controls.
- Primary threshold-free metrics: VUS-PR and AUPRC. Preserve all negative outcomes in route decisions.
- When a long GPU job starts, create a 15-minute monitoring automation and use runtime for independent literature, analysis, documentation, and packaging.

## Compute and Repository

- Hardware: one NVIDIA RTX 4090, 24 GB VRAM.
- Create a fresh route-specific Conda environment after implementation is frozen.
- Reuse verified local TSB-AD data when compatible; otherwise use official sources with manifests/checksums.
- Publish important code, frozen configs, compact result tables, and manuscript-facing documents to `https://github.com/hansu650/msnmsn`.
- Never commit raw datasets, large caches, full model weights, Conda environments, downloaded PDFs, or temporary renders.

## Phase E Execution Confirmation

- Runtime environment: `D:/Anaconda/envs/paano_msn` with Python 3.11 and separately installed CUDA PyTorch 2.7.1 (`cu128`).
- Device: one NVIDIA RTX 4090; use CUDA for smoke and all main trajectories.
- Dataset handling: reuse the six hash-verified files under `D:/qintian_datasets/TSB-AD/paano_k0`; no additional baseline dataset download is required for K0.
- Auto-run strategy: Codex runs fast tests, smoke, long primary K0, evaluation, and aggregation automatically. The user does not need to confirm routine transitions; pause only under the standing scientific-failure/material-scope rules above.
- Git: use the whole compact project at `https://github.com/hansu650/msnmsn`; include important code, frozen configuration, compact result tables, and research-facing documents only.
- README: keep the repository-level `README.md` as the primary entry point and a code-level run guide under `code/README.md` when useful.

## Document Preferences

- Research artifacts and manuscript-facing text: English.
- User-facing progress and decisions: concise Chinese.
- No detailed tutorial for every paper; use evidence matrices and deep summaries only for key works.
