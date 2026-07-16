# User Requirements - MSN 2026 PaAno Route
> Updated: 2026-07-16 | Workflow: ResearchPilot | Status: PHASE_F_FULL_BENCHMARK_OVERRIDE

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

The original K0 terminal result remains preserved. The user's Phase F override authorizes one full-benchmark evaluation of the already registered `PAPERNEG_NONOVERLAP-LAST` arm plus component-removal ablations; it does not authorize a new rescue module.

## Evidence and Execution Rules

- ResearchPilot skills lead every stage. Preserve paper-code cross-validation, independent literature support, cross-family real-data evidence, and same-code causal controls.
- The user grants standing authorization to advance through normal ResearchPilot stages without routine confirmation. Pause only after a scientific experiment fails, a new hypothesis is required, scope/compute must materially expand, or an external submission/publishing action is imminent.
- Move quickly and minimize nonessential audits. Routine download, environment, implementation, and technically equivalent retry decisions are autonomous.
- First K0 uses six deterministic real telemetry files spanning both tracks: NAB-U, IOPS-U, Exathlon-M, SMD-M, SMAP-M, and SWaT-M.
- No online algorithm/scorer may consume test labels. Labels are evaluator-only after scores are written.
- Do not reproduce the paper's complete baseline suite. Use PaAno paper-reported headline values as the main external comparison. `OFFICIAL-LAST` may run only as a component-removal ablation of the project's full arm, not as a separately tuned headline reproduction.
- Primary threshold-free metrics: VUS-PR and AUPRC. Preserve all negative outcomes in route decisions.
- When a long GPU job starts, create a 15-minute monitoring automation and use runtime for independent literature, analysis, documentation, and packaging.
- Treat `20 GiB` free on the C drive as a hard operational floor during long
  experiments. Below that floor, stop nonessential new writes and move or
  clean only clearly identified inactive temporary caches or hash-verified
  cold archives. Never delete or relocate active PaAno runs, scores,
  checkpoints, manifests, logs, the route environment, TSB-AD data, or any
  unverified file.
- Treat `20 GiB` available physical RAM as a separate operational floor during
  long experiments. Below that floor, stop nonessential parallel work and
  release only unrelated idle processes or disposable caches. Never terminate,
  suspend, or alter the active PaAno runner solely to reclaim RAM; if no safe
  release target is identifiable, notify rather than take a risky action.

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

## Phase F User-Confirmed Full-Benchmark Override

- The user explicitly confirmed that the manuscript-facing comparison should use PaAno's exact paper-reported full-Eval VUS-PR values (`U=0.5296`, `M=0.4263`; rounded headlines `0.53/0.43`) rather than a separately reproduced baseline table.
- Run only the project's registered full arm, `PAPERNEG_NONOVERLAP-LAST`, on the complete official Eval lists: 350 TSB-AD-U series and 180 TSB-AD-M series.
- Run the minimum component ablations needed for a paper table: `PAPERNEG-LAST` (remove non-overlap positives) and `OFFICIAL-LAST` (remove both registered execution changes). No additional baseline models are run.
- The six-file K0 same-code negative contrasts remain disclosed as pilot evidence but no longer block this user-directed full benchmark.
- The primary full benchmark uses seed 2027. Seeds 2028/2029 are conditional on the full method exceeding both paper-reported track values; ablations remain seed 2027 unless manuscript review later demonstrates a specific uncertainty requirement.
- Full-run labels remain evaluator-only after score commits. Eval labels cannot choose arm, checkpoint, family, file, or hyperparameters.
- Main tables may compare directly with paper-reported values only when the complete 350/180 lists, PaAno VUS-PR implementation, and paper-compatible file-weighted aggregation are used. The runtime environment is reported separately and does not substitute for protocol alignment.
- Produce numeric CSV/JSON tables first. Figures are deferred until manuscript writing.

## Document Preferences

- Research artifacts and manuscript-facing text: English.
- User-facing progress and decisions: concise Chinese.
- No detailed tutorial for every paper; use evidence matrices and deep summaries only for key works.
- Citation recency is frozen as follows: beyond indispensable citations for the primary baseline, evaluation protocol, metric, inherited component, or comparator provenance, use publications from **2024--2026**.
- Related Work must be predominantly supported by **2025--2026** publications, with only a very small number of directly necessary 2024 papers.
- Pre-2024 citations may establish historical lineage or indispensable provenance only; they must not be used as evidence for a claim of current novelty.
- Keep manuscript sources in a dedicated directory and Tectonic build outputs
  in the repository-level `.latex-build/` tree. Edit LaTeX/BibTeX and textual
  tables only during drafting. Do not create manuscript figure assets,
  figure-generation scripts, figure environments, or placeholder graphics in
  this phase. The user will supply or finalize figures later.

## Goal-Mode Continuous Execution and Paper Delivery

- Use the installed `LMDHQ-0420/ResearchPilot-Skills` workflow as the primary process authority. Remain in Phase F until the full evidence gate is resolved, then execute G.0 through G.7 in order.
- Keep a 15-minute heartbeat automation active whenever a long GPU experiment is running. Use GPU time for independent documentation, official-requirement verification, evidence organization, and packaging.
- The user authorizes continuous execution through a complete IEEE MSN 2026 English manuscript. Routine stage transitions, manuscript drafting, local compilation, testing, and Git pushes do not require a new confirmation.
- If evidence is insufficient, perform only evidence-driven, bounded Phase F iterations. Each iteration must state its diagnosis, frozen change, expected effect, evaluation boundary, and stop condition before code or experiments; preserve every negative result and never tune against Eval labels.
- Pause only when a proposed change materially changes the task/baseline/data scope, requires substantially more compute or external coordination, or immediately precedes an external submission/publishing action.
- Treat `C:/Users/qintian/Downloads/CODEX_NEW_PAPER_WORKFLOW.md` as an auxiliary writing and compilation checklist. Venue-official IEEE MSN requirements override its generic LNCS examples.
- Use `D:/qintian_tools/tectonic/0.16.9/tectonic.exe` for isolated LaTeX builds. This verified binary lives in a dedicated, non-repository tool directory. Keep build intermediates outside the manuscript source tree and verify logs, page count, references, numerical consistency, and rendered PDF layout.
- Commit important experiment code, complete compact results, manuscript sources, and final checked PDF to `https://github.com/hansu650/msnmsn`; do not commit raw datasets, large checkpoints, caches, downloaded papers, or temporary build trees.
