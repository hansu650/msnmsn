# Dev Log - PaAno Execution-Fidelity and Objective-Activity K0
> Created: 2026-07-16 | Last updated: 2026-07-16
> Linked implementation guide: `docs/implementation.md`
> This file is append-only. Progress snapshots and entries are appended; prior entries are never rewritten.

## Project Overview

| Item | Detail |
|---|---|
| Research direction | PaAno execution fidelity, objective activity, and temporal-overlap mechanism gate |
| Implementation strategy | SHA-guarded overlay around a frozen strong baseline |
| Framework | Python 3.11, PyTorch 2.7.1+cu128 |
| Git repository | `https://github.com/hansu650/msnmsn` |
| Push scope | Compact whole project; exclude data, weights, caches, PDFs, and environments |

## Project Architecture

The runner reads features only, replays matched random plans through registered trajectories, commits hashed score artifacts, and exits. A separate evaluator verifies score commits before reading labels. Aggregation applies only the pre-registered contrasts and decision gates.

## Project Logic

`OFFICIAL`, `PAPERNEG`, and `PAPERNEG_NONOVERLAP` share one encoder initialization, anchor batches, optimizer budget, memory builder, and scorer. `RAND_BN` provides a no-update architecture-memory floor. Trained trajectories expose both released `BEST` and final `LAST` states.

## Implementation Progress - Snapshot 0

| Module | Files | Status | Notes |
|---|---|---|---|
| Initialization | `requirements.txt`, `pyproject.toml`, READMEs | Done | File creation verified; runtime installation still in progress |
| Contracts/config/vendor | `schemas.py`, `config.py`, `vendor.py` | TODO | |
| Data/replay | `feature_data.py`, `label_data.py`, `replay.py` | TODO | |
| Objectives/training | `objectives.py`, `instrumentation.py`, `trainer.py` | TODO | |
| Memory/scoring/artifacts | `memory.py`, `scoring.py`, `artifacts.py` | TODO | |
| Runner/evaluator/aggregate | entry points | TODO | |
| Scripts/tests | PowerShell and pytest | TODO | |

## Dev Log Entries

### 2026-07-16 13:10 - Phase E initialized

- **Completed**: created the package metadata, library-only requirements file, code README, notebook placeholder, and append-only development log; updated repository status and generated-output exclusions.
- **Issues**: the Phase D guide used a stale environment name and described pinned requirements, conflicting with the installed environment and Phase E rules.
- **Solution**: corrected `docs/implementation.md` first, standardized on `D:/Anaconda/envs/paano_msn`, and kept exact package versions in the environment manifest instead of `requirements.txt`.
- **Run-manual impact**: added environment installation commands below.

### 2026-07-16 13:14 - Transactional artifact module drafted

- **Completed**: added `code/src/paano_k0/artifacts.py` with atomic JSON/NumPy/checkpoint writes, score SHA256 verification, score-manifest serialization, and success-marker-last semantics.
- **Issues**: the module depends on the concurrently implemented `ScoreManifest.from_dict`, so it cannot yet be import-tested.
- **Solution**: recorded the dependency explicitly and left this file WIP until the contract module lands and focused tests pass.
- **Run-manual impact**: none; the public runner command is not yet implemented.

## Implementation Progress - Snapshot 1

| Module | Files | Status | Notes |
|---|---|---|---|
| Initialization | metadata and READMEs | Done | Static files verified |
| Artifacts | `artifacts.py` | WIP | Awaiting schema contract and tests |
| Other implementation modules | per Snapshot 0 | WIP | Parallel implementation in progress |

### 2026-07-16 13:17 - Score-manifest pre-commit contract aligned

- **Completed**: aligned `artifacts.py` with the frozen `ScoreManifest` validator by accepting an all-zero SHA256 sentinel only before the score file is atomically written and hashed.
- **Issues**: `ScoreManifest` correctly rejects an empty hash, while the final `.npy` file hash is unavailable until serialization.
- **Solution**: allow one explicit 64-zero pre-commit sentinel and replace it with the verified durable file hash before exposing `_SUCCESS`; any other conflicting hash remains an error.
- **Run-manual impact**: none.

### 2026-07-16 13:19 - Environment and artifact transaction verified

- **Completed**: verified Python 3.11.15, PyTorch 2.7.1+cu128, CUDA 12.8, RTX 4090 tensor execution, every scientific dependency, and `pip check`; ran `tests/test_artifact_contract.py` with 3/3 passing.
- **Issues**: none; only benign warnings that environment scripts are not globally on PATH.
- **Solution**: all scripts use the absolute environment interpreter, so no PATH mutation is needed. `artifacts.py` is now run-verified and marked Done.
- **Run-manual impact**: environment commands remain correct; added the focused artifact-test command below.

## Implementation Progress - Snapshot 2

| Module | Files | Status | Notes |
|---|---|---|---|
| Environment | `paano_msn` | Done | CUDA tensor test and `pip check` passed |
| Artifacts | `artifacts.py`, `test_artifact_contract.py` | Done | 3 focused tests passed |
| Other implementation modules | per prior snapshots | WIP | Parallel implementation continues |

### 2026-07-16 13:23 - Aggregate provenance contract corrected before runner code

- **Completed**: added the frozen contrast/gate/decision aggregator and two focused decision tests; corrected the Phase D guide before code so activity logs map `series_id` through the frozen manifest and training summaries carry explicit series/family provenance.
- **Issues**: iteration rows intentionally contain `series_id` but not `family`, and the typed training summary alone does not identify a family. The original aggregation wording therefore lacked a provenance join.
- **Solution**: documented and implemented a manifest-based `series_id -> family` join for activity and an explicit runner provenance wrapper for training summaries. This changes no metric, arm, or decision threshold.
- **Verification**: `tests/test_aggregate_decision.py` passed 2/2.
- **Run-manual impact**: added the focused aggregate-test command below.

## Implementation Progress - Snapshot 3

| Module | Files | Status | Notes |
|---|---|---|---|
| Aggregate decision | `aggregate.py`, `test_aggregate_decision.py` | Done | 2 focused tests passed |
| Runner provenance | implementation contract | WIP | Wrapper to be emitted by runner |
| Other implementation modules | per prior snapshots | WIP | Parallel implementation continues |

### 2026-07-16 13:26 - Evaluator-only metric path drafted

- **Completed**: added `evaluate_scores.py` with success/hash verification before label I/O, PaAno VUS-PR/VUS-ROC/AUPRC/AUROC calls, exact 42-artifact coverage, and provenance-preserving metric JSON.
- **Issues**: focused label-order and vendor metric tests are still landing with the concurrent foundational module.
- **Solution**: leave the evaluator WIP until those tests and the full import suite pass; no runner API imports this module.
- **Run-manual impact**: final evaluator CLI will be added after end-to-end runner paths are verified.

## Implementation Progress - Snapshot 4

| Module | Files | Status | Notes |
|---|---|---|---|
| Evaluator | `evaluate_scores.py` | WIP | Implementation complete; focused tests pending |
| Aggregate/artifacts | prior snapshots | Done | 5 combined tests pass |
| Other implementation modules | per prior snapshots | WIP | Parallel implementation continues |

### 2026-07-16 13:30 - Objective, instrumentation, and trainer focused tests passed

- **Completed**: reviewed `objectives.py`, `instrumentation.py`, and `trainer.py`; verified released and paper-faithful negative semantics, exact pretext schedule, separate encoder-gradient diagnostics, post-update BEST timing, LAST preservation, and RAND_BN parameter invariance. Ran 7 focused tests successfully.
- **Issues**: no scientific or shape mismatch found. Memory/scoring parity tests are still being completed separately.
- **Solution**: mark the objective/instrumentation/trainer group Done and keep memory/scoring WIP until direct vendor parity passes.
- **Run-manual impact**: added the focused training-core test command below.

## Implementation Progress - Snapshot 5

| Module | Files | Status | Notes |
|---|---|---|---|
| Objectives | `objectives.py`, `test_objectives.py` | Done | Numerical and semantic tests pass |
| Instrumentation | `instrumentation.py` | Done | Exercised by training tests |
| Trainer | `trainer.py`, `test_trainer.py` | Done | BEST/LAST and RAND_BN tests pass |
| Memory/scoring | `memory.py`, `scoring.py` | WIP | Direct vendor parity test pending |

### 2026-07-16 13:33 - Foundation, replay, vendor, data, memory, and scoring verified

- **Completed**: implemented and reviewed `schemas.py`, `config.py`, `vendor.py`, `feature_data.py`, `label_data.py`, `replay.py`, `memory.py`, and `scoring.py` plus their focused tests. The current suite reports 25 passed and one expected runner-pending skip.
- **Issues**: the frozen PaAno loader is not portable to ordinary Windows backslash paths; this was already identified in Phase D. No other schema or tensor incompatibility was found.
- **Solution**: the overlay uses `pathlib`, a frozen hash manifest, feature-only `pandas.usecols`, and `[N,C,96]` unfold views. Direct vendor tests verify encoder surfaces and memory/score parity without changing vendor code.
- **Run-manual impact**: added the current full-suite command below.

## Implementation Progress - Snapshot 6

| Module | Files | Status | Notes |
|---|---|---|---|
| Contracts/config/vendor | `schemas.py`, `config.py`, `vendor.py` | Done | Frozen protocol/SHA and model surfaces tested |
| Data/labels/replay | `feature_data.py`, `label_data.py`, `replay.py` | Done | Label isolation and deterministic replay tested |
| Memory/scoring | `memory.py`, `scoring.py` | Done | Direct numerical parity tests pass |
| Current test suite | all current tests | Done | 25 passed, 1 runner-pending skip |

### 2026-07-16 13:36 - Overlay installed in editable mode

- **Completed**: installed `paano-k0==0.1.0` from `code/pyproject.toml` into the fresh `paano_msn` environment.
- **Issues**: none.
- **Solution**: editable installation keeps launch scripts and tests bound to the current reviewed source tree.
- **Run-manual impact**: the environment setup command is now run-verified.

### 2026-07-16 13:40 - Label-free runner integrated; full unit suite clean

- **Completed**: integrated `run_series.py` and `test_experiment_coverage.py`. The runner now commits trained BEST/LAST or RAND_BN scores with shared initialization/replay provenance, writes family-aware training summaries, and fails with retained `_FAILED.json` rather than skipping. The complete suite now passes 28/28 with no skip.
- **Issues**: no unit-level mismatch. End-to-end CUDA execution remains to be verified by the frozen NAB smoke.
- **Solution**: keep runner/evaluator status WIP until one actual CUDA trajectory, both checkpoints, independent evaluation, and artifact hashes pass.
- **Run-manual impact**: full suite command now expects no skip; runner CLI will be documented after smoke.

## Implementation Progress - Snapshot 7

| Module | Files | Status | Notes |
|---|---|---|---|
| Runner | `run_series.py`, coverage test | WIP | 28/28 unit suite passes; CUDA smoke pending |
| Evaluator | `evaluate_scores.py`, label-order test | WIP | Hash-before-label test passes; real metrics pending |
| All unit tests | 28 tests | Done | No failures or skips |

### 2026-07-16 13:44 - First full CUDA smoke succeeded; replay warning removed

- **Completed**: ran the real NAB `OFFICIAL` trajectory for all 100 iterations on RTX 4090 and committed `BEST`/`LAST` checkpoints and score artifacts with trajectory `_SUCCESS`. All 100 instrumentation rows were written.
- **Issues**: PyTorch warned when wrapping the intentionally read-only replay anchor array with `torch.from_numpy`; no mutation or numerical failure occurred.
- **Solution**: copy only the integer anchor index buffer before tensor wrapping. Replay values/hashes, model inputs, and scientific semantics are unchanged. A second independent smoke will verify identical checkpoint/score hashes.
- **Run-manual impact**: runner command will be added after the repeatability smoke and evaluation pass.

## How to Run

### Environment setup

```powershell
D:\Anaconda\envs\paano_msn\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
D:\Anaconda\envs\paano_msn\python.exe -m pip install -r C:\Users\qintian\Desktop\msn\msnmsn\code\requirements.txt
D:\Anaconda\envs\paano_msn\python.exe -m pip install -e C:\Users\qintian\Desktop\msn\msnmsn\code
```

- **What happens**: installs the CUDA-specific PyTorch wheel separately, installs non-PyTorch dependencies, then installs the local overlay package in editable mode.
- **Output**: environment `D:/Anaconda/envs/paano_msn`; no dataset or experiment artifact is created.

### Focused artifact-contract test

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn\code
D:\Anaconda\envs\paano_msn\python.exe -m pytest -q tests\test_artifact_contract.py
```

- **What happens**: tests atomic score commit, success-marker ordering, round-trip schema verification, and mutation detection.
- **Output**: pytest console result only; temporary artifacts are isolated by pytest.

### Focused aggregate-decision test

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn\code
D:\Anaconda\envs\paano_msn\python.exe -m pytest -q tests\test_aggregate_decision.py
```

- **What happens**: validates strict gate boundaries, terminal-outcome precedence, real frozen config parsing, and failure on incomplete metric coverage.
- **Output**: pytest console result only.

### Current full unit-test suite

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn\code
D:\Anaconda\envs\paano_msn\python.exe -m pytest -q
```

- **What happens**: validates frozen config/vendor guards, feature-only input, replay identity, objective and trainer semantics, memory/scoring parity, artifact transactions, and aggregate decisions.
- **Output**: pytest summary; before `run_series.py` lands, exactly one runner-surface test is expected to skip.

### Focused objective and trainer tests

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn\code
D:\Anaconda\envs\paano_msn\python.exe -m pytest -q tests\test_objectives.py tests\test_trainer.py
```

- **What happens**: checks official triplet transcription, paper anchor-space negatives, pretext schedule, gradient non-interference, BEST/LAST timing, and RAND_BN buffer-only updates.
- **Output**: pytest console result only.

### 2026-07-16 14:05 - Independent CUDA smoke repeat is bitwise reproducible

- **Completed**: repeated the complete 100-iteration NAB `OFFICIAL` CUDA trajectory in an independent output directory and evaluated both BEST and LAST score artifacts.
- **Issues**: none after copying the read-only replay index view before tensor wrapping; the repeat stderr is empty.
- **Solution**: compared initialization hash, replay hash, BEST/LAST checkpoint hashes, memory hashes, score hashes, best iteration, and evaluator metrics across both runs. Every compared value is identical. Both selected BEST iteration 20.
- **Observed smoke metrics**: BEST VUS-PR `0.854310`, AUPRC `0.890501`, VUS-ROC `0.987592`; LAST VUS-PR `0.910388`, AUPRC `0.925941`, VUS-ROC `0.993031`. These values are a single-file execution smoke, not a paper headline comparison.
- **Run-manual impact**: the primary, evaluator, and aggregator scripts are now syntax-checked and documented in `code/README.md`.

## Implementation Progress - Snapshot 8

| Module | Files | Status | Notes |
|---|---|---|---|
| CUDA runner | `run_series.py`, primary launcher | Done | Two independent full smoke runs match bitwise |
| Evaluator | `evaluate_scores.py` | Done | Hash commit is verified before labels are loaded |
| PowerShell orchestration | `code/scripts/*.ps1` | Done | Six scripts parse; monitor and confirmation logic reviewed |
| Full implementation tests | `code/tests/` | Done | 28 passed, no skips |

### Verified primary workflow

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn
powershell -ExecutionPolicy Bypass -File .\code\scripts\01_run_primary_k0.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\02_evaluate_primary_k0.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\03_aggregate_decision.ps1
```

- **What happens**: runs the frozen 24 primary jobs, commits 42 label-free score artifacts, evaluates them only after score-hash verification, then applies the preregistered decision gates.
- **Output**: raw runs under `results/k0` (ignored by Git), compact aggregate tables and the terminal decision are exported for the final research record.

### 2026-07-16 13:44 - Primary K0 completed and stopped at the frozen performance gate

- **Completed**: all 24/24 primary jobs and 42/42 score evaluations completed with zero failures. The preregistered aggregator returned `STOP_NO_PERFORMANCE_HEADROOM`.
- **Mechanism evidence**: all six families have post-pretext median active-hinge fraction `0.0`, and every OFFICIAL run selects BEST at iteration 20.
- **Performance evidence**: LAST-vs-BEST macro VUS-PR is `+0.009059` but macro AUPRC is `-0.008610` and worst-family VUS-PR is `-0.259217`; PAPERNEG and NONOVERLAP have negative macro deltas.
- **Decision**: do not run confirmation seeds, do not freeze a method, and do not add a rescue module. Preserve the result as a mechanism-only failure record.
- **Artifacts**: copied the exact aggregate decision, file/family metrics, and paired contrasts into `artifacts/paano_k0/`; added the full English result record at `docs/experiments/PAANO_K0_RESULTS.md`.

### 2026-07-16 14:10 - Iteration #1: full paper-reference benchmark authorized

**Reason**: the six-file K0 established the mechanism but failed its matched performance gate. The user explicitly confirmed a Phase F experiment-design change: evaluate the already registered project arm over the complete paper-compatible U/M Eval lists, compare against paper-reported PaAno values, and run only component ablations.

**Changes**:
- `docs/user_requirements.md`: records the user-confirmed full-benchmark override, paper-reported comparison policy, conditional seeds, and minimum ablation scope.
- `docs/idea_report.md`: adds the Phase F diagnostic, experiment-design backtrack, frozen full arm, ablations, and terminal outcome boundary.
- `docs/implementation.md`: specifies the generic full-manifest, runner, evaluator, aggregator, scripts, tests, outputs, and execution order.

**Expected effect**: produce manuscript-scale U/M numbers without reproducing the paper's full baseline suite, while keeping the score-before-label boundary and a fixed LAST endpoint.

**Document sync**: idea_report.md yes | implementation.md yes | configs no (existing scientific protocol reused unchanged)

### 2026-07-16 14:18 - Iteration #1: generic full manifest and label-free series runner

**Reason**: the K0 loader is intentionally restricted to six unique families and cannot safely be reused for the complete 350/180 benchmark.

**Changes**:
- `code/src/paano_k0/benchmark_manifest.py`: adds fixed Eval-list ingestion, path traversal and duplicate checks, filename-derived family/train split parsing, per-file shape/byte/SHA verification, full 530-row loading, and efficient one-series verification for worker processes.
- `code/src/paano_k0/run_benchmark_series.py`: adds a label-free full-benchmark CLI restricted to the three registered trained trajectories and seeds 2027-2029; it delegates unchanged model execution to the tested `run_job`.

**Expected effect**: enable immediate full U/M execution without weakening the six-file K0 contract or adding labels to the runner surface.

**Document sync**: idea_report.md yes | implementation.md yes | configs no

### 2026-07-16 14:30 - Iteration #1: full-manifest and runner contract tests

**Reason**: protect the two full-benchmark extensions that differ structurally from K0: repeated families in a 530-row manifest and a separate label-free worker CLI.

**Changes**:
- `code/src/paano_k0/benchmark_manifest.py`: derives the unique-series coverage assertion from the frozen track-count mapping, enabling compact synthetic contract tests without changing production counts.
- `code/tests/test_benchmark_manifest.py`: tests repeated-family support, one-series lazy verification, and duplicate-series rejection.
- `code/tests/test_run_benchmark_series.py`: tests that the worker has no label surface and rejects the diagnostic RAND_BN arm.

**Expected effect**: prevent full-run coverage drift and accidental label/arm expansion before long execution.

**Document sync**: idea_report.md yes | implementation.md yes | configs no

### 2026-07-16 14:25 - Iteration #1: main launcher and 15-minute monitor

**Reason**: start the full method on GPU as soon as the canonical 350/180 manifest is available, while evaluator and ablation packaging continue independently.

**Changes**:
- `code/scripts/05_run_full_main.ps1`: adds exact 530-job seed-2027 `PAPERNEG_NONOVERLAP` execution with U-first ordering, fail-fast retained errors, and safe resume of committed trajectories.
- `code/scripts/monitor_full.ps1`: adds read-only main-arm coverage, current-series, GPU, and disk monitoring.
- `docs/TSB_AD_FULL_EVAL_MANIFEST.csv`: generated from the fixed 350/180 Eval filename lists; every local file records byte count, shape, train boundary, and SHA-256.

**Expected effect**: keep the RTX 4090 occupied on the frozen full method while remaining Phase F implementation work proceeds.

**Document sync**: idea_report.md yes | implementation.md yes | configs no
### 2026-07-16 15:02 - Iteration #1: full ablation and evaluation launch surfaces

**Reason**: complete the frozen Phase F full-benchmark execution surface while the main `PAPERNEG_NONOVERLAP` jobs run, without changing model, objective, endpoint, data, or comparison semantics.
**Changes**:
- `code/scripts/06_run_full_ablations.ps1`: adds fail-fast, resumable seed-2027 execution for exactly 530 `PAPERNEG` and 530 `OFFICIAL` trajectories.
- `code/scripts/07_evaluate_full.ps1`: requires exact 1,590-trajectory/LAST-score coverage, then invokes the planned evaluator-only and compact aggregation CLIs and validates all registered manuscript outputs.
- `code/scripts/monitor_full.ps1`: adds backward-compatible `-Mode main|ablations` counting and current-job reporting while preserving `main` as the default.
**Expected effect**: make the approved component ablations and exact LAST-only 530-file evaluation executable immediately after main coverage completes, with retained failures and no silent file removal.
**Document sync**: idea_report.md yes | implementation.md yes | configs/ no

### 2026-07-16 15:08 - Iteration #1: full benchmark evaluator and aggregator

**Reason**: complete the label-isolated reporting path for the registered 530-series, three-arm, seed-2027 full benchmark without adding trajectories or changing runner/model behavior.

**Changes**:
- `code/src/paano_k0/evaluate_benchmark.py`: requires requested registered trajectories at LAST, verifies every expected committed score hash and provenance before the first label read, reuses each series label across arms, and writes per-file metrics.
- `code/src/paano_k0/aggregate_benchmark.py`: requires exact three-arm metric coverage, writes file/family/track/overall and runtime tables, and gates the main `PAPERNEG_NONOVERLAP-LAST` file means against the external paper-reported Table 15 default full-Eval values U=`0.5296`, M=`0.4263` using strict `>` comparisons.
- `code/tests/test_benchmark_evaluator.py`: covers global hash preflight, one label read per series, registered-arm enforcement, and LAST-only enforcement.
- `code/tests/test_benchmark_aggregate.py`: covers complete aggregation outputs, exact external reference values, strict gate boundaries, and incomplete-coverage failure before runtime scanning.

**Tests**:
- `D:\Anaconda\envs\paano_msn\python.exe -m pytest -q tests\test_benchmark_evaluator.py tests\test_benchmark_aggregate.py` -> `7 passed in 1.98s`.
- `D:\Anaconda\envs\paano_msn\python.exe -m pytest -q` -> `39 passed in 7.69s`.

**Expected effect**: the long-running label-free score artifacts can be evaluated only after complete hash preflight, and manuscript tables cannot silently use the Table 12 k=1 M=`0.431` value as the default-paper comparator.

**Document sync**: Phase F implementation contract already present; paper-reference prose correction coordinated separately | configs unchanged

### 2026-07-16 14:36 - Iteration #2: replay float32 open-endpoint correction

**Reason**: the 60th full-main series retained a fail-fast record when one valid NumPy float64 uniform (`0.9999999801825278`) rounded to the closed endpoint `1.0` during float32 storage, violating the registered `[0,1)` replay invariant before training began.

**Changes**:
- `docs/implementation.md`: freezes the representation-only endpoint canonicalization before code modification.
- `code/src/paano_k0/replay.py`: preserves the original float64 RNG calls and stream, then maps only float32 values rounded to `1.0` to `nextafter(float32(1), float32(0))`.
- `code/tests/test_replay.py`: adds a direct endpoint regression and the exact failed MITDB replay-seed regression.

**Validation**:
- Focused replay tests: `8 passed`.
- Full suite: `41 passed`.
- Exact old/new replay comparison on the failed seed found one and only one representation difference: iteration 94, unadjacent draw `[385,0]`, `1.0 -> 0.9999999403953552`; all other stored draws and RNG ordering were unchanged.

**Expected effect**: allow the valid final-candidate draw to be materialized without changing the experiment arm, seed, optimizer, data, model, or any completed score artifact.

**Document sync**: implementation.md yes | idea_report.md no semantic change | configs unchanged

### 2026-07-16 17:19 - Iteration #3: full-result finalization and conditional confirmation pipeline

**Reason**: use the running GPU ablation window to complete the already authorized, non-GPU manuscript-data closure path so evaluation, reporting, tests, and conditional seeds can start without an implementation gap.

**Changes**:
- `docs/implementation.md`: specifies the compact-only report renderer, finalization script, and the previously frozen conditional seeds 2028/2029 aggregation before their code was written.
- `code/src/paano_k0/report_benchmark.py` and `code/tests/test_benchmark_report.py`: strictly validate the seven compact aggregate inputs and atomically render the complete English numeric report without dataset, label, raw-score, or evaluator imports.
- `code/scripts/08_finalize_full.ps1`: chains exact full evaluation, compact validation, report rendering, the complete test suite, and Git-facing output checks without committing or filtering results.
- `code/src/paano_k0/aggregate_confirmation.py` and `code/tests/test_confirmation_aggregate.py`: preserve all 530 files for each registered seed and report fixed three-seed U/M means and population standard deviations without a new success gate.
- `code/scripts/09_run_full_confirmation.ps1` and `code/scripts/10_evaluate_confirmation.ps1`: refuse execution unless seed 2027 returns `CONTINUE_FULL_CONFIRMATION`, then run/evaluate only the frozen main arm for seeds 2028/2029 and reuse the complete seed-2027 metrics.
- `README.md` and `code/README.md`: update the reproducible full-benchmark and finalization commands while retaining the negative K0 status.

**Validation**:
- Complete Python suite: `50 passed in 7.63s` while GPU ablations continued independently.
- PowerShell 5.1 AST parse: scripts 08/09/10 all passed (`947`, `788`, and `1116` tokens; zero errors).
- No current Eval label or partial metric was read by this iteration.

**Expected effect**: when all 1,590 seed-2027 LAST scores are complete, the task can immediately produce all numeric paper tables and a transparent stop/confirmation decision; confirmation runs remain impossible unless both fixed paper comparisons pass.

**Document sync**: implementation.md yes | idea_report.md unchanged (conditional seeds already frozen) | configs unchanged

### 2026-07-16 17:34 - Iteration #4: packaging cross-review provenance hardening

**Reason**: an independent static cross-review found that the compact renderer did not independently bind its 530 rows to the canonical Eval membership, the Git SHA could be attached to dirty core code, and the import-boundary regression test checked only relative imports.

**Changes**:
- `report_benchmark.py`: validates a frozen SHA-256 over canonical `track/family/series_id/data_sha256` membership while remaining compact-only.
- `08_finalize_full.ps1`: refuses to render unless the scoring/evaluation/report code, frozen config, and full manifest match the reported Git `HEAD`.
- `test_benchmark_report.py`: adds a noncanonical-membership rejection test and checks relative, absolute project, and third-party imports against an explicit standard-library allowlist.

**Expected effect**: the manuscript-facing report now proves exact registered membership and identifies committed code without reopening data or labels.

**Document sync**: implementation.md unchanged | idea_report.md unchanged | configs unchanged

### 2026-07-16 18:31 - Iteration #5: persistent goal and paper-delivery requirements

**Reason**: the user enabled Codex goal mode and authorized continuous execution from the running full benchmark through evidence-gated iteration, ResearchPilot G.0--G.7 writing, Tectonic compilation, and final repository delivery.

**Changes**:
- `docs/user_requirements.md`: records the installed ResearchPilot workflow as primary, retains 15-minute GPU monitoring, defines bounded evidence-driven failure iterations, and registers the auxiliary paper workflow and Tectonic path.
- No model, data, score, metric, experiment parameter, or current Eval artifact was changed or read.

**Expected effect**: long-running experiments and later paper stages can advance without routine user gates while preserving scientific stop rules and venue-specific requirements.

**Document sync**: user_requirements.md yes | idea_report.md unchanged | implementation.md unchanged | configs unchanged

### 2026-07-16 18:49 - Iteration #6: freeze MSN paper-format authority

**Reason**: the auxiliary paper workflow contains LNCS examples, whereas the target venue requires IEEE Computer Society conference formatting. The paper infrastructure needs a verified venue-specific authority before G.0 begins.

**Changes**:
- `docs/MSN2026_SUBMISSION_REQUIREMENTS.md`: records the official eight-page inclusive double-blind IEEE format, Big Data and AI track, dates, EasyChair entry, IEEEtran contract, and local build-tool hashes.
- The venue-hosted 2024 template link was recorded as temporarily returning HTTP 404; the official CTAN IEEEtran package was downloaded and hash-verified as a fallback, with a mandatory pre-submission author-kit retry.
- No manuscript claim, model, experiment, score, label, or current full-benchmark result was changed or read.

**Expected effect**: G.0 can initialize an independent IEEE manuscript without inheriting the generic workflow's incompatible LNCS template.

**Document sync**: user_requirements.md unchanged | idea_report.md unchanged | implementation.md unchanged | configs unchanged

### 2026-07-16 18:50 - Iteration #6 validation: IEEEtran/Tectonic smoke build

- Source: CTAN `bare_conf.tex` using `\documentclass[conference]{IEEEtran}`.
- Compiler: Tectonic 0.16.9 at the frozen local path.
- Result: exit code 0, two TeX passes, PDF emitted (`26,118` bytes), and zero matched undefined-reference/citation, overfull, or error log lines.
- Build directory: external `D:/qintian_experiments/latex_smoke/ieeetran-20260716-184749`; no build products were added to Git.
- Non-blocking environment warning: Fontconfig reported no default configuration after the PDF was successfully written. Final visual QA must verify that this does not affect manuscript fonts.

### 2026-07-16 18:54 - Iteration #7: pre-G.0 claim--evidence freeze

**Reason**: paper infrastructure can be prepared during the GPU run, but manuscript claims must remain bounded by the completed six-file K0 and must not anticipate unfinished full-Eval metrics.

**Changes**:
- `docs/PAPER_CLAIM_EVIDENCE_PRE_GATE.md`: separates K0-supported observations, claims blocked on full results, hypotheses ruled out by K0, and claims that the current full design cannot establish.
- Explicitly prevents the legacy K0 M value `0.431` from entering the manuscript; the corrected external Table 15 value is `0.4263`.
- No incomplete full-Eval metric, raw score, or label was opened.

**Expected effect**: G.0 starts from a conservative evidence map, and a negative full result triggers a bounded Phase F diagnosis instead of post-hoc claim rewriting.

**Document sync**: idea_report.md unchanged | implementation.md unchanged | configs unchanged

### 2026-07-16 18:57 - Iteration #7 cross-review corrections

- Clarified that only cross-seed means/variance/stability remain blocked; the complete seed-2027 file-weighted mean is a valid endpoint.
- Restricted frozen full endpoints to VUS-PR, AUPRC, and VUS-ROC.
- Added the TimesURL/SoftCLT, PAI, and DADA novelty boundaries and recorded that triplet division by 10 is a paper-defined setting, not a bug.
- Tightened the STOP transition: diagnosis precedes any new iteration, which requires independent evidence, no material scope expansion, preregistration, and no Eval-label tuning.

### 2026-07-16 18:58 - Iteration #8: isolate manuscript build products

**Changes**:
- `.gitignore`: excludes only the repository-level `.latex-build/` tree while leaving manuscript sources and explicitly checked release PDFs trackable.

**Expected effect**: Tectonic intermediates cannot pollute result or manuscript commits.

**Document sync**: `MSN2026_SUBMISSION_REQUIREMENTS.md` already specifies the isolated build contract.

### 2026-07-16 19:24 - Iteration #9: pre-G.0 bibliography and comparator provenance

**Reason**: stage the complete Related Work and comparison-table citation set
while the full GPU ablations run, without initializing a manuscript before the
user supplies the official IEEE MSN author kit.

**Changes**:
- `docs/bibliography/candidate_references.bib`: records 35 candidate entries
  with a one-sentence core contribution and an explicit reason for citation.
- `docs/bibliography/PAANO_TABLE23_CITATION_MAP.csv`: maps every distinct
  PaAno Table 2/3 comparator and alias to one canonical BibTeX key.
- `docs/bibliography/CITATION_SOURCE_MANIFEST.csv`: binds all 35 entries to a
  canonical source, verification status, role, and any venue or wrapper caveat.
- `docs/bibliography/RELATED_WORK_CITATION_PLAN.md`: freezes a recent-first
  prewriting structure led by 2025--2026 work; older references are limited to
  irreplaceable method, metric, module, or evaluation provenance.
- Formal publication metadata replaces PaAno's preprint-year labels for
  TimesNet (2023), FITS (2024), and Anomaly Transformer (2022). MOMENT,
  TimesFM, OFA, DeepAnT, and USAD metadata were corrected against primary
  proceedings or publisher records.

**Provenance**:
- Official PaAno OpenReview PDF: 1,445,413 bytes, SHA-256
  `4860AAAEEEB04114464A98588924AAF5FB97FFE42CD769A1207D0085ECC00689`.
- PaAno arXiv-v3 source archive (mapping aid only): SHA-256
  `37C4D7D47F975FDF1C68551B4054BDBF7CD6CA3827F1F9E5908251C116551356`.
- All numerical comparison rows copied later from PaAno remain explicitly
  `PaAno-paper-reported`; citing an original model paper does not make a row a
  local reproduction.

**Validation**:
- 35 entries, 35 unique keys, zero missing ResearchPilot citation annotations,
  zero missing PaAno-map keys, and balanced BibTeX braces.
- External Tectonic 0.16.9 `article`/BibTeX syntax smoke: exit 0; BibTeX read
  the candidate database with no warning, error, missing, or undefined line.
- The smoke build is syntax-only and does not substitute for the pending
  venue-author-kit manuscript build.

**Expected effect**: G.0 can copy only used, primary-verified entries into the
final manuscript bibliography and keep recent narrative work separate from
older comparator provenance.

**Document sync**: bibliography staging yes | manuscript not initialized |
idea_report.md unchanged | experiment artifacts unread

### 2026-07-16 19:55 - Iteration #10: conditional-confirmation monitoring coverage

**Reason**: the registered full benchmark can conditionally launch seeds 2028
and 2029, but the existing read-only monitor distinguished only the seed-2027
main and ablation stages. A 15-minute confirmation automation therefore needed
an exact progress mode before the result gate could authorize that branch.

**Changes**:
- `code/scripts/monitor_full.ps1`: adds a read-only `confirmation` mode that
  counts only `PAPERNEG_NONOVERLAP` runs under seeds 2028/2029, reads the
  dedicated confirmation launcher log, and reports progress over exactly 1,060
  registered runs.
- `docs/implementation.md`, `code/README.md`, and `README.md`: document the
  conditional launch/evaluation sequence and the three monitor modes.
- No model, checkpoint, dataset, label, score, metric, hyperparameter, seed
  gate, or unfinished full-Eval artifact was changed or inspected.

**Expected effect**: if the frozen seed-2027 gate passes, the required
15-minute automation can monitor confirmation without contaminating progress
with already completed seed-2027 artifacts.

**Document sync**: implementation.md yes | README files yes |
user_requirements.md citation policy independently updated | configs unchanged

**Validation**:
- PowerShell AST parse: pass.
- Backward-compatible live ablation monitor: `537/1060`, zero failures,
  current launcher event resolved, runner alive.
- Confirmation isolation check before launch: `0/1060`, zero failures, no
  seed-2027 artifact counted.
- Complete Python suite: `51 passed`.
- `git diff --check`: pass (line-ending notices only).

### 2026-07-16 20:02 - Iteration #11: freeze recent-first citation balance

**Reason**: the user requires the paper's substantive Related Work to be led
by 2025--2026 research, with very few 2024 papers and older work retained only
when its provenance cannot be replaced.

**Changes**:
- `docs/user_requirements.md`: records the 2024--2026 citation boundary and
  prohibits pre-2024 work from supporting current-novelty claims.
- `docs/bibliography/RELATED_WORK_CITATION_PLAN.md`: freezes ten narrative
  citations: eight from 2025--2026 and two indispensable 2024 pair-semantics
  precedents. TSB-AD and rigorous-evaluation sources remain protocol evidence
  outside Related Work.
- Bibliography metadata and PaAno Table 2/3 identity mappings were reconciled
  against primary sources; external values remain explicitly paper-reported.

**Validation**:
- Narrative balance: 8/10 (80%) from 2025--2026, 2/10 (20%) from 2024, and
  zero pre-2024 narrative keys.
- All ten narrative keys resolve in the 35-entry BibTeX database and primary
  source manifest.
- BibTeX/Tectonic syntax smoke: pass; exact BibTeX/manifest coverage: 35/35;
  PaAno table map: 22 rows and 21 distinct resolved keys.
- No unfinished experiment result, score, or label was opened.

**Expected effect**: the final English paper uses current literature for its
research argument while preserving necessary historical citations only at the
metric, protocol, component, or comparison-identity locations they support.

### 2026-07-16 19:55 - Iteration #12: harden conditional confirmation and registered reporting

**Reason**: an independent Phase F review found that a stale positive decision
could authorize confirmation under changed config/vendor state, a resume skip
checked only file existence, and recursive progress counting could include
stale run directories. The same review found that the manuscript-facing report
still displayed AUROC although the frozen full endpoints are VUS-PR, AUPRC,
and VUS-ROC.

**Changes**:
- `confirmation_guard.py`: adds label-free decision authorization and
  score-hash/run-provenance/summary validation for every confirmation resume
  skip and every freshly committed confirmation run.
- `09_run_full_confirmation.ps1`: calls the guard before any compute and before
  accepting a completed run.
- `monitor_full.ps1`: enumerates expected paths from the exact 350-U/180-M
  manifest instead of recursively counting every matching directory.
- `report_benchmark.py`: retains AUROC only in internal schema-compatible
  artifacts and removes it from the manuscript-facing Markdown tables.
- Added a discriminating unequal-family-size regression test proving that
  track aggregation is file-weighted rather than equal-family macro.
- The active seed-2027 config, vendor, model, dataset, score artifacts, labels,
  and experiment process were not changed or reopened.

**Validation**:
- PowerShell AST parse: pass for launcher and monitor.
- Targeted confirmation/aggregate/report tests: `16 passed`.
- Complete suite: `54 passed`.
- Live manifest-exact monitor: main `530/530`, confirmation `0/1060`, with no
  stale seed-2027 result counted in confirmation.
- Active ablation process remained healthy and continued without restart.
- `git diff --check`: pass (line-ending notices only).

**Expected effect**: an authorized confirmation can run and resume without
silently accepting stale/corrupt artifacts, while the eventual paper report
contains only preregistered endpoints.

### 2026-07-16 20:33 - Iteration #13: distinguish runtime bytecode from vendor drift

**Reason**: the active PaAno jobs generate untracked `__pycache__/*.pyc` files
inside the SHA-frozen vendor checkout. Treating this runtime-only state as a
source modification would incorrectly block an otherwise authorized
confirmation run, while accepting every dirty checkout would weaken the
config/vendor binding introduced in Iteration #12.

**Changes**:
- `confirmation_guard.py`: permits only untracked Python bytecode below an
  `__pycache__` directory and still rejects every tracked change, untracked
  source file, or other untracked artifact before confirmation compute.
- `test_confirmation_guard.py`: adds positive coverage for runtime bytecode and
  negative coverage for tracked source drift, an untracked Python module, and
  a non-bytecode file placed under `__pycache__`.

**Expected effect**: seed-2028/2029 confirmation remains bound to the exact
PaAno source revision without being blocked by harmless caches created by the
ongoing seed-2027 jobs.

**Document sync**: implementation.md already specifies semantic vendor
cleanliness | idea_report.md unchanged | configs unchanged

### 2026-07-16 20:08 - Iteration #13 validation and timestamp correction

- The preceding Iteration #13 header was entered as `20:33`; the host clock
  shows the work was completed at `20:08`. This append-only correction retains
  the original entry rather than silently rewriting it.
- Temporary-Git-repository guard tests: `9 passed`; this includes ignored
  untracked source rejection and Unicode/space bytecode-path acceptance.
- Complete Python suite: `61 passed in 10.15s`.
- Python compilation, PowerShell AST parsing, and `git diff --check`: pass
  (line-ending notices only).
- The stricter guard accepted the actual frozen vendor at
  `d4c67116190efa4592dc6a8a157ced0def68b6af`; its only untracked files are
  direct runtime bytecode caches.
- Manifest-exact live monitoring continued at `662/1060`, zero failures,
  without restarting or modifying the active seed-2027 process.

### 2026-07-16 20:10 - Iteration #13 independent review

- Independent read-only review: `PASS`.
- The reviewer confirmed that tracked staged/unstaged changes, ordinary and
  ignored untracked files, nested bytecode, and non-bytecode cache files all
  fail closed; direct runtime bytecode with Unicode/space paths remains valid.
- A future Windows-junction-specific regression was identified as optional and
  non-blocking; the current real checkout contains no such path.

### 2026-07-16 20:14 - Operational disk-space floor

- `docs/user_requirements.md`: records the user-specified C-drive hard floor of
  `20 GiB` and the protected active-experiment paths.
- Updated the existing 15-minute heartbeat automation to trigger safe cleanup
  or migration below the floor; it must notify rather than delete when no
  clearly safe candidate exists.
- Current read-only status: C `77.86 GiB`, D `828.84 GiB`, ablations
  `676/1060`, zero failures; no cleanup was necessary and no experiment file
  was touched.

### 2026-07-16 20:23 - Iteration #14: available-RAM safety floor design

**Reason**: the user specified a second `20 GiB` floor for available physical
memory during the long benchmark. Disk free space and RAM pressure are
different resources, so the existing disk-only status cannot enforce this
operational constraint.

**Changes**:
- `docs/user_requirements.md`: records the available-physical-RAM floor and
  fail-safe response; the active PaAno runner is explicitly protected.
- `docs/implementation.md`: extends the read-only full-run monitor contract to
  report available physical RAM without changing experiment state.

**Expected effect**: the 15-minute monitor can stop unrelated parallel work
before memory pressure becomes critical while preserving the frozen runner,
scores, data, and configuration.

**Document sync**: user_requirements.md yes | implementation.md yes | configs
unchanged | idea_report.md unchanged

### 2026-07-16 21:09 - Manuscript tool isolation and text-only boundary

- Moved the verified Tectonic 0.16.9 executable from the Downloads folder to
  `D:/qintian_tools/tectonic/0.16.9/tectonic.exe`; SHA-256 remains
  `A0A9A5EAF1A940D9A615AD78D35225CA59420C7984576C6402FFFB3E9FB05CEB`.
- Updated the manuscript and venue documentation to use the isolated tool and
  repository-level `.latex-build/` output directory.
- Reconfirmed the user-owned figure boundary: Codex will write manuscript
  prose, equations, BibTeX, and textual tables only; no figure assets or
  figure-generation scripts will be created in this workflow.
- The active PaAno ablation runner, its model, data, protocol, and parameters
  were not modified.

**Document sync**: user_requirements.md yes | manuscript README yes |
submission requirements yes | implementation.md unchanged | configs unchanged

### 2026-07-16 21:11 - Text-only LaTeX source enforcement

- Removed the unused `graphicx` dependency from the provisional manuscript.
- Tightened the drafting boundary: no figure asset, figure-generation script,
  figure environment, or placeholder graphic is created before the user
  supplies the final figures.
- No experiment code, score, model, data, protocol, or active process changed.

**Document sync**: user_requirements.md yes | manuscript source yes |
implementation.md unchanged | configs unchanged

### 2026-07-16 21:13 - Isolated Tectonic build validation

- Compiled `docs/manuscripts/msn2026/main.tex` with the relocated verified
  binary and wrote all generated files to `.latex-build/msn2026/`.
- Build exit: 0; BibTeX and required TeX reruns completed; output is one
  612-by-792-point US-Letter page.
- Log scan: zero undefined citations/references, missing files, overfull boxes,
  or LaTeX fatal errors. The existing underfull bibliography warning from long
  URLs remains non-blocking.
- Source scan: no figure environment, image include, graphic path, or figure
  reference exists in `main.tex` or `sections/*.tex`.

**Validation**: isolated build pass | text-only source pass | active experiment
unchanged

### 2026-07-16 21:19 - Pre-G.0 format metadata reconciliation

- Updated the provisional manuscript README to acknowledge the recovered
  CFP-linked venue kit while retaining the user-supplied generic IEEE bundle as
  a replaceable structural reference.
- Updated the Related Work inventory status: the formatting scaffold exists,
  but scientific drafting remains blocked on the terminal Phase F gate.
- Added `docs/manuscripts/examples/style-notes.md`, a text-only record of the
  ten supplied MSN 2025 papers' reusable layout conventions and the publisher
  elements that must not enter the anonymous review source.
- No result, score, label, model, protocol, parameter, or active process was
  changed or inspected.

**Document sync**: manuscript readiness yes | bibliography status yes |
implementation.md unchanged | configs unchanged

### 2026-07-16 21:30 - Terminal claim-gate design freeze

**Reason**: independent readiness review found that the external performance
branch is machine-checked, while component attribution and the fixed-three-seed
claim branches remained manual manuscript checks.

**Design changes**:
- `docs/implementation.md`: specifies a standard-library-only compact claim
  gate, its terminal-only invocation points, and separate Git-facing artifact.
- `docs/PAPER_CLAIM_EVIDENCE_PRE_GATE.md`: freezes strict raw-value comparisons,
  the component tie rule, three confirmation branches, prohibited claims, and
  the no-interim-artifact rule.

**Expected effect**: after the final compact results exist, the paper claim
boundary is reproduced mechanically without reopening labels, raw scores,
datasets, or caches and without changing any experiment decision.

**Document sync**: implementation.md yes | claim-evidence policy yes |
idea_report.md unchanged | configs unchanged

### 2026-07-16 21:43 - Terminal claim-gate implementation and validation

**Implementation**:
- Added `code/src/paano_k0/claim_gate.py`, a standard-library-only terminal
  audit over the frozen compact seed-2027 and optional confirmation outputs.
- Added `code/tests/test_claim_gate.py` with synthetic fixtures for every
  registered external, component-attribution, and confirmation branch plus
  exact schema, provenance, cross-file, finite-value, deterministic-hash, and
  atomic-write failure checks.
- The audit remains downstream of scripts 08--10 and does not read scores,
  labels, datasets, caches, or authorize compute.

**Validation**:
- Focused claim-gate suite: `35 passed`.
- Complete project suite: `96 passed in 10.65s`.
- `py_compile`, standard-library import audit, and `git diff --check` passed.

**Document sync**: implementation.md yes | claim-evidence policy yes |
user requirements unchanged | configs unchanged

### 2026-07-16 22:49 - Iteration #15 design freeze: resumable parallel evaluator

**Reason**: the frozen 1,590-row evaluator remained correct but processed only
193 atomic rows after the outer finalizer shell timed out and left its child
evaluator alive.  The machine has 32 logical CPUs and sufficient memory, while
the unchanged VUS implementation is CPU-bound and serial.

**Frozen execution-only change**:
- retain the complete all-score preflight before any cache access, worker
  creation, or label read;
- evaluate one series per spawned worker with a hard maximum of four workers
  and one numerical-library thread per worker;
- add fail-closed, provenance-bound resume for valid atomic metric JSONs;
- preserve the exact score artifacts, LAST checkpoint, seed, trajectories,
  labels, `thresholds=250` metric call, canonical ordering, aggregation, gates,
  and paper-reported comparison values;
- require exact shadow parity and full tests before stopping the current serial
  PID or adopting any of its outputs.

**Expected effect**: reduce evaluator wall time without changing any scientific
result.  A mismatch, corrupt cache, provenance ambiguity, RAM-floor violation,
or parity failure aborts the acceleration and preserves the serial evidence.

**Document sync**: implementation.md yes | idea_report.md unchanged |
configs unchanged | manuscript unchanged

### 2026-07-16 23:03 - Iteration #15 implementation and parity validation

**Implementation**:
- `evaluate_benchmark.py` now supports a provenance-bound evaluator contract,
  strict fail-closed metric-cache validation, deterministic series-level
  Windows `spawn` workers, one label read per pending series, atomic per-run
  outputs, and canonical terminal reconstruction.
- Worker selection is capped at four and automatically falls back to two below
  28 GiB available RAM or one below 24 GiB.  BLAS/OpenMP thread counts are
  fixed to one before worker creation.
- Scripts 07 and 10 request four resumable workers without changing registered
  scores, seeds, trajectories, checkpoints, metrics, thresholds, aggregation,
  or gates.
- Added a deterministic U/M shadow parity utility and strict resume/provenance
  unit coverage.

**Validation**:
- focused evaluator suite: `17 passed`;
- complete project suite: `109 passed in 10.65s`;
- Python compilation, PowerShell AST parsing, and `git diff --check`: pass;
- serial versus four-worker shadow check: 4 deterministic short U/M series,
  all 3 registered arms, 12 metric rows, byte-identical metric JSON/CSV/summary;
- partial-resume shadow check: 6 cached rows retained byte-for-byte with
  unchanged mtimes and 6 missing rows reproduced byte-for-byte.

The live serial PID remained untouched throughout design, implementation, and
all parity checks.  Its legacy cache will be preserved rather than adopted
because it predates the evaluator-contract sidecar.

**Document sync**: implementation.md yes | tests yes | scripts yes |
idea_report.md unchanged | configs unchanged | manuscript unchanged

### 2026-07-16 20:45 - Provisional IEEE manuscript scaffold

**Reason**: the user supplied a generic IEEE conference template and ten 2025
MSN papers as a temporary formatting reference while the frozen Phase F
benchmark continues. The 2026 official template remains unavailable to the
user, so formatting provenance must stay explicit and replaceable.

**Changes**:
- `docs/manuscripts/msn2026/README.md`: records the generic-template hashes,
  last-year visual conventions, production-only elements, and replacement
  rules.
- `docs/manuscripts/msn2026/main.tex` and `sections/*.tex`: add an anonymous,
  claim-safe IEEE conference skeleton aligned with future ResearchPilot
  G.1--G.6 outputs.
- `docs/manuscripts/msn2026/references.bib`: includes only the three
  indispensable citations already rendered by the scaffold.

**Expected effect**: formatting and citation compilation can be verified now,
while incomplete scores and unsupported contribution wording remain excluded.
This preparation does not move the project out of Phase F.

**Document sync**: manuscript scaffold yes | implementation.md unchanged |
idea_report.md unchanged | configs unchanged

- Exact template assets added without modification: `IEEEtran.cls` SHA-256
  `C972ACA108FDA004C3514D63658E02816DA2E54D9A1451E870B9BD970E003F55`;
  CTAN `IEEEtran.bst` SHA-256
  `314F0ECE704568FAF827011BAC498650691B2B5EE06320720830E782416D5A5F`.

### 2026-07-16 20:50 - Provisional format and claim-gate validation

- Tectonic 0.16.9 compiled the provisional IEEE source, ran BibTeX, and
  resolved all three citations: exit 0, one US-Letter page, zero overfull box,
  zero missing file, and no undefined citation/reference. The only content
  warning is an underfull bibliography line caused by long provenance URLs;
  IEEEtran/Tectonic also substitutes unavailable TU Times font shapes.
- Rendered-page inspection: anonymous two-column title/abstract/keywords,
  Roman-numeral section hierarchy, numbered references, and US-Letter margins
  are visually intact; no clipping, overlap, black square, publisher header,
  DOI, copyright strip, or download watermark appears.
- Independent pre-gate review identified and closed three claim gaps in
  `PAPER_CLAIM_EVIDENCE_PRE_GATE.md`: one-track-pass wording, component
  attribution under mixed/dominated ablations, and fixed-three-seed stability
  branches. None of these gates changes the running experiment or permits
  Eval-label selection.
- User formatting boundary: manuscript source stays under
  `docs/manuscripts/msn2026/`, Tectonic output stays under `.latex-build/`, and
  Codex drafts text/BibTeX/textual tables only. Manuscript figures are deferred
  to the user and will not be generated during the current workflow.

### 2026-07-16 20:31 - Bibliography readiness and PaAno source sync

- Independent read-only audit: 35 BibTeX entries, 35 unique keys, no duplicate
  DOI/URL/normalized title, no missing type-required field, and a one-to-one
  verified source-manifest match.
- The frozen Related Work plan contains three 2026, five 2025, and two 2024
  narrative sources; no pre-2024 work frames the novelty claim.
- Every PaAno Table 2/3 comparator key resolves. Canonical-algorithm and
  preprint/workshop caveats remain explicit and must be carried into table
  notes and prose.
- `docs/experiments/PAANO_PAPER_REFERENCE.md` now binds the Table 15 external
  values to the official ICLR 2026 OpenReview PDF and its verified local hash;
  the arXiv copy is supplementary only.
- The manuscript `references.bib` remains intentionally uncreated until G.0;
  it will contain only actually cited entries and will be checked for unused
  and undefined keys.

### 2026-07-16 20:27 - Iteration #14 complete validation

- Complete Python suite: `61 passed in 10.29s`.
- PowerShell AST parsing and live execution of the updated monitor: pass.
- `git diff --check`: pass (line-ending notices only).
- The existing 15-minute automation now reports `ram_free_gib` and applies
  the separate `20 GiB` disk and physical-RAM safeguards without terminating
  the active PaAno runner.
- Independent finalization-transition review: `PASS`. The exact 1,060-job
  ablation count, fail-fast 1,590-score evaluation, external-paper-value
  labeling, conditional confirmation authorization, and exact 1,060-job
  confirmation count are consistent. The stage-separated scripts rely on the
  active automation for the `08 -> decision -> 09/10` handoff.

### 2026-07-16 20:25 - Iteration #14 validation and venue-kit recovery

- PowerShell AST parsing of `monitor_full.ps1`: pass.
- Live exact-manifest status: ablations `721/1060`, zero failures; C
  `77.83 GiB`, D `828.69 GiB`, available physical RAM `41.37 GiB`.
- The active runner remained unchanged and continued without restart.
- The CFP-linked MSN 2026 author kit was recovered from its literal HTTP URL;
  the previously attempted HTTPS variant is the source of the 404.
- Frozen local venue kit:
  `C:/Users/qintian/Downloads/IEEE-MSN-2026-author-kit.zip`, 856,412 bytes,
  SHA-256
  `DCE5B5F34EF738CECE3A86A336795394CB06C2345F79E79B2D456F3D61EC9B9F`.
- Its seven-file layout was independently inspected and its sample compiled
  successfully with Tectonic 0.16.9. The included IEEEtran V1.8b class equals
  the local CTAN class after line-ending normalization; CTAN remains the
  source for `IEEEtran.bst` because the venue ZIP omits it.

### 2026-07-16 20:24 - Iteration #14 implementation

**Changes**:
- `code/scripts/monitor_full.ps1`: adds a read-only `ram_free_gib` field from
  `Win32_OperatingSystem.FreePhysicalMemory` while retaining the exact frozen
  manifest count, GPU status, and C/D disk readings.

**Expected effect**: every 15-minute status row now distinguishes available
physical RAM from disk and VRAM, allowing the operational floor to be applied
without changing the active experiment.

**Document sync**: user_requirements.md yes | implementation.md yes | configs
unchanged | idea_report.md unchanged
