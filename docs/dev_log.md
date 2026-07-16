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
