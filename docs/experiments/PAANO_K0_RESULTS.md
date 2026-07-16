# PaAno Execution-Activity K0 Results

## Decision

```text
STOP_NO_PERFORMANCE_HEADROOM
```

The K0 established the execution mechanism but rejected all preregistered performance branches. No method is frozen, no confirmation seeds are authorized, and no rescue module is added.

## Frozen scope and coverage

- Baseline: PaAno, ICLR 2026, commit `d4c67116190efa4592dc6a8a157ced0def68b6af`.
- Primary seed: 2027.
- Data: six fixed TSB-AD files (NAB, IOPS, Exathlon, SMD, SMAP, SWaT).
- Runs: 24/24 trajectory jobs completed; 0 failed.
- Evaluated artifacts: 42/42 committed score artifacts.
- Arms: OFFICIAL, PAPERNEG, PAPERNEG_NONOVERLAP, and the diagnostic RAND_BN control.
- Confirmation seeds 2028/2029 were not run because the primary decision was terminal.

## Mechanism result

The low-activity and early-checkpoint gates both passed:

- Median post-pretext active-hinge fraction was exactly `0.0` in all 6/6 families.
- OFFICIAL selected BEST at iteration 20 in all 6/6 families.

This is strong evidence that the official triplet objective becomes inactive after the pretext phase and that checkpoint selection collapses to the first post-pretext evaluation point. It is not, by itself, evidence of recoverable accuracy headroom.

## Preregistered performance contrasts

| Contrast | Macro delta VUS-PR | Macro delta AUPRC | Positive families | Worst family delta VUS-PR | Passed |
|---|---:|---:|---:|---:|---|
| OFFICIAL LAST vs BEST | +0.009059 | -0.008610 | 3/6 | -0.259217 | No |
| PAPERNEG LAST vs OFFICIAL LAST | -0.054459 | -0.050189 | 2/6 | -0.316071 | No |
| NONOVERLAP LAST vs PAPERNEG LAST | -0.011941 | -0.012545 | 1/6 | -0.049759 | No |

The checkpoint branch missed the frozen `+0.010` macro VUS-PR threshold, had negative macro AUPRC, and produced a large SMD regression. The paper-negative and non-overlap branches both reduced macro performance.

## Aggregate K0 metrics

| Arm | VUS-PR | AUPRC | VUS-ROC |
|---|---:|---:|---:|
| OFFICIAL BEST | 0.553420 | 0.682741 | 0.909695 |
| OFFICIAL LAST | 0.562479 | 0.674131 | 0.928224 |
| PAPERNEG BEST | 0.540146 | 0.668239 | 0.901041 |
| PAPERNEG LAST | 0.508020 | 0.623942 | 0.899427 |
| PAPERNEG_NONOVERLAP BEST | 0.526850 | 0.651016 | 0.884727 |
| PAPERNEG_NONOVERLAP LAST | 0.496079 | 0.611397 | 0.895515 |
| RAND_BN | 0.585346 | 0.703073 | 0.934350 |

RAND_BN is a diagnostic matched-update control, not a proposed method and not a preregistered route to METHOD_GO. Its macro advantage is heterogeneous: it improves SMAP and SWaT but substantially harms SMD.

## External paper-number reference

The PaAno paper reports VUS-PR `0.530` on TSB-AD-U and `0.431` on TSB-AD-M. The selected-file K0 OFFICIAL BEST values are `0.559711` (two U files) and `0.550274` (four M files). These numbers are included only as an external headline reference: the K0 uses six fixed mechanism-probing files rather than the paper's complete benchmark, so the difference is not a matched reproduction and cannot support a claim that the K0 arm beats PaAno.

## Runtime and memory

- Sum of recorded trajectory runtime: `58.60 s`.
- Maximum recorded peak VRAM: `1277.89 MiB`.
- OFFICIAL mean runtime: `2.99 s` per series; mean peak VRAM: `1205.45 MiB`.
- PAPERNEG_NONOVERLAP mean runtime: `3.05 s`; mean peak VRAM: `1206.57 MiB`.

## Integrity checks

- The runner surface does not accept labels.
- Each score array is atomically committed and SHA-256 verified before the evaluator loads labels.
- Labels are used only by the post-run evaluator.
- Two independent CUDA smoke runs produced identical initialization, replay, BEST/LAST checkpoint, memory-bank, and score hashes.
- Implementation test suite: `28 passed`, no skips.
- The recorded vendor dirty flag is caused only by untracked Python bytecode directories; `git diff` against the frozen vendor commit is empty.

## Scope deliberately not run

Per the accelerated user-approved protocol, this project did not reproduce the full PaAno benchmark. It also did not run full suffix/root-cause audits, a large tuning decomposition, complete U/M evaluation, append-only replay, figures, or ablations. Those omissions do not affect the K0 terminal decision because every authorized performance branch failed its frozen gate.

## Final interpretation

PaAno has a real execution-activity mismatch, but the tested literal objective, checkpoint, and overlap repairs do not convert it into stable performance gains. The scientifically supported result is therefore mechanism-only; this route should be closed rather than expanded into a new network.
