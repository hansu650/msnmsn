# Pre-G.0 Claim--Evidence Gate

> Phase F snapshot, 2026-07-16. This file contains no incomplete full-Eval
> metrics and authorizes no manuscript performance claim. It defines what the
> completed K0 can support, what must wait for the 530-series full benchmark,
> and what the pilot evidence has already ruled out.

## Evidence Already Available

| Candidate claim | Evidence | Permitted wording and boundary |
|---|---|---|
| The public PaAno paper and code use different negative-candidate execution paths. | Paper--code cross-check in `literature_evidence_matrix.md` and frozen vendor audit. | State the execution difference. Do not infer a performance loss without matched evidence. |
| PaAno's official positive construction creates near-duplicate raw patches. | `w=96`, shift radius `r=2`, and measured 97.92%--98.96% raw overlap. | Describe near-duplicate geometry. Do not call it a harmful shortcut. |
| The official K0 triplet term becomes inactive after the scheduled pretext phase. | Median active-hinge fraction is 0.0 during iterations 20--100 for all six K0 families. | A six-file, seed-2027 observation; not a universal convergence claim. |
| The official BEST checkpoint is selected at iteration 20 in K0. | Six of six K0 families select iteration 20. | Report as a stable pilot observation, not proof of inferior detection. |
| The three preregistered K0 repair branches do not establish performance headroom. | LAST: VUS-PR `+0.009059`, AUPRC `-0.008610`; PAPERNEG: `-0.054459/-0.050189`; NONOVERLAP: `-0.011941/-0.012545`. | Preserve as negative causal evidence. No branch is a validated method. |
| The architecture-and-memory floor is family-dependent. | RAND_BN K0 macro VUS-PR `0.585346`, with heterogeneous family outcomes. | Use only as a pilot diagnostic; do not claim training is globally unnecessary. |
| The implementation enforces evaluator-only labels. | Score commits and hashes precede evaluator label access; tests and CUDA smoke runs passed. | Reproducibility/protocol claim only. |

## Claims Blocked Until Full Benchmark Completion

The following require exact 350-series TSB-AD-U and 180-series TSB-AD-M
coverage for all three registered LAST arms:

1. Full-arm U/M VUS-PR, AUPRC, and VUS-ROC.
2. Track- and family-level effects of removing non-overlap positives or both
   registered execution changes.
3. Comparison with the fixed external PaAno Table 15 means: U `0.5296`, M
   `0.4263`.
4. Complete-runtime and peak-VRAM statements.
5. The Phase F terminal outcome: `CONTINUE_FULL_CONFIRMATION` or
   `STOP_FULL_MAIN_FAILURE`.
6. Any cross-seed mean, variance, or stability wording. Seeds 2028/2029 are
   forbidden unless seed 2027 exceeds both fixed external track means; the
   complete seed-2027 file-weighted track mean is still a valid single-seed
   endpoint.

Even if both tracks exceed the external values, the safe wording is:

> Under the paper-compatible complete-list and file-weighted protocol, the
> registered arm exceeds PaAno's paper-reported mean.

It is not permissible to call that comparison a local baseline reproduction,
a paired improvement, or a statistically significant win over PaAno.

If exactly one track exceeds its fixed external mean, that track result is
descriptive only. A one-track pass cannot support an overall improvement,
method-success, or cross-domain generality claim.

## Component-Attribution Gate

The external two-track performance gate and the same-code component evidence
answer different questions. Passing the external gate does not by itself
validate either registered execution change.

- If the complete arm is descriptively higher than both removal arms on both
  tracks, the manuscript may report cautious seed-2027 component evidence. It
  still cannot call the component effect statistically significant.
- If `PAPERNEG-LAST` or `OFFICIAL-LAST` matches or exceeds the complete arm on
  either track, the result is mixed or dominated. The manuscript must not
  attribute an external-score difference to non-overlap positives or
  paper-negative execution, and it must not present the registered arm as a
  validated method solely because it exceeds both external paper means.
- Mixed/dominated ablations require a Phase F diagnosis or a negative-audit
  framing. Eval labels may explain the completed result but cannot select a
  new arm, component, family, or threshold.

## Claims Ruled Out by K0

- Switching from BEST to LAST is a reliable repair.
- Paper-faithful negative selection consistently improves detection.
- Removing positive overlap creates stable performance headroom.
- Objective inactivity alone implies recoverable anomaly-detection accuracy.
- The K0 establishes a new frozen method.
- The six-file K0 locally reproduces or beats the full PaAno benchmark.
- The complete causal chain `overlap -> inactivity -> ranking loss` has been
  demonstrated.
- Dividing the triplet term by 10 is an implementation bug. The factor is an
  explicit paper setting.

The following generic novelty claims are also unavailable because closely
related mechanisms already exist in the cited literature:

- generic pair construction or soft contrastive learning (TimesURL/SoftCLT),
- amplitude fusion as a new mechanism (PAI), and
- generic pseudo-anomaly training as a new mechanism (DADA).

## Claims Not Supported by the Current Full Design

The completed full benchmark cannot automatically establish these statements,
regardless of its headline score:

- Learning consistently improves on architecture plus memory: the full design
  contains no RAND_BN arm.
- The proposed arm preserves local-shift invariance: no local-shift
  consistency endpoint is registered.
- Objective activity is restored on all 530 files: no full-scale activity
  summary is registered.
- `OFFICIAL-LAST` is a local PaAno reproduction: it is only a component-removal
  ablation under this project's runner.
- Every family improves, or a family-specific selector is justified.

## Numerical Source Priority

1. Full compact outputs generated by `08_finalize_full.ps1`, once complete.
2. `docs/experiments/PAANO_PAPER_REFERENCE.md` for the corrected external
   Table 15 values.
3. K0 compact artifacts for pilot-only mechanism and negative results.

The legacy K0 CSV value M `0.431` must never enter a manuscript table or claim.
The corrected external value is M `0.4263`.

## Transition Rule

After full finalization:

- If the result is `CONTINUE_FULL_CONFIRMATION`, retain only supported
  seed-2027 claims, run the frozen seeds 2028/2029, and enter G.0 after the
  fixed three-seed summary is complete.
- If the result is `STOP_FULL_MAIN_FAILURE`, keep the complete negative result
  and return to Phase F diagnosis. A new iteration is allowed only when an
  independent, evidence-supported, non-scope-expanding hypothesis remains; it
  must be preregistered before code or experiment changes and must not tune
  against Eval labels. Otherwise, stop this route rather than manufacture a
  rescue variant.

After the fixed seeds 2028/2029 complete, apply a claim-only gate; it must not
alter the method or choose results:

1. **Stable external exceedance**: all three registered seeds exceed both
   fixed external track means and the fixed three-seed mean exceeds both. The
   manuscript may state that the registered arm exceeded the paper-reported
   means across the three fixed seeds, subject to the component-attribution
   gate. It still cannot claim a paired or statistically significant win over
   PaAno.
2. **Mean pass with seed instability**: the fixed mean exceeds both references
   but at least one registered seed does not. Report every seed, the fixed
   mean/dispersion, and the instability. Wording is limited to a descriptive
   fixed-three-seed mean; do not claim stable improvement.
3. **Mean failure**: either fixed three-seed track mean does not exceed its
   external reference. Return to Phase F/stop and do not advance a positive
   method claim.
