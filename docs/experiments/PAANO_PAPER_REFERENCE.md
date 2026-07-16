# PaAno ICLR 2026 Paper-Reported Reference

This document records paper-only values from the local published PaAno PDF. It does not use or summarize any result from this project's full Eval run.

## Source and integrity

- Local PDF: `C:/Users/qintian/Desktop/msn/msn444_release/docs/papers/PaAno_ICLR2026_arXiv_2602.01359.pdf`
- SHA-256: `25b51f8d48d1809ce0d6955a24fc0bca64eadaaaaf0db091fd507f255692d599`
- Title-page status: *Published as a conference paper at ICLR 2026*
- Relevant locations: Table 1 (dataset protocol), Tables 2-3 (rounded headline results), Table 4 (core ablations), Table 12 (nearest-neighbor sensitivity), and Table 15 (exact full-Eval averages).

## Full-Eval protocol

| Track | Tuning series | Eval series | Paper protocol |
|---|---:|---:|---|
| TSB-AD-U | 48 | 350 | A predefined split supplies the preceding training segment. |
| TSB-AD-M | 20 | 180 | A predefined split supplies the preceding training segment. |

PaAno selects patch size and learning rate using Tuning VUS-PR, then evaluates on the fixed Eval split. The reported defaults are patch size 96, learning rate `1e-4`, 100 AdamW iterations, batch size 512, 10% memory, and top-3 scoring. The paper states that each experiment was repeated with ten random seeds and reports averages. VUS-PR is the primary metric; the evaluation avoids point adjustment and post-hoc threshold tuning.

## Paper-reported main results

The metric order is VUS-PR, VUS-ROC, Range-F1, AUC-PR, AUC-ROC, and Point-F1.

### Tables 2-3: rounded headline values

| Track | VUS-PR | VUS-ROC | Range-F1 | AUC-PR | AUC-ROC | Point-F1 |
|---|---:|---:|---:|---:|---:|---:|
| TSB-AD-U | 0.53 | 0.89 | 0.49 | 0.47 | 0.87 | 0.52 |
| TSB-AD-M | 0.43 | 0.79 | 0.41 | 0.38 | 0.76 | 0.43 |

### Table 15: exact full-Eval averages

| Track | VUS-PR | VUS-ROC | Range-F1 | AUC-PR | AUC-ROC | Point-F1 |
|---|---:|---:|---:|---:|---:|---:|
| TSB-AD-U | 0.5296 +/- 0.0027 | 0.8877 +/- 0.0012 | 0.4869 +/- 0.0030 | 0.4682 +/- 0.0038 | 0.8660 +/- 0.0011 | 0.5164 +/- 0.0032 |
| TSB-AD-M | 0.4263 +/- 0.0051 | 0.7940 +/- 0.0023 | 0.4065 +/- 0.0067 | 0.3772 +/- 0.0044 | 0.7623 +/- 0.0022 | 0.4275 +/- 0.0035 |

The `+/-` terms are transcribed exactly from the PDF's ten-seed table. The paper does not explicitly label them as standard deviation versus standard error in the table caption, so this project refers to them as the paper-reported dispersion rather than assigning an unsupported estimator.

## Table 4 core-ablation names

The names below preserve the paper's component semantics. VUS-PR is shown as a percentage.

| Paper ablation variant | U VUS-PR | M VUS-PR |
|---|---:|---:|
| `w/o InstanceNorm` | 45.3 | 33.4 |
| `w/o L_triplet and L_pretext` | 48.0 | 35.6 |
| `w/o Negative Selection in L_triplet` | 50.9 | 40.2 |
| `Replace L_triplet with InfoNCE loss` | 48.3 | 36.2 |
| `w/o L_pretext` | 51.1 | 42.2 |
| `Continuous Use of L_pretext` | 47.4 | 40.8 |
| `w/o Linear Decay on L_pretext` | 52.9 | 42.4 |
| `PaAno (Ours)` | 53.0 | 42.6 |

These paper ablations are contextual external evidence, not matched controls for this project's `OFFICIAL`, `PAPERNEG`, or `PAPERNEG_NONOVERLAP` arms. In particular, Appendix Table 8 identifies `w/o Negative Selection in L_triplet` with a random-negative triplet variant; it is not the released-code execution arm or the paper-faithful negative-space repair used here.

## Correction of the M reference

The previously used value `0.431` is not PaAno's default full-Eval M headline in this PDF. Its apparent source is Table 12's M sensitivity result of `43.1` at `k=1`. PaAno's stated default is `k=3`, whose Table 12 VUS-PR is `42.6`; Table 15 gives the exact default full-Eval mean as `0.4263`, rounded to `0.43` in Table 3.

Accordingly, manuscript-facing external thresholds are frozen as:

```text
TSB-AD-U VUS-PR: 0.5296 (rounded headline: 0.53)
TSB-AD-M VUS-PR: 0.4263 (rounded headline: 0.43)
```

The already-running full benchmark retains the original frozen
`configs/k0_protocol.yaml` bytes so that every score artifact has one identical
configuration hash.  Its legacy `paper_vus_pr_*` metadata fields are not used by
training, scoring, evaluation, or the full-benchmark decision.  The decision
code uses the corrected constants above; changing the frozen YAML mid-run would
create an avoidable mixed-provenance artifact set.

They must be labeled `paper-reported`. They are not a local baseline reproduction, a paired same-seed comparison, or a substitute for this project's same-file component ablations.
