# Literature Evidence Matrix - PaAno Objective Activity

> Date: 2026-07-16  
> Decision: `GO_K0_MECHANISM_ONLY / NO_METHOD_FREEZE`  
> Scope: the eight locally verified PDFs listed in the download manifest.

## Baseline-Specific Evidence

| Claim | Evidence location | Short verified source text | Consequence |
|---|---|---|---|
| PaAno positives are very small temporal shifts | PaAno, Section 3.3, PDF p.6 | "the positive patch ... is obtained by randomly shifting the anchor ... within r time steps" | With official `w=96, r=2`, the derived input overlap is 97.92-98.96%. |
| PaAno uses the farthest minibatch patch as negative | PaAno, Section 3.3, PDF p.6 | "the farthest negative ... has the largest cosine distance" | The selected negative maximizes `d_neg`, making the hinge easier to satisfy; activity must be measured rather than assumed. |
| Division by ten is official | PaAno, Appendix B.1, PDF p.17 | "The triplet loss was divided by 10 during training." | This is not a paper-code mismatch. It remains a valid one-line causal control. |
| The pretext loss is short-lived | PaAno, Appendix B.1, PDF p.17 | "linear decayed from 1 to 0 during the first 20 iterations" | The post-pretext activity of the triplet objective is a primary K0 endpoint. |
| The full objective has aggregate utility | PaAno, Table 4, PDF p.10 | Full: 53.0/42.6 U/M; without both losses: 48.0/35.6 | The route cannot claim that PaAno learning is globally useless. It must identify performance-relevant temporal or family variation. |
| PaAno already favors farthest negatives | PaAno, Table 8, PDF pp.21-22 | "the farthest negative consistently achieved the strongest performance" | A generic semi-hard or hard-negative replacement is contradicted by the baseline and is not a defensible novelty claim. |
| PaAno headline and efficiency reference | PaAno, Tables 2-3, PDF p.9 | VUS-PR 0.53 U and 0.43 M; approximately 0.3M parameters | These are external paper-reported headline references, not same-code K0 measurements. |

## Paper-Code Cross-Check

| Finding | Paper | Released code | K0 control |
|---|---|---|---|
| Negative candidate pool and selection space | Algorithm 1, PDF p.16: select from minibatch anchors in encoder space, then project | `train.py:117-132`: compare projected anchors against projected positive views | `OFFICIAL` versus `PAPERNEG` |
| Returned checkpoint | Algorithm 1, PDF p.16: return the encoder after `T_iter` | `train.py:33-34, 173-179`: restore weights associated with minimum scheduled minibatch loss | score `BEST` and `LAST` from one trajectory |
| Effective memory ratio | Appendix B.1, PDF p.17: memory size 10% | `utils/utils.py:33-40`: minimum 500 exemplars | record requested and effective ratios; identical within paired comparisons |

> These are execution facts, not proof of performance harm. The K0 tests their downstream consequence.

## Novelty-Collision Evidence

| Work | Evidence location | Short verified source text | Boundary for this project |
|---|---|---|---|
| TimesURL, AAAI 2024 | Abstract/Introduction, PDF p.1 | "unsuitable positive and negative pair construction may introduce inappropriate inductive biases" | Generic better-pair construction and hard negatives are occupied. |
| SoftCLT, ICLR 2024 | Abstract, PDF p.1 | "instance-wise and temporal contrastive loss with soft assignments" | Soft temporal labels, distance weighting, and generic soft contrastive learning are occupied. |
| No More Shortcuts, AAAI 2024 | Abstract, PDF p.1 | "tasks are too simple" and use "local appearance statistics" | Supports the activity/shortcut diagnostic pattern, but in video rather than TSAD. |
| DADA, ICLR 2025 | Section 3.4 and Appendix A.3, PDF pp.6-7, 17-18 | Injected common anomalies train normal/anomaly discrimination | Generic pseudo-anomaly generation is occupied and excluded. |
| PAI, 2026 preprint | Sections 4.3-4.4, PDF p.8 | PaAno plus raw-amplitude evidence rises "from 0.52 to 0.58" | PaAno amplitude preservation and raw-score fusion are occupied. |

## Evaluation Evidence

| Work | Evidence location | Short verified source text | Adopted rule |
|---|---|---|---|
| TSB-AD, NeurIPS 2024 | Section 5.2, PDF p.9 | "VUS-PR emerges as the most robust" evaluation measure | VUS-PR is primary; AUPRC is secondary; use separate tuning/evaluation logic. |
| Quo Vadis, ICML 2024 | Abstract and discussion, PDF pp.1-3 | Complex methods must be evaluated against "simpler baselines" | Frozen-random, pretext-only, and one-line scale controls are mandatory. |
| Quo Vadis, ICML 2024 | Metrics discussion, PDF p.6 | "include the Area Under the Precision Recall Curve" | Do not rely on threshold-optimized F1 for the K0 decision. |

## Surviving Question

> Does PaAno's combination of near-identical positives and maximally distant negatives cause its triplet hinge to lose activity prematurely, and does the resulting activity pattern predict anomaly-ranking loss across real telemetry families?

No reviewed work reports PaAno's active-hinge trajectory, hinge-margin distribution, per-loss encoder-gradient norms, gradient cosine, or a mechanism-to-VUS-PR association. That narrow measurement gap survives. It authorizes a K0 diagnosis only, not a final method.

## Immediate Falsification Conditions

- The triplet remains substantially active after the pretext phase across the selected real families.
- Activity collapse is not associated with learned contribution or paired VUS-PR/AUPRC.
- A matched activity-restoring diagnostic does not improve detection ranking.
- Any apparent improvement is fully explained by removing the official division by ten.
- The only plausible intervention reduces to TimesURL-style mining or SoftCLT-style weighting.
