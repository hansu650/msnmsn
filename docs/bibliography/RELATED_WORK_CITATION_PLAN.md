# Related Work Citation Plan

> Status: prewriting inventory only. The manuscript and venue template are not
> initialized until the official IEEE MSN author kit is provided.

## Recency rule

The narrative is led by 2025--2026 peer-reviewed work. A small number of 2024
papers are retained when they define the closest objective-learning or benchmark
boundary. Earlier sources appear only as indispensable provenance for a metric,
an inherited module, an evaluation protocol, or an original comparison method.

## Topic 1: Recent Representation and Patch-Based TSAD

The paragraph will distinguish recent representation/architecture changes from
the present question of whether a released training objective is executed as
specified and remains useful.

| Key | Role in the argument |
| --- | --- |
| park2026paano | Primary 2026 patch-representation baseline and exact execution path studied. |
| zhang2026pai | Closest amplitude-preservation extension; explicitly identified as a 2026 arXiv preprint. |
| xing2026doknowad | AAAI 2026 normal-representation calibration using auxiliary knowledge. |
| sun2025igad | NeurIPS 2025 mechanism-oriented idempotent reconstruction for overgeneralization. |
| li2025crossad | NeurIPS 2025 cross-scale/cross-window architectural modeling. |
| shentu2025towards | ICLR 2025 general TSAD through adaptive bottlenecks and dual adversarial decoders. |
| pmlr-v267-zhou25u | ICML 2025 nonlinear KAN-based TSAD architecture. |
| wu2025catch | ICLR 2025 channel-aware frequency patching for multivariate TSAD. |

**Transition.** These works change representations, architectures, external
knowledge, or scores. They do not isolate whether PaAno's released negative
geometry, positive overlap, and checkpoint semantics suppress the intended
training objective.

## Topic 2: Temporal Pair Semantics and Shortcut-Aware Objectives

This paragraph uses only the closest 2024 mechanism papers because they directly
bound the proposed claim.

| Key | Role in the argument |
| --- | --- |
| liu2024timesurl | Time-series-aware contrastive pair construction and informative negatives. |
| lee2024softclt | Graded temporal similarity instead of uniform hard pair labels. |
| dave2024nomore | Cross-domain mechanism precedent showing temporal self-supervision can exploit shortcuts. |

**Transition.** These works establish that pair semantics and shortcuts matter
in general, but they do not analyze PaAno's paper--code mismatch, near-duplicate
patch views, sustained triplet activity, or anomaly-ranking utility.

## Topic 3: Reliable TSAD Evaluation

| Key | Role in the argument |
| --- | --- |
| liu2024elephant | TSB-AD benchmark, U/M protocol, and reliable threshold-independent evaluation. |
| pmlr-v235-sarfraz24a | ICML 2024 argument for strong simple controls and rigorous TSAD evaluation. |
| paparrizos2022vus | Indispensable provenance for VUS-PR/VUS-ROC. |
| kim2022rigorous | Indispensable provenance for avoiding point-adjustment inflation. |

**Transition.** Benchmark reliability does not by itself identify whether a
training-objective implementation difference causes a detector's observed
ranking behavior.

## Research Gap

Recent work improves TSAD representations and architectures, closer objective
work studies temporal pair semantics, and benchmark work strengthens evaluation.
The unresolved baseline-specific link is between implementation fidelity,
sustained objective activity, and anomaly-ranking utility in PaAno. The final
claim will be conditioned on the full Phase F evidence gate; a failed gate is
reported as a negative result rather than rewritten as an improvement.

## Comparator Citations Outside the Narrative

Every distinct model behind PaAno Tables 2 and 3 is mapped in
PAANO_TABLE23_CITATION_MAP.csv. Those citations identify methods in the
Experiments setup or table note; they are not all forced into Related Work.
All copied values remain labeled **PaAno-paper-reported** and cite
park2026paano. Original model papers do not convert external values into local
reproductions.
