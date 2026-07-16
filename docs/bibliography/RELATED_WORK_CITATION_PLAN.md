# Related Work Citation Plan

> Status: frozen prewriting inventory. The manuscript and venue template are
> not initialized until the official IEEE MSN author kit is provided.

## Frozen recency policy

The narrative Related Work is dominated by verified 2025--2026 publications.
The current narrative set is frozen at a 4:1 recent-to-2024 ratio:

| Publication year | Narrative keys | Count | Share |
| --- | --- | ---: | ---: |
| 2026 | `park2026paano`, `zhang2026pai`, `xing2026doknowad` | 3 | 30.0% |
| 2025 | `sun2025igad`, `li2025crossad`, `shentu2025towards`, `pmlr-v267-zhou25u`, `wu2025catch` | 5 | 50.0% |
| 2024 | `liu2024timesurl`, `lee2024softclt` | 2 | 20.0% |
| 2023 or earlier | none | 0 | 0.0% |
| **Total** | **10 narrative keys** | **10** | **100.0%** |

Thus, 2025--2026 work contributes 8/10 narrative citations (80.0%). Seven of
those eight papers are peer reviewed; `zhang2026pai` is retained as the closest
2026 PaAno extension and must always be identified as an arXiv preprint.

The two 2024 papers are exceptions for exact, irreplaceable pair-semantics
claims: pair construction and graded temporal similarity. The 2024 TSB-AD and
evaluation-reliability papers remain protocol/evidence provenance outside the
narrative. A later 2024 narrative addition requires a written indispensability
reason and must not reduce the 4:1 recent-to-2024
ratio. Papers from 2023 or earlier cannot frame narrative novelty. They may be
cited only for an original metric, inherited component, evaluation rule, or
comparison-method identity.

## Topic 1: Recent Representation and Patch-Based TSAD

The paragraph distinguishes recent representation and architecture changes
from the present question of whether a released training objective is executed
as specified and remains useful.

| Key | Year | One-sentence contribution | Exact reason for citation |
| --- | ---: | --- | --- |
| `park2026paano` | 2026 | PaAno learns compact patch representations with triplet and temporal-pretext objectives and scores them against a compressed normal memory. | It is the accepted 2026 baseline and the exact execution path studied. |
| `zhang2026pai` | 2026 | PAI restores amplitude information that representation-distance anomaly scores may discard. | It is the closest PaAno score-level extension and closes amplitude preservation as the present novelty; it is a preprint, not a peer-reviewed comparator. |
| `xing2026doknowad` | 2026 | DoKnowAD uses auxiliary knowledge to calibrate normal representations for anomaly detection. | It represents current knowledge-assisted normal-representation modeling and separates that route from execution-fidelity analysis. |
| `sun2025igad` | 2025 | IGAD uses idempotent reconstruction to address overgeneralization in time-series anomaly detection. | It is a recent mechanism-oriented detector and shows that reconstruction overgeneralization is an occupied failure mode. |
| `li2025crossad` | 2025 | CrossAD models cross-scale and cross-window dependencies for anomaly detection. | It represents recent architectural modeling of temporal context rather than training-objective execution. |
| `shentu2025towards` | 2025 | DADA learns a general detector through adaptive bottlenecks and dual adversarial decoders across time-series domains. | It establishes general multi-domain and synthetic-discrimination modeling as occupied territory. |
| `pmlr-v267-zhou25u` | 2025 | KAN-AD applies Kolmogorov--Arnold networks to nonlinear TSAD representation learning. | It represents a recent learned-detector architecture in PaAno's comparison landscape. |
| `wu2025catch` | 2025 | CATCH performs channel-aware frequency patching for multivariate anomaly detection. | It shows that patch, frequency, and channel modeling are established model-level directions distinct from objective fidelity. |

**Transition.** These works change representations, architectures, external
knowledge, or scores. They do not isolate whether PaAno's released negative
geometry, positive overlap, and checkpoint semantics suppress the intended
training objective.

## Topic 2: Closest Objective-Semantics Precedents

This compact paragraph uses only two indispensable 2024 papers. They bound the
claim without allowing older representation-learning literature to dominate
the narrative.

| Key | Year | One-sentence contribution | Exact reason for citation |
| --- | ---: | --- | --- |
| `liu2024timesurl` | 2024 | TimesURL develops time-series-aware positive and negative construction with informative negatives. | It is the closest direct precedent for the claim that unsuitable pair semantics impose harmful inductive bias. |
| `lee2024softclt` | 2024 | SoftCLT replaces uniform hard pair labels with graded temporal and instance similarity. | It establishes correlation-aware pair weighting as prior art and prevents an overbroad novelty claim. |

**Transition.** These works establish that pair semantics matter in general,
but they do not analyze PaAno's paper--code mismatch, near-duplicate patch
views, sustained triplet activity, or anomaly-ranking utility.

`dave2024nomore` is excluded from the frozen Related Work narrative to keep the
2024 set sparse. It may be used once as cross-domain mechanism provenance in a
method-motivation sentence only if the shortcut claim cannot be supported by
the time-series-specific sources above; it must not be described as TSAD prior
art.

## Experimental-Protocol Provenance Outside Related Work

| Key | Year | One-sentence contribution | Exact reason for citation |
| --- | ---: | --- | --- |
| `liu2024elephant` | 2024 | TSB-AD curates heterogeneous U/M benchmarks and studies reliable threshold-independent TSAD evaluation. | It defines the shared data protocol and the provenance of the paper-compatible benchmark comparison. |
| `pmlr-v235-sarfraz24a` | 2024 | Quo Vadis argues that TSAD progress should be checked against strong simple baselines under rigorous evaluation. | It justifies matched controls, negative-result preservation, and avoiding unsupported architectural complexity. |

These papers support the Experiments protocol and evidence policy; they are not
part of the frozen Related Work narrative and do not count toward its recency
distribution.

Metric and evaluation provenance is deliberately outside the narrative set:
`paparrizos2022vus` is cited where VUS-PR/VUS-ROC are defined, and
`kim2022rigorous` is cited only where point-adjustment inflation or strict
evaluation is specified. Neither paper is used to frame the current novelty.

## Research Gap

Recent work improves TSAD representations and architectures, while the closest
objective work studies temporal pair semantics. The unresolved baseline-specific
link is between implementation
fidelity, sustained objective activity, and anomaly-ranking utility in PaAno.
The final claim is conditioned on the full Phase F evidence gate; a failed gate
is reported as a negative result rather than rewritten as an improvement.

## Comparator and Provenance Citations Outside the Narrative

Every distinct model behind PaAno Tables 2 and 3 is mapped in
`PAANO_TABLE23_CITATION_MAP.csv`. Those citations identify methods in the
Experiments setup or table note; they are not forced into Related Work and do
not count toward its recency distribution. Older sources are allowed there
only because they establish the original identity of a comparison method.

Likewise, `kim2022revin` is restricted to the Method description of the
inherited RevIN component, and `park2026paano_code` is restricted to immutable
implementation provenance. All copied comparator values remain labeled
**PaAno-paper-reported** and cite `park2026paano`; original model papers do not
convert external values into local reproductions.
