# Execution Fidelity and Objective Activity in Patch-Based Time-Series Anomaly Detection - Idea Report
> Generated: 2026-07-16 | Status: PHASE_C_COMPLETE - GO_K0_MECHANISM_ONLY - METHOD_NOT_FROZEN
> Extended: Baseline-specific code evidence is included because it defines the retained mechanism hypothesis.

---

## Part 1 Topic Overview

### 1 Motivation

Lightweight time-series anomaly detection is attractive for industrial sensing, IoT, and server telemetry, where a detector must process long streams without the training and inference cost of a large Transformer. PaAno, accepted at ICLR 2026, is a particularly strong baseline: it learns a compact patch encoder using triplet and temporal-pretext losses, compresses normal embeddings into a memory bank, and reports VUS-PR of 0.53 on TSB-AD-U and 0.43 on TSB-AD-M while using approximately 0.3 million parameters [1]. Its use of TSB-AD and VUS-PR also aligns with recent calls for threshold-independent, benchmark-consistent evaluation [3,4].

The retained question concerns the *execution fidelity, activity, and semantics of PaAno's training objective*, not its CNN architecture or final score calibration. The official scripts use length-96 patches extracted with unit stride, so positives at offsets -2, -1, +1, or +2 share either 95/96 or 94/96 raw samples. More importantly, the released negative geometry differs from the paper's algorithm: the paper selects a farthest patch from minibatch anchors in encoder space and then projects it, whereas the code compares projected anchors against the batch's positive views in projection space [1,9]. The paper returns the encoder after all iterations, while the code restores post-update weights associated with the lowest minibatch loss even though the pretext coefficient changes during training [1,9]. These facts create two exact parity controls and one still-unproven shortcut hypothesis.

> This is a falsifiable code-level hypothesis, not a declared flaw. PaAno's own ablations report that farthest negatives outperform random, closest, median, and InfoNCE alternatives [1]. Therefore, mathematical ease or a loss mismatch alone cannot support a paper; the mechanism must predict real cross-family detection loss.

Prior work sharply limits the novelty claim. TimesURL already argues that unsuitable positive and negative construction introduces inappropriate time-series inductive biases [6]. SoftCLT already replaces hard temporal assignments with correlation-aware soft assignments [7]. No More Shortcuts shows more generally that easy temporal pretexts can saturate through low-level local statistics rather than high-level temporal structure [8]. Accordingly, this project cannot claim that positive/negative sampling is generally overlooked. Its potentially defensible contribution is narrower: a PaAno-specific causal diagnosis linking extreme patch overlap and triplet inactivity to anomaly-detection headroom under the exact 2026 baseline protocol.

PAI independently identifies amplitude-agnostic scoring in representation-based detectors and raises PaAno's reported TSB-AD-U VUS-PR from approximately 0.52 to 0.58 using amplitude-aware score augmentation [2]. This directly closes amplitude preservation as the primary new route. DADA likewise demonstrates that generic synthetic-anomaly discrimination and multi-domain representation learning are established directions [5]. Any later intervention must therefore target objective activity rather than repackage amplitude scoring, generic pseudo-anomaly generation, or a larger backbone.

> Plain-language view: PaAno may be answering an easier training question - "are these two windows almost copies?" - instead of the deployment question - "do these windows express the same normal behavior?" The K0 must show that this distinction changes anomaly ranking, not merely that two formulas differ.

**Why this research is necessary:**

- **Application necessity**: PaAno is designed for fast, resource-constrained anomaly monitoring, but a cheap encoder is useful only if its training signal contributes anomaly-relevant information across real telemetry families [1]. TSB-AD shows that apparently sophisticated detectors often fail under reliable evaluation, so objective contribution must be measured rather than assumed [3].
- **Theoretical necessity**: Existing contrastive literature establishes that temporal pair semantics matter [6,7], but it does not analyze the exact combination of unit-stride near-duplicate positives, farthest-negative triplets, released loss scaling, and short-lived pretext learning used by PaAno [1,9].
- **Timing necessity**: PaAno is a newly accepted 2026 baseline with runnable code, while PAI has already exposed and addressed its major score-level amplitude limitation [1,2]. This creates a timely, bounded opportunity to determine whether a distinct training-objective limitation remains before the baseline accumulates further extensions.

### 2 Research Questions

The questions separate the scientific goal, the causal mechanism, and the boundary of any conclusion. RQ1 asks whether learned training adds value beyond PaAno's architecture and memory scorer; RQ2 tests the proposed objective shortcut; RQ3 prevents a repair from being credited if it merely destroys useful shift invariance or works on one anomaly family.

#### RQ1: Core Question

**How much does PaAno's learned objective contribute to anomaly-discriminative normal representations beyond its patch architecture and memory scorer across heterogeneous telemetry families?**

- **Corresponding gap**: PaAno reports component ablations, but does not report triplet active-rate trajectories, per-objective gradient contribution, or a pretext-only/no-triplet control under the final implementation [1,9].
- **Novelty**: The question is baseline-specific and concerns mechanism-to-performance causality, not a generic claim that contrastive learning requires good pairs [6,7].
- **Corresponding experiment**: Six-family K0 comparing the released path, paper-faithful negative selection, best versus last checkpoints, a BN-calibrated random encoder, and a non-overlapping-positive diagnostic with identical memory and scoring.

#### RQ2: Mechanism Question

**Do PaAno's negative-selection execution mismatch, non-comparable checkpoint rule, and extreme raw overlap suppress useful triplet activity or bias the learned representation?**

- **Relationship to RQ1**: If the learned encoder adds little, RQ2 identifies the specific training dynamics that can explain why and constrain a matched intervention.
- **Corresponding gap**: Prior work establishes pair-construction and shortcut risks in general [6-8], but does not test PaAno's exact loss activity or connect it to TSAD ranking.
- **Corresponding experiment**: Factor paper-faithful negative selection, final-versus-best checkpoint scoring, and non-overlap positives while logging active-hinge fraction, distance margins, gradient norms, and overlap dependence without exposing labels to training.

#### RQ3: Boundary Question

**Under which anomaly families and temporal regimes does restoring meaningful objective activity improve detection without losing useful local-shift invariance or PaAno's efficiency?**

- **Relationship to RQ1/RQ2**: A harder objective is not automatically better; the intended small-shift invariance may be beneficial. RQ3 distinguishes robust semantic learning from indiscriminate sensitivity.
- **Corresponding experiment**: Report per-family VUS-PR/AUPRC, worst-family delta, runtime, and local-shift consistency across NAB-U, IOPS-U, Exathlon-M, SMD-M, SMAP-M, and SWaT-M.

### 3 Key Works

The core reading set covers the 2026 baseline, its direct 2026 extension, reliable TSAD evaluation, representation-learning controls, and general temporal-shortcut evidence. The summary table lists the six works most directly tied to the retained hypothesis; every downloaded paper is recorded below.

| Short Name | Venue | Year | Core Contribution | Borrowing Value |
|---|---:|---:|---|---|
| PaAno | ICLR | 2026 | Lightweight patch metric learning for TSAD | Primary baseline and mechanism source |
| PAI | arXiv | 2026 | Restores amplitude evidence in representation scores | Closes amplitude route; recent comparator |
| TSB-AD | NeurIPS D&B | 2024 | Curated 1,070-series benchmark and reliable metrics | Dataset protocol and VUS-PR |
| TimesURL | AAAI | 2024 | Time-series-aware positive and negative construction | Pair-semantics precedent |
| SoftCLT | ICLR | 2024 | Soft temporal and instance contrastive assignments | Non-binary similarity principle |
| No More Shortcuts | AAAI | 2024 | Diagnoses easy temporal-pretext shortcuts | Mechanism diagnostics |

#### PaAno (ICLR 2026) [1]

PaAno trains a compact 1D-CNN with triplet and temporal-pretext objectives and scores test patches against a compressed normal memory bank.

> Borrowing value: primary baseline, official TSB-AD protocol, loss ablations, and efficiency target. [1]
> Key work: yes, because every causal and performance claim is anchored to this implementation.

#### PAI (arXiv 2026) [2]

PAI augments representation-bank scores with Euclidean and raw-amplitude evidence and reports PaAno+PAI VUS-PR near 0.58 on TSB-AD-U.

> Borrowing value: mandatory recent comparator and evidence that amplitude scoring is no longer an open PaAno novelty claim. [2]
> Key work: yes, because it defines the strongest direct extension and an exclusion boundary.

#### TSB-AD (NeurIPS 2024 Datasets and Benchmarks) [3]

TSB-AD curates 1,070 time series from 40 datasets and identifies VUS-PR as a reliable primary TSAD measure.

> Borrowing value: official U/M splits, telemetry-family coverage, and threshold-independent evaluation. [3]
> Key work: yes, because it supplies the shared experimental protocol.

#### Quo Vadis (ICML 2024) [4]

Quo Vadis argues that TSAD progress must be tested against simple baselines under rigorous, non-inflating evaluation.

> Borrowing value: supports frozen-random/no-triplet controls and the rule that complexity requires measurable utility. [4]
> Key work: no, because it governs evaluation rather than the candidate mechanism.

#### DADA (ICLR 2025) [5]

DADA learns a general detector with adaptive bottlenecks and dual adversarial decoders over multi-domain time series.

> Borrowing value: establishes generic anomaly-discrimination and synthetic-abnormal training as occupied territory. [5]
> Key work: no, because its architecture and zero-shot protocol differ from PaAno.

#### TimesURL (AAAI 2024) [6]

TimesURL shows that unsuitable time-series positives and negatives impose harmful inductive biases and develops temporal-frequency augmentation with informative negatives.

> Borrowing value: direct literature basis for testing pair semantics while preventing an overbroad novelty claim. [6]
> Key work: yes, because it is the closest positive/negative-construction precedent.

#### SoftCLT (ICLR 2024) [7]

SoftCLT represents temporal and instance similarity with soft assignments rather than binary contrastive labels.

> Borrowing value: potential principle for a later minimal intervention if hard pair semantics are proven causal; not yet a frozen module. [7]
> Key work: yes, because it sets the novelty boundary for correlation-aware pair weighting.

#### No More Shortcuts (AAAI 2024) [8]

No More Shortcuts demonstrates that temporal pretexts can saturate through low-level local statistics and validates the need to measure task activity and learned feature quality separately.

> Borrowing value: diagnostic logic for objective saturation, shortcut controls, and downstream validation. [8]
> Key work: yes, because it supplies the general mechanism pattern tested in PaAno.

---

## References

1. Park, Jinju, and Seokho Kang. "PaAno: Patch-Based Representation Learning for Time-Series Anomaly Detection." *International Conference on Learning Representations*, 2026. https://arxiv.org/abs/2602.01359.
2. Zhang, Kang, et al. "PAI: Preserving Amplitude Information in Representation-Based Time-Series Anomaly Detection." *arXiv preprint arXiv:2606.08935*, 2026. https://arxiv.org/abs/2606.08935.
3. Liu, Qinghua, and John Paparrizos. "The Elephant in the Room: Towards a Reliable Time-Series Anomaly Detection Benchmark." *Advances in Neural Information Processing Systems, Datasets and Benchmarks Track*, 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html.
4. Sarfraz, M. Saquib, et al. "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?" *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://proceedings.mlr.press/v235/sarfraz24a.html.
5. Shentu, Qichao, et al. "Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders." *International Conference on Learning Representations*, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/ca7998666c2e53cc1e882b7268414d8a-Abstract-Conference.html.
6. Liu, Jiexi, and Songcan Chen. "TimesURL: Self-Supervised Contrastive Learning for Universal Time Series Representation Learning." *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 12, 2024, pp. 13918-13926. https://doi.org/10.1609/aaai.v38i12.29299.
7. Lee, Seunghan, Taeyoung Park, and Kibok Lee. "Soft Contrastive Learning for Time Series." *International Conference on Learning Representations*, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/ccc48eade8845cbc0b44384e8c49889a-Abstract-Conference.html.
8. Dave, Ishan Rajendrakumar, Simon Jenni, and Mubarak Shah. "No More Shortcuts: Realizing the Potential of Temporal Self-Supervision." *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 2, 2024, pp. 1481-1491. https://doi.org/10.1609/aaai.v38i2.27913.
9. Park, Jinju, and Seokho Kang. *PaAno Official PyTorch Implementation*. Commit `d4c67116190efa4592dc6a8a157ced0def68b6af`, 2026. https://github.com/jinnnju/PaAno.

## Pending Verification

- [ ] Whether the triplet hinge is actually inactive after the pretext phase across real families; this is the primary K0 measurement, not a literature claim.
- [ ] Whether objective inactivity causes measurable anomaly-ranking loss rather than serving as benign regularization.
- [ ] Whether any later intervention remains distinct from TimesURL and SoftCLT after implementation details are frozen.

---

## Part 2 Technical Framework

> **Confirmed-content card**
>
> - Research direction: PaAno execution fidelity, objective activity, and temporal-overlap diagnosis for telemetry anomaly detection.
> - Primary RQ: quantify the learned objective's contribution beyond the patch architecture and memory scorer.
> - Secondary RQs: test negative/checkpoint execution, overlap-driven activity, and the regimes in which meaningful activity helps.
> - Direction constraints: ICLR 2026 PaAno baseline; one RTX 4090; CCF-A-level evidence; no score calibration, large backbone, learned selector, or family-specific rescue.
> - Technical framework: instrument the frozen PaAno path, factor paper-code negative selection and checkpoint semantics, compare matched objective controls with an identical memory scorer, and permit method design only after a real-data causal gate passes.
> - Pipeline: cache deterministic real-family inputs, run three matched training trajectories plus a BN-calibrated random control, score both best and last checkpoints, write scores before labels are exposed, then issue a stop/simple-fix/method-design decision.

### 1 Introduction

Compact representation learners are an attractive basis for anomaly monitoring in networks, servers, IoT devices, and industrial sensing systems. PaAno demonstrates that a roughly 0.3-million-parameter patch encoder and a compressed normal memory can be competitive on the broad TSB-AD benchmark [1,3]. Its exact paper-reported full-Eval VUS-PR of 0.5296 on TSB-AD-U and 0.4263 on TSB-AD-M (rounded headlines 0.53/0.43) therefore provides a timely 2026 baseline for studying how much self-supervised training actually contributes to lightweight telemetry detection.

The open issue is not whether PaAno's design works on average; its published ablations establish useful performance and specifically favor farthest-negative sampling [1]. The issue is whether the released code executes the paper's intended negative geometry and checkpoint semantics, and whether the resulting objective remains informative throughout optimization. The paper chooses a negative from minibatch anchors in encoder space, but the code uses other anchors' positive views in projection space. The paper returns the final encoder, but the code selects one post-update minibatch checkpoint using a loss whose pretext weight changes and then vanishes. Positives also share 97.92-98.96% of their input samples [1,9]. Existing work already shows that time-series pair construction and soft temporal similarity matter [6,7], while temporal self-supervision can exploit local shortcuts in another domain [8]. None establishes which exact PaAno execution choice affects anomaly ranking.

We therefore define an execution-fidelity and objective-activity audit rather than prematurely proposing a new network. The audit records hinge activation, margin distributions, per-objective gradients, selected-negative identity, checkpoint iteration, and overlap dependence while preserving PaAno's architecture, memory construction, and scorer. Three trajectories isolate released behavior, paper-faithful negative selection, and paper-faithful selection with non-overlapping positives; a BN-calibrated random encoder measures the architecture-memory floor. Both the code-selected and final checkpoints are scored from each trained trajectory at no extra training cost. PaAno's published ablation already shows that the full objective beats removing both losses by 5.0 U and 7.0 M VUS-PR points, so the K0 does not assume globally useless learning [1].

The intended contribution at this stage is threefold. First, the study operationalizes objective activity for patch-based TSAD rather than inferring representation quality from a scalar training loss. Second, it separates architecture, memory scoring, pretext learning, and triplet activity through matched controls. Third, it uses a pre-registered stop rule: no final method is created if the measured dynamics do not predict real VUS-PR or AUPRC headroom. This design follows the benchmark community's recommendation to privilege reliable evaluation and strong simple controls over architectural accumulation [3,4].

### 2 Related Works

#### 2.1 Lightweight patch-based anomaly detection

PaAno learns local patch embeddings with a triplet objective and a temporal-order pretext task, compresses normal embeddings with MiniBatchKMeans, and scores test patches by their nearest normal-memory distances [1]. Its combination of a small CNN, RevIN, and patch-level retrieval makes it suitable for single-GPU and edge-oriented telemetry experiments. PAI later shows that representation distance can discard amplitude evidence and augments PaAno-like scores with raw-space information [2]. Because PAI directly occupies the amplitude-preservation route, the present work keeps PaAno's detector and score path fixed and studies training-objective activity instead.

#### 2.2 Pair semantics in time-series representation learning

TimesURL argues that unsuitable positives and negatives impose inappropriate inductive bias and combines temporal-frequency augmentation with informative negatives [6]. SoftCLT replaces binary temporal and instance assignments with similarity-aware soft weights and evaluates the learned representations on anomaly detection among other downstream tasks [7]. These papers rule out broad novelty claims around better sampling, hard negatives, or soft temporal labels. They instead motivate a narrower question: whether the exact PaAno pair geometry measurably starves its released hinge objective, and whether that starvation affects its own memory-based anomaly rankings.

#### 2.3 Shortcut diagnosis and reliable TSAD evaluation

No More Shortcuts demonstrates that temporal pretext tasks can be solved through low-level local statistics and separates pretext success from downstream representation utility [8]. TSB-AD and Quo Vadis independently emphasize that TSAD claims should survive reliable metrics, diverse datasets, and simple controls [3,4]. We adopt this diagnostic logic but test a different domain and endpoint: the activity of a patch-triplet objective and its causal effect on VUS-PR and AUPRC across real telemetry families.

#### 2.4 Research gap

The literature establishes that temporal pair semantics matter, that easy self-supervision can exploit local cues, and that TSAD evaluation must be rigorous [3,4,6-8]. PaAno also establishes that its full design and farthest-negative strategy are empirically competitive [1]. What remains unmeasured is the bridge between these observations: whether PaAno's official hinge remains active after the early pretext phase, whether its embedding similarity is dominated by raw overlap, and whether either dynamic explains detection performance beyond the architecture and normal-memory scorer. This baseline-specific mechanism-to-performance chain is the only retained gap.

### 3 Method: Execution-Fidelity and Objective-Activity Audit

> This section specifies a diagnostic research method, not a frozen paper method. A learned or hand-designed replacement objective is prohibited until the real-data gate in Section 3.6 passes.

#### 3.1 Baseline execution graph

For a normal training sequence, PaAno forms every length-96 patch with stride one. An anchor at patch index \(i\) receives a positive from \(i + \delta\), where \(\delta\in\{-2,-1,1,2\}\). The raw overlap ratio is

\[
o(i,i+\delta)=\frac{96-|\delta|}{96}.
\]

> Under the official offsets, \(o\) is either \(95/96\) or \(94/96\). This is a code-derived property, not evidence that the representation is defective [9].

Let \(h_i\) and \(z_i\) denote encoder and projected embeddings. The paper selects a negative index from the anchor pool by distance in \(h\)-space and then forms its projected representation. The released code instead constructs a \(z_{\mathrm{anchor}}z_{\mathrm{positive}}^\top\) matrix and uses off-diagonal positive views as the negative pool. With positive distance \(d_i^+\), selected negative distance \(d_i^-\), and margin \(m=0.1\), the released triplet term is

\[
\mathcal L_{\mathrm{tri}}=\frac{1}{10M}\sum_{i=1}^{M}
\max(0,d_i^+-d_i^-+m).
\]

> The appendix reports the division by ten used in the code, so the audit treats it as an official design choice rather than a reproduction error [1,9].

The temporal-pretext term compares each anchor with a patch one patch-length earlier and with randomly paired batch patches. Its weight reaches zero at iteration 20 of 100. The paper's pseudocode returns the final encoder, whereas the released implementation restores the weights associated with the minimum scheduled minibatch loss. At inference, the projection and classification heads are discarded; the encoder embedding is compared with a requested 10% MiniBatchKMeans normal memory using mean top-3 cosine distance [1,9]. The code enforces a 500-exemplar floor, so the effective memory ratio is logged.

#### 3.2 Plain-language pipeline

**Step 1 - Freeze inputs and the baseline path.** Deterministically select the first eligible Eval file from NAB-U, IOPS-U, Exathlon-M, SMD-M, SMAP-M, and SWaT-M. Preserve PaAno's patch length, stride, seed, optimizer, iteration count, RevIN, memory compression, top-3 scorer, and point-score distribution.

**Step 2 - Instrument official training.** During every training iteration, record which triplets still violate the margin, the positive and selected-negative distances, each loss value, and the gradient contributed by each objective. Separately probe the checkpoint with patch pairs spanning several temporal offsets to test whether embedding similarity mainly follows raw overlap.

**Step 3 - Run matched causal controls.** Train the released trajectory, a trajectory implementing the paper's anchor-pool negative selection, and the same paper-faithful negative trajectory with non-overlapping positives. Evaluate a random encoder after replaying the same batches only to calibrate BatchNorm. All variants share one initial state, architecture, memory builder, and anomaly scorer.

**Step 4 - Write scores before evaluation.** Each runner receives values and split metadata but never anomaly labels. It writes point scores, activity traces, runtime, and hashes. A separate evaluator then loads the labels and computes VUS-PR and AUPRC.

**Step 5 - Link dynamics to utility.** Compare objective activity with the paired detection delta on each family. Geometry alone cannot pass the gate: an inactive hinge matters only if a matched control restores activity and improves ranking, or if official training is indistinguishable from simpler encoders.

**Step 6 - Issue one bounded decision.** Stop if no performance-relevant execution or activity failure is observed. Classify a gain from final-checkpoint scoring or paper-faithful negatives as a simple implementation fix. Enter method design only if the non-overlap diagnostic consistently restores activity and ranking while neither simple parity fix explains the gain.

> The pipeline keeps anomaly labels outside all training and scoring code. Labels are evaluator-only, after immutable score artifacts exist.

#### 3.3 Objective-activity measurements

The active-hinge fraction at iteration \(k\) is

\[
A_k=\frac{1}{M}\sum_{i=1}^{M}
\mathbf 1[d_{i,k}^{+}-d_{i,k}^{-}+m>0].
\]

> \(A_k\) distinguishes a numerically present triplet term from one that actually supplies gradient. We report its complete trajectory, its mean after the pretext term reaches zero, and the iteration at which it first remains below the pre-registered activity threshold.

For each objective \(q\in\{\mathrm{tri},\mathrm{pre}\}\), let

\[
G_{q,k}=\left\|\nabla_{\theta_e}\mathcal L_{q,k}\right\|_2,
\]

where \(\theta_e\) contains only shared encoder parameters.

> Absolute gradient norms prevent a tiny triplet from appearing important merely because it is the only surviving objective. We also record gradient cosine when both terms are active to distinguish aligned supervision from conflict.

For an offline diagnostic pair bank with several temporal offsets, we compute Spearman correlation between raw overlap and embedding cosine similarity. High correlation is interpreted only as shortcut dependence when it co-occurs with weak objective activity and detection headroom; correlation alone is not a failure criterion.

#### 3.4 Matched controls

| ID | Encoder training | Purpose | Status |
|---|---|---|---|
| `OFFICIAL` | Released positive-view negative pool and released losses | Same-code baseline and activity trace | Required |
| `PAPERNEG` | Farthest anchor selected in encoder space, then projected | Isolates the paper-code negative mismatch | Required |
| `PAPERNEG_NONOVERLAP` | `PAPERNEG` with valid positives at offset at least 96 | Tests raw-overlap dependence | Diagnostic only |
| `RAND_BN` | Frozen seed-matched initialization with forward-only BN replay | Measures architecture/memory performance without optimizer learning | Required |
| `BEST` / `LAST` | Two checkpoints scored from each trained trajectory | Isolates non-comparable best-loss checkpoint selection | Required, no extra training |

> `PAPERNEG_NONOVERLAP` is a causal probe, not a candidate method. A gain from `PAPERNEG` or `LAST` alone contracts the route to a simple implementation correction. A non-overlap gain authorizes method design only after ruling out generic TimesURL/SoftCLT collisions.

All variants reuse the same seed, patch architecture, test patches, memory fraction, clustering seed, top-k, point aggregation, and metric implementation. The baseline paper's full benchmark is not rerun; official and control variants are run only on identical K0 files, while the exact published 0.5296 U and 0.4263 M VUS-PR values (rounded 0.53/0.43) remain clearly marked external references [1].

#### 3.5 Mechanism-to-performance chain

The retained causal graph has three independently testable branches:

\[
\begin{aligned}
&\text{positive-view negative pool}\rightarrow\text{different negative geometry},\\
&\text{scheduled-loss checkpoint}\rightarrow\text{early representation selection},\\
&\text{near-duplicate positives}\rightarrow\text{low sustained hinge activity},\\
&\text{each branch}\rightarrow\text{paired anomaly-ranking change}.
\end{aligned}
\]

> Negative identity and temporal-gap logs test the first branch; `BEST` versus `LAST` tests the second without retraining; active rate, gradients, and `PAPERNEG_NONOVERLAP` test the third; VUS-PR/AUPRC determine whether any measured difference matters.

#### 3.6 Conditional decision boundary

The exact thresholds and deterministic file manifest are frozen in Phase C. A gain from `LAST` alone yields `SIMPLE_CHECKPOINT_FIX`; a gain from `PAPERNEG` alone yields `SIMPLE_PAPER_PARITY_FIX`; inactive dynamics without ranking gain yields `STOP_NO_PERFORMANCE_HEADROOM`; substantial activity with no execution effect yields `STOP_NO_ACTIVITY_FAILURE`. Only a cross-family gain unique to the overlap/activity branch yields `GO_METHOD_DESIGN`.

> This conditional boundary is central to the design. It prevents an already occupied generic idea - harder negatives or soft temporal positives [6,7] - from being relabeled as novelty after an inconclusive diagnosis.

#### 3.7 Baseline reference and evaluation metrics

The primary baseline is PaAno at official commit `d4c67116190efa4592dc6a8a157ced0def68b6af` [9]. The exact external full-Eval references are PaAno's reported TSB-AD-U VUS-PR 0.5296 and TSB-AD-M VUS-PR 0.4263 (rounded headlines 0.53/0.43) [1]. PAI's PaAno-enhanced U result is retained as a recent external score-level comparator, not as a same-code causal baseline [2].

Primary K0 metrics are VUS-PR and AUPRC, with per-family paired deltas. Mechanism metrics are active-hinge fraction, hinge-margin quantiles, positive/negative distance distributions, absolute encoder-gradient norms, gradient cosine, and overlap-similarity correlation. Runtime, peak VRAM, parameter count, and score-artifact hashes are mandatory system and reproducibility outputs [1,3].

## Part 2 Decision

`GO_K0_MECHANISM_ONLY / NO_METHOD_FREEZE`

The technical framework and pipeline are sufficiently specified to enter ResearchPilot Phase C. The user has granted standing authorization for automatic phase progression, so no additional routine confirmation is required. The route stops after K0 if the objective remains active, dynamics do not predict performance, or the only improvement is a one-line scaling change.

## Part 2 Pending Verification

- [x] Page/section-level source snippets and collision boundaries are recorded in `docs/literature_evidence_matrix.md`.
- [ ] Freeze Phase C numerical gates only after confirming metric variance on the six deterministic K0 files.
- [x] `PAPERNEG_NONOVERLAP` is frozen as a diagnostic intervention, not an implicit final method.

---

## Part 3 Experiment Design

> Experiment outline confirmed under the user's standing authorization: six real telemetry families, three trained trajectories plus one no-training control, paper-code and checkpoint factorial comparisons, threshold-free metrics, and a conditional multi-seed confirmation. No full baseline reproduction or FT experiment is authorized before K0 passes.

### 0 Baseline Experiment Survey

#### 0.1 PaAno (ICLR 2026) [1]

**Paper**: *PaAno: Patch-Based Representation Learning for Time-Series Anomaly Detection* | **Code**: https://github.com/jinnnju/PaAno at `d4c67116190efa4592dc6a8a157ced0def68b6af`

**Core idea**: a compact RevIN-equipped 1D CNN learns normal patch representations with triplet and temporal-pretext objectives, compresses training embeddings into a memory bank, and scores nearest-memory cosine distance.

| Dataset | Scale | Split strategy | Notes |
|---|---:|---|---|
| TSB-AD-U | 350 Eval, 48 Tuning series | Official TSB split | Exact paper VUS-PR 0.5296 (rounded 0.53) |
| TSB-AD-M | 180 Eval, 20 Tuning series | Official TSB split | Exact paper VUS-PR 0.4263 (rounded 0.43) |

| Experiment | Purpose | Comparison models | Metrics |
|---|---|---|---|
| Full U/M benchmark | Overall accuracy and efficiency | statistical, neural, and pretrained TSAD baselines | VUS-PR primary; VUS-ROC, AUPRC, AUROC, Range-F1, Point-F1 |
| Component ablation | Test InstanceNorm, losses, negative selection, memory, top-k | no-loss, no-pretext, random/closest/median/farthest, InfoNCE | same six metrics |
| Sensitivity | Test patch, batch, memory, top-k, pretext schedule | fixed parameter grid | same six metrics |

**Key hyperparameters**: patch 96, stride 1, batch 512, 100 iterations, AdamW, learning rate `1e-4`, seed 2027 in released scripts, RevIN, margin 0.1, triplet divisor 10, requested memory 10%, and top-3 scoring.

> The paper reports ten seeds, but the released scripts expose one seed. K0 uses seed 2027 for broad mechanism screening and reserves two additional seeds for conditional confirmation. This avoids presenting a one-seed screen as a final benchmark.

#### 0.2 Direct extension and evaluation references

PAI is a June 2026 direct PaAno score-level extension that restores raw-amplitude evidence and reports approximately 0.58 U VUS-PR [2]. It is retained only as prior art because its route and evaluation subset differ from the objective-activity question. TSB-AD supplies the official U/M benchmark, separate Tuning/Eval split, and VUS-PR primary metric [3]. Quo Vadis motivates simple matched controls and AUPRC rather than test-optimized F1 [4].

#### 0.3 Field convention synthesis

**Standard benchmarks**: TSB-AD-U and TSB-AD-M are the shared benchmark because they cover diverse anomaly families and define explicit Tuning/Eval partitions [1,3].

**Standard metrics**: VUS-PR is primary; AUPRC and VUS-ROC are threshold-free secondary metrics. F1 may be reported descriptively later but cannot select a K0 arm because the available implementation optimizes thresholds using labels [1,3,4].

**Ablation conventions**: PaAno varies every objective, negative strategy, memory size, patch size, batch size, and pretext schedule [1]. The present K0 uses tighter same-initialization causal arms because it is diagnosing released execution rather than claiming a final model.

**Reporting norms**: final claims require multiple seeds, per-family values, paired deltas, and uncertainty. The one-seed six-family screen is explicitly a gate, followed by three-seed confirmation only if a branch passes.

### Data and code availability summary

| Item | Status | Notes |
|---|---|---|
| PaAno code | Available | Official repository frozen and clean at the recorded SHA |
| PaAno paper | Available | Local PDF hash recorded in the literature manifest |
| TSB-AD-U archive | Available | Official archive and SHA256 stored on D drive |
| TSB-AD-M archive | Available | Official archive and SHA256 stored on D drive |
| Six K0 files | Available | Extracted under `D:/qintian_datasets/TSB-AD/paano_k0`; file hashes in `docs/K0_DATA_MANIFEST.csv` |

### 1 Datasets

#### 1.1 Frozen K0 files

| Family | Track | Domain | Rows | Channels | Train end | Selection rule |
|---|---|---|---:|---:|---:|---|
| NAB | U | facility telemetry | 4,031 | 1 | 1,007 | first eligible Eval file |
| IOPS | U | web-service telemetry | 20,000 | 1 | 5,000 | first eligible Eval file |
| Exathlon | M | cloud/facility telemetry | 43,066 | 18 | 10,766 | first eligible Eval file |
| SMD | M | server-machine telemetry | 23,694 | 38 | 4,529 | first eligible Eval file |
| SMAP | M | spacecraft sensors | 8,209 | 25 | 2,052 | first eligible Eval file |
| SWaT | M | industrial control sensors | 14,996 | 66 | 3,749 | first eligible Eval file |

> Files were selected by family and fixed ordering before any label-derived metric was computed. They span univariate and multivariate telemetry, short and long normal prefixes, and network/server/industrial sensing domains.

#### 1.2 Data preprocessing

The training split is encoded in each TSB filename. Only the normal prefix is used to fit the PaAno encoder and memory. The runner drops the final label column before creating length-96, stride-one patches and applies official RevIN. It writes complete point-score arrays before a separate evaluator reads labels. No anomaly label is accepted by the training, memory, or scoring APIs.

#### 1.3 Backup and later main-benchmark data

No backup file may replace a failed K0 file. A technical failure must be fixed on the same file. The remaining official TSB-AD-U/M Eval series are reserved for a later untouched main benchmark only after `GO_METHOD_DESIGN` leads to a frozen method; they are not used to rescue this K0.

### 2 Experiment design

#### 2.1 Main K0: execution fidelity, objective activity, and paired ranking

**Purpose**: determine whether the negative-selection mismatch, checkpoint rule, or raw-overlap shortcut changes objective activity and real anomaly ranking.

**Why designed this way**: three trained trajectories isolate one factor at a time and scoring `BEST` and `LAST` requires no additional training. `RAND_BN` is forward-replayed so BatchNorm exposure is matched without optimizer learning. All arms reuse the same per-file initialization, batch order for the matched trajectory, memory construction seed, and score code.

| Model/arm | Difference from released PaAno | Significance | Type | Source |
|---|---|---|---|---|
| `OFFICIAL_BEST` | none | same-code primary causal baseline | core baseline | PaAno code [9] |
| `OFFICIAL_LAST` | final rather than minimum-loss checkpoint | isolates checkpoint semantics | simple parity control | PaAno Algorithm 1 [1] |
| `PAPERNEG_BEST/LAST` | anchor pool and encoder-space negative selection | isolates negative execution mismatch | simple parity control | PaAno Algorithm 1 [1] |
| `PAPERNEG_NONOVERLAP_BEST/LAST` | paper negative plus zero-overlap positives at valid +/-96 offsets | tests overlap-driven activity | mechanism diagnostic | project diagnostic; not a method |
| `RAND_BN` | no optimizer update; only BN forward replay | architecture-memory floor | simple control | motivated by Quo Vadis [4] |

**Metrics**:

| Metric | Meaning | K0 use |
|---|---|---|
| VUS-PR | range-aware precision-recall ranking | primary paired endpoint [3] |
| AUPRC | pointwise threshold-free precision-recall ranking | secondary paired endpoint [4] |
| VUS-ROC | range-aware ranking complement | descriptive secondary endpoint [1,3] |
| active-hinge fraction | proportion of positive triplet margins | objective-activity endpoint |
| encoder gradient norms/cosine | magnitude and agreement of the two losses | mechanism endpoint |
| best iteration | location of released checkpoint | checkpoint endpoint |
| runtime and peak VRAM | deployment and reproducibility cost | system endpoint |

**Primary seed**: 2027. **Training count**: 6 files x 3 trajectories = 18; `RAND_BN` adds no optimization. Each trajectory produces both `BEST` and `LAST` scores.

#### 2.2 Factorial causal comparisons and decisions

| Contrast | Isolated question | Passing evidence | Decision if uniquely sufficient |
|---|---|---|---|
| `OFFICIAL_LAST - OFFICIAL_BEST` | Does scheduled-loss checkpointing hurt? | macro VUS-PR >= +0.010, macro AUPRC > 0, >=3/6 positive, worst >= -0.010 | `SIMPLE_CHECKPOINT_FIX` |
| `PAPERNEG - OFFICIAL` at matched checkpoint | Does paper-faithful negative execution help? | same performance gate | `SIMPLE_PAPER_PARITY_FIX` |
| `PAPERNEG_NONOVERLAP - PAPERNEG` at matched checkpoint | Does overlap starvation have residual headroom? | same performance gate plus low post-pretext activity in >=4/6 | `GO_METHOD_DESIGN` |
| `RAND_BN - OFFICIAL` | Does learned training add material value? | diagnostic only; no method is selected from random performance | supports stop or headroom interpretation |

Low activity is pre-registered as a median active-hinge fraction no greater than 0.10 over iterations 20-100 in at least four families. Early checkpointing is a selected best iteration no later than 30 in at least four families. Geometry or activity without paired ranking improvement cannot pass.

#### 2.3 Additional experiments

**A. Field-standard mandatory confirmation**

If and only if one K0 branch passes, rerun `OFFICIAL` and the winning branch on seeds 2027, 2028, and 2029. Report mean, standard deviation, per-family paired deltas, and a family-blocked paired bootstrap. This follows PaAno's multi-seed reporting norm while avoiding full-benchmark computation before the mechanism exists [1].

Record requested and effective memory ratios, parameters, runtime, peak VRAM, and score hashes for every run. A cached full-memory versus released-memory scoring control is performed only if effective memory ratio correlates with the branch delta; it requires no retraining.

**B. Selected extension experiments**

- **Activity-performance association**: correlate family-level activity change with paired VUS-PR change. This tests the central mechanism rather than only reporting a better score.
- **Local-shift consistency**: compare embedding consistency at offsets 1, 2, 8, 32, and 96 from saved checkpoints. This tests whether a diagnostic destroys the invariance PaAno intended.
- **Negative identity audit**: record temporal gap and anchor-positive index collisions under both pools. This verifies that `PAPERNEG` changes the intended factor.

No ablation figure is required during K0; compact CSV/JSON data are sufficient. Figures are deferred until a branch passes and manuscript writing begins.

#### 2.4 Label isolation and failure policy

The runner has no label argument and writes a completion marker plus score SHA256 before evaluation. The evaluator reads the frozen manifest and labels afterward. Hyperparameters, arm selection, checkpoint choice, and retry decisions cannot use labels. A file failure is retried only with technically equivalent batch or memory-safe execution; it cannot be replaced by a more favorable file.

Allowed final outcomes are:

```text
STOP_NO_ACTIVITY_FAILURE
STOP_NO_PERFORMANCE_HEADROOM
SIMPLE_CHECKPOINT_FIX
SIMPLE_PAPER_PARITY_FIX
GO_METHOD_DESIGN
```

No new rescue arm may be added after a failed gate.

### 3 Resource estimate (reference)

| Experiment | Estimated VRAM | Estimated time on RTX 4090 | Groups |
|---|---:|---:|---:|
| Primary K0 trajectories | 2-8 GiB, channel-dependent | 2-6 hours total | 18 training trajectories |
| BN-random controls and scoring | 2-8 GiB | 0.5-1.5 hours total | 6 files |
| Conditional three-seed confirmation | 2-8 GiB | 3-8 hours total | 24 additional trajectories maximum |
| CPU aggregation and metrics | <4 GiB RAM plus metric workspace | <1 hour | one aggregate job |

> Estimates are planning references, not scientific design constraints. Actual timing will be measured by the smoke run. The 15-minute monitor is created when the first long GPU job starts.

## Part 3 Decision

`PHASE_C_COMPLETE / PROCEED_PHASE_D`

The dataset, arm structure, instrumentation, gates, and stop outcomes are frozen. The user's standing authorization satisfies the phase-transition confirmation; implementation design may proceed immediately.

## Part 3 Phase F Amendment: User-Directed Full Main Experiment

### F-1 diagnostic conclusion

The six-file K0 completed without data, implementation, leakage, or convergence failures. It established objective inactivity and early checkpointing, but the registered execution changes did not improve the matched six-file controls. The user explicitly chose not to use that paired gate as the manuscript headline and confirmed a change in experiment design: run the project's code on the complete paper-compatible Eval lists and compare only with PaAno's paper-reported headline values.

This amendment does not erase the K0 outcome and does not introduce a new method. It expands evaluation coverage for the already registered arm.

### F-2 confirmed backtrack scope

```text
Change experiment design only.
No new architecture, loss, selector, calibration, threshold, or hyperparameter.
```

### Full main experiment

| Role | Fixed arm | Checkpoint | Seed | Coverage |
|---|---|---|---:|---:|
| Proposed full arm | PAPERNEG_NONOVERLAP | LAST | 2027 | 350 U + 180 M |
| Ablation: remove non-overlap positives | PAPERNEG | LAST | 2027 | 350 U + 180 M |
| Ablation: remove both execution changes | OFFICIAL | LAST | 2027 | 350 U + 180 M |

All arms use the same PaAno encoder, official hyperparameters, memory builder, top-3 scorer, metric code, and paper-compatible Eval lists. BEST artifacts may be generated transactionally by the existing runner but cannot replace the frozen LAST endpoint in the main table.

### Main comparison and conditional replication

The external comparison is PaAno's exact paper-reported full-Eval VUS-PR: `0.5296` for U and `0.4263` for M (rounded headlines `0.53/0.43`). Primary success requires the full method's file-weighted VUS-PR to exceed both values on the complete 350/180 lists. If and only if both tracks exceed their references, repeat the full method on seeds 2028 and 2029 and report mean and standard deviation. The ablation table remains seed 2027 and reports VUS-PR, AUPRC, and VUS-ROC for U and M.

The paper must label PaAno as `paper-reported`; it must not describe the external values as a local reproduction. The six-file K0 negative result remains a disclosed limitation of the causal interpretation.

### Phase F outcome boundary

```text
both full tracks exceed paper references -> CONTINUE_FULL_CONFIRMATION
either full track does not exceed its reference -> STOP_FULL_MAIN_FAILURE
```

No post-failure variant, family-specific rule, file removal, or metric substitution is permitted.
