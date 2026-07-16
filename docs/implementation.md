# Implementation Guide - PaAno Execution-Fidelity and Objective-Activity K0
> Generated: 2026-07-16 | Strategy: strong-baseline overlay | Status: VALIDATED_FOR_PHASE_E
> Linked design: `docs/idea_report.md` Part 3 | Frozen protocol: `configs/k0_protocol.yaml`
> Extended: deterministic RNG replay, label-isolation contract, artifact transaction contract, and K0 decision logic.
> No final method is defined here. The implementation only realizes the frozen mechanism audit and its pre-registered controls.

---

## 0 Original Project Information and Rewrite Scope

### 0.1 Frozen strong baseline

| Item | Frozen value |
|---|---|
| Project | PaAno, ICLR 2026 |
| Official repository | `https://github.com/jinnnju/PaAno` |
| Local vendor path | `C:/Users/qintian/Desktop/msn/vendor/PaAno` |
| Required Git SHA | `d4c67116190efa4592dc6a8a157ced0def68b6af` |
| Framework | PyTorch 2.7.1, Python 3.11 |
| Reused behavior | `PatchEncoder`, RevIN, patch layout, optimizer/schedule, memory construction, cosine top-k scoring, TSB VUS implementation |
| Paper headline references | Exact full-Eval VUS-PR: TSB-AD-U 0.5296; TSB-AD-M 0.4263 (rounded headlines 0.53/0.43); external references only |

> The vendor repository is read-only. The project adds an overlay package that imports the frozen encoder and metric implementation after checking the Git SHA. It does not edit, copy over, or commit vendor files. This keeps the causal controls traceable to the accepted baseline while allowing instrumentation that the official training function does not expose.

### 0.2 Rewrite scope summary

| Scope | Baseline behavior retained | Overlay change | Scientific role |
|---|---|---|---|
| Model | Official `PatchEncoder` and RevIN | SHA-guarded import only | Hold architecture fixed |
| Data | Length-96, stride-one patches; normal prefix training | Feature-only reader and memory-efficient patch views | Enforce label isolation |
| Initialization | Seeded PyTorch initialization | Persist initial-state hash; identical reconstruction for every arm | Paired causal comparison |
| Sampling | Shuffled anchor minibatches, local positives, random pretext negatives | Persist one replay plan per series/seed | Remove RNG confounding |
| Training | AdamW, cosine LR, triplet divisor 10, 19 nonzero pretext iterations | Expose `BEST` and `LAST`; log objective activity | Test checkpoint and activity hypotheses |
| Negative pool | Released positive-view projection-space pool | Add paper-faithful anchor/encoder-space pool | Test execution mismatch |
| Positive semantics | Offsets `{-2,-1,+1,+2}` | Add diagnostic offsets `{-96,+96}` | Test zero-overlap activity; not a method |
| Random control | Not released | Forward-replay BN without optimizer updates | Architecture-memory floor |
| Memory/scoring | 10% request, minimum 500, MiniBatchKMeans, top-3 cosine | Reimplemented exactly with audit fields and parity test | Same scorer for all arms |
| Evaluation | TSB metrics | Separate evaluator reads labels only after score commit | Prevent test-label use by algorithm |

The only scored arms are:

```text
OFFICIAL_BEST
OFFICIAL_LAST
PAPERNEG_BEST
PAPERNEG_LAST
PAPERNEG_NONOVERLAP_BEST
PAPERNEG_NONOVERLAP_LAST
RAND_BN
```

No learned selector, score blending, family-specific setting, new backbone, rescue arm, or final method is permitted.

## 1 Full Directory Tree After Overlay

```text
msnmsn/
├── configs/
│   └── k0_protocol.yaml                 # already frozen; sole scientific configuration
├── docs/
│   ├── idea_report.md
│   ├── user_requirements.md
│   ├── K0_DATA_MANIFEST.csv
│   └── implementation.md                # this guide
├── code/
│   ├── README.md                        # environment and exact run commands
│   ├── requirements.txt                 # excludes torch/torchvision/torchaudio
│   ├── pyproject.toml                   # local package and pytest configuration
│   ├── src/
│   │   └── paano_k0/
│   │       ├── __init__.py
│   │       ├── schemas.py               # enums and immutable data contracts
│   │       ├── config.py                # YAML parsing, validation, job expansion
│   │       ├── vendor.py                # PaAno SHA guard and symbol loader
│   │       ├── feature_data.py          # runner-visible, feature-only I/O and patch views
│   │       ├── label_data.py            # evaluator-only label I/O
│   │       ├── replay.py                # shared initialization and RNG replay plans
│   │       ├── objectives.py            # OFFICIAL/PAPERNEG loss semantics
│   │       ├── instrumentation.py       # per-iteration mechanism records
│   │       ├── trainer.py               # three trajectories and RAND_BN replay
│   │       ├── memory.py                # exact PaAno memory-bank construction
│   │       ├── scoring.py               # patch and point scoring
│   │       ├── artifacts.py             # atomic writes, hashes, completion markers
│   │       ├── run_series.py            # label-free per-series entry point
│   │       ├── evaluate_scores.py       # label-reading evaluation entry point
│   │       └── aggregate.py             # frozen contrasts, gates, and decision
│   ├── scripts/
│   │   ├── 00_smoke.ps1
│   │   ├── 01_run_primary_k0.ps1
│   │   ├── 02_evaluate_primary_k0.ps1
│   │   ├── 03_aggregate_decision.ps1
│   │   ├── 04_run_confirmation.ps1
│   │   └── monitor_k0.ps1
│   └── tests/
│       ├── test_vendor.py
│       ├── test_label_isolation.py
│       ├── test_replay.py
│       ├── test_objectives.py
│       ├── test_trainer.py
│       ├── test_scoring_parity.py
│       ├── test_artifact_contract.py
│       ├── test_aggregate_decision.py
│       └── test_experiment_coverage.py
├── results/                             # gitignored; scores, checkpoints, metrics
└── logs/                                # gitignored; PowerShell and monitor logs
```

## 2 Per-File Function Table

| File | Public functions/classes | Input | Output | Called by |
|---|---|---|---|---|
| `schemas.py` | `Trajectory`, `CheckpointKind`, `SeriesSpec`, `IterationReplay`, `ReplayPlan`, `RunJob`, `TrainingSummary`, `ScoreManifest`, `MetricRow`, `scored_checkpoints`, `make_run_id` | typed primitive fields | immutable contracts and IDs | all modules |
| `config.py` | `load_protocol`, `validate_protocol`, `load_series_manifest`, `expand_primary_jobs`, `expand_confirmation_jobs` | YAML/CSV paths | validated config and job lists | scripts, runners, tests |
| `vendor.py` | `verify_vendor_repo`, `load_vendor_symbols`, `build_encoder`, `compute_baseline_window` | vendor path, expected SHA, channel count | guarded baseline symbols/model/window | runner, evaluator, parity tests |
| `feature_data.py` | `read_feature_series`, `split_normal_prefix`, `PatchStore`, `make_patch_store` | `SeriesSpec`, CSV | float32 features and `[N,C,L]` patch view | runner, trainer, scorer |
| `label_data.py` | `read_labels`, `validate_score_alignment` | `SeriesSpec`, committed score manifest | binary labels | evaluator only |
| `replay.py` | `stable_seed`, `seed_everything`, `state_dict_sha256`, `build_initial_state`, `build_replay_plan`, `save_replay_plan`, `load_replay_plan`, `materialize_positive_indices`, `materialize_unadjacent_indices` | series hash, seed, patch count | repeatable initialization and 100-step plan | runner, trainer |
| `objectives.py` | `pretext_weight`, `official_negative_indices`, `paper_negative_indices`, `compute_triplet_batch`, `compute_pretext_batch`, `encoder_gradient_diagnostics` | embeddings `[B,D]`, indices `[B]` | losses and mechanism tensors | trainer |
| `instrumentation.py` | `quantile_summary`, `build_iteration_record`, `IterationRecorder` | loss/geometry tensors and metadata | schema-checked JSONL | trainer |
| `trainer.py` | `cosine_learning_rate`, `train_trajectory`, `replay_rand_bn` | model, patch store, replay plan, arm config | checkpoints and training summary | runner |
| `memory.py` | `encode_store`, `effective_memory_count`, `create_memory_bank` | checkpoint model, training patches | memory `[M,D]`, indices, audit | runner, parity tests |
| `scoring.py` | `score_patch_store`, `distribute_patch_scores`, `score_checkpoint` | checkpoint, full patches, memory | point score `[T]` and score audit | runner |
| `artifacts.py` | `sha256_file`, `atomic_write_json`, `atomic_save_numpy`, `atomic_save_checkpoint`, `commit_score_artifact`, `verify_committed_score` | paths and payloads | durable artifacts/verification | runner, evaluator |
| `run_series.py` | `parse_args`, `run_job`, `main` | no-label CLI job | checkpoints, scores, logs, `_SUCCESS` | PowerShell scripts |
| `evaluate_scores.py` | `compute_threshold_free_metrics`, `evaluate_score_artifact`, `parse_args`, `main` | committed score and labels | metric JSON/CSV rows | evaluation script |
| `aggregate.py` | `load_metric_rows`, `family_macro`, `contrast_rows`, `performance_gate`, `activity_gate`, `checkpoint_gate`, `decide_k0`, `write_aggregate_outputs`, `main` | evaluator outputs and training logs | summaries and frozen outcome | aggregation script |
| `00_smoke.ps1` | orchestration only | one frozen file/seed | smoke artifacts and test exit code | operator |
| `01_run_primary_k0.ps1` | orchestration only | six series, seed 2027 | 18 trajectories plus RAND_BN | operator/automation |
| `02_evaluate_primary_k0.ps1` | orchestration only | committed score roots | evaluator outputs | operator/automation |
| `03_aggregate_decision.ps1` | orchestration only | metrics/log roots | K0 decision | operator |
| `04_run_confirmation.ps1` | conditional orchestration only | passing branch and seeds 2027-2029 | confirmation artifacts | operator after a pass |
| `monitor_k0.ps1` | read-only status collection | runner PID/result root | one concise status row | 15-minute automation |

> `scripts/` contains no scientific logic. Every arm, seed, threshold, and gate is loaded from `configs/k0_protocol.yaml`.

## 3 Shared Contracts and Tensor Shapes

### 3.1 Type contracts in `schemas.py`

**`Trajectory(str, Enum)`** has exactly `OFFICIAL`, `PAPERNEG`, `PAPERNEG_NONOVERLAP`, and `RAND_BN`.

**`CheckpointKind(str, Enum)`** has exactly `BEST`, `LAST`, and `BN_CALIBRATED`.

**`SeriesSpec`**

```python
@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    family: str
    track: Literal["U", "M"]
    csv_path: Path
    csv_sha256: str
    rows: int
    channels: int
    train_end: int
    feature_columns: tuple[str, ...]
    label_column: str
```

`csv_path` must match the frozen manifest, `train_end >= patch_size`, `rows` must match the CSV, and the runner may use only `feature_columns`. `label_column` is metadata for `label_data.py`; it is never passed to feature or model functions.

**`IterationReplay`** stores `anchor_indices: NDArray[int64]` of shape `[B_i]`, `positive_uniform: NDArray[float32]` of shape `[B_i]`, and `unadjacent_uniform: NDArray[float32]` of shape `[B_i,5]`. **`ReplayPlan`** stores exactly 100 such records plus `series_id`, `seed`, `n_train_patches`, `batch_size`, and a payload SHA256.

**`RunJob`** contains `SeriesSpec`, `Trajectory`, `seed`, frozen paths, device, and output root. It deliberately has no label field.

**`TrainingSummary`** contains trajectory, seed, initial-state hash, replay hash, best iteration/loss, last iteration, runtime seconds, peak VRAM MiB, and checkpoint hashes.

**`ScoreManifest`** contains the provenance and schema listed in Section 8.2. **`MetricRow`** is created only in the evaluator.

**`scored_checkpoints(trajectory: Trajectory) -> tuple[CheckpointKind, ...]`** returns `(BEST,LAST)` for trained trajectories and `(BN_CALIBRATED,)` for `RAND_BN`.

**`make_run_id(series_id: str, seed: int, trajectory: Trajectory, checkpoint: CheckpointKind) -> str`** returns a filesystem-safe, deterministic ID and rejects unregistered combinations.

### 3.2 Shape table

| Symbol | dtype/shape | Meaning |
|---|---|---|
| `x_full` | float32 `[T,C]` | full feature series, no label column |
| `x_train` | float32 `[T_train,C]` | chronological normal prefix |
| `train_store` | float32 view `[N_train,C,96]` | `N_train=T_train-96+1` stride-one patches |
| `full_store` | float32 view `[N_full,C,96]` | `N_full=T-96+1` full-series patches |
| `anchor_indices` | int64 `[B]` | replayed training patch indices |
| `anchors`, `positives` | float32 `[B,C,96]` | paired patch tensors |
| `h_anchor`, `h_positive` | float32 `[B,64]` | official encoder embeddings |
| `z_anchor`, `z_positive`, `z_negative` | float32 `[B,256]` | projection-head outputs, L2 normalized for triplet geometry |
| `similarity` | float32 `[B,B]` | candidate pair cosine similarities |
| `negative_index` | int64 `[B]` | candidate row selected for each anchor |
| `pretext_features` | float32 `[B_valid+5B,128]` | concatenated embedding pairs |
| `memory` | float32 CPU `[M,64]` | compressed normal encoder embeddings |
| `patch_scores` | float32 `[N_full]` | mean distance to top-3 memory vectors |
| `point_scores` | float32 `[T]` | overlap-averaged patch scores |
| `labels` | int8 `[T]` | evaluator-only binary labels |

Every function asserts rank, dtype, finiteness, and matching leading dimensions at its boundary. Multivariate input is never flattened across channels before the encoder.

## 4 Per-File Implementation Details

### 4.1 `config.py`

**`load_protocol(path: Path) -> ProtocolConfig`** reads YAML with `yaml.safe_load`, converts it into immutable dataclasses, calls `validate_protocol`, and retains the source SHA256.

**`validate_protocol(config: ProtocolConfig) -> None`** fails unless the baseline SHA, arm IDs, seeds, patch/stride, 100 iterations, batch 512, LR, margin, divisor, pretext schedule, memory rule, top-k, instrumentation fields, gates, and forbidden list equal the frozen protocol. Unknown keys are errors rather than warnings.

**`load_series_manifest(path: Path) -> tuple[SeriesSpec, ...]`** parses `docs/K0_DATA_MANIFEST.csv`, resolves paths without following unverified substitutions, verifies six unique families/tracks and file hashes, and returns the manifest order. It never reads CSV labels or metrics.

**`expand_primary_jobs(config: ProtocolConfig, series: Sequence[SeriesSpec], vendor_root: Path, output_root: Path) -> tuple[RunJob, ...]`** emits 24 runner jobs: three optimized trajectories plus one `RAND_BN` job for each of six files. A trained job creates both checkpoints, so this is 18 optimized trajectories, not 36.

**`expand_confirmation_jobs(config: ProtocolConfig, decision_path: Path, ...) -> tuple[RunJob, ...]`** runs only after a passing outcome. It permits `OFFICIAL` and the single winning trajectory on seeds 2027-2029, de-duplicates primary-seed artifacts, and rejects stop outcomes.

### 4.2 `vendor.py`

**`verify_vendor_repo(vendor_root: Path, expected_sha: str) -> VendorFingerprint`** executes read-only `git -C <root> rev-parse HEAD`, checks the 40-character SHA exactly, records dirty status without changing files, verifies `model.py`, `utils/metrics.py`, `utils/basic_metrics.py`, and `utils/data_preprocess.py`, and raises `VendorMismatchError` on any mismatch.

**`load_vendor_symbols(vendor_root: Path, expected_sha: str) -> VendorSymbols`** calls the guard, prepends the resolved root to `sys.path` only for the process, imports `PatchEncoder`, `basic_metricor`, `generate_curve`, and `find_length_rank`, validates their expected call surfaces, and returns references plus the fingerprint.

**`build_encoder(symbols: VendorSymbols, channels: int, use_revin: bool, device: torch.device) -> nn.Module`** instantiates the official encoder with default layer/projection settings and moves it to the device. No wrapper changes `forward`, BN momentum, RevIN, or heads.

**`compute_baseline_window(symbols: VendorSymbols, x_first_channel: NDArray[float32]) -> int`** calls frozen `find_length_rank(..., rank=1)` and returns the integer used later by VUS metrics.

### 4.3 `feature_data.py` and `label_data.py`

**`read_feature_series(spec: SeriesSpec) -> NDArray[np.float32]`** reads only `spec.feature_columns` through `pandas.read_csv(usecols=...)`, returns contiguous `[T,C]`, verifies the file hash/row/channel counts, and fails on NaN/Inf. It never materializes the label column.

**`split_normal_prefix(x_full: NDArray[np.float32], train_end: int) -> NDArray[np.float32]`** returns `x_full[:train_end]` after range checks. Because the frozen protocol uses RevIN, it performs no full-series or label-conditioned standardization.

**`PatchStore`** stores a contiguous CPU tensor `[T,C]`, patch length, and stride. `__len__() -> int` returns `(T-L)//s+1`. `take(indices: Tensor) -> Tensor` returns `[B,C,L]` by indexed access to an `unfold` view. `iter_batches(batch_size: int) -> Iterator[tuple[Tensor,Tensor]]` yields ordered full-store patches and their starts for memory/scoring. It never stores labels.

**`make_patch_store(x: NDArray[np.float32], patch_size: int, stride: int) -> PatchStore`** validates `T>=L`, creates the view, and verifies the first/last patch against NumPy slices.

**`read_labels(spec: SeriesSpec) -> NDArray[np.int8]`** exists only in `label_data.py`; it reads only `spec.label_column`, checks length and binary values, and is imported only by `evaluate_scores.py`.

**`validate_score_alignment(labels: NDArray, scores: NDArray, expected_rows: int) -> None`** checks one-dimensional equal length, finite scores, and the expected series length.

### 4.4 `replay.py`: shared initialization and random replay

**`stable_seed(base_seed: int, series_sha256: str, namespace: str) -> int`** hashes the tuple with SHA256 and maps it into `[0,2**31-1]`. Namespaces `model_init` and `training_replay` are separate.

**`seed_everything(seed: int, deterministic: bool=True) -> None`** seeds Python, NumPy, CPU CUDA RNGs; enables deterministic algorithms and fixed cuDNN settings; and records relevant environment flags.

**`state_dict_sha256(state: Mapping[str,Tensor]) -> str`** hashes sorted keys, dtype, shape, and contiguous CPU bytes. BN buffers are included.

**`build_initial_state(build_fn: Callable[[],nn.Module], init_seed: int) -> tuple[dict[str,Tensor],str]`** seeds, constructs one official encoder, clones its complete state to CPU, and returns the state plus hash. Every arm reconstructs this state independently and must record the same hash.

**`build_replay_plan(n_patches: int, batch_size: int, iterations: int, seed: int, num_unadjacent: int=5) -> ReplayPlan`** uses one local NumPy generator, emits successive shuffled epochs until exactly 100 minibatches are recorded, and stores for each minibatch one positive uniform per anchor and five unadjacent uniforms per anchor. It never touches global RNG or data values. NumPy draws are generated in the original float64 stream and then stored as float32; if the cast rounds a valid draw immediately below one to the closed endpoint `1.0`, only that stored endpoint is canonicalized to `nextafter(float32(1), float32(0))`. This preserves draw count, RNG state, every non-endpoint value, and the intended final-candidate ordinal while enforcing the registered `[0,1)` replay invariant.

**`materialize_positive_indices(anchor_indices: NDArray[int64], n_patches: int, offsets: tuple[int,...], uniform: NDArray[float32]) -> NDArray[int64]`** forms valid candidates in the declared offset order and maps each uniform to a candidate ordinal. `OFFICIAL` and `PAPERNEG` therefore receive identical local positives. For `PAPERNEG_NONOVERLAP`, it asserts that at least one of `-96,+96` is valid for every anchor and that raw overlap is exactly zero; there is no self-patch fallback.

**`materialize_unadjacent_indices(batch_size: int, uniform: NDArray[float32]) -> NDArray[int64]`** maps uniforms to offsets `1..B-1` and applies modulo indexing, matching PaAno's non-self minibatch pretext sampling.

**`save_replay_plan(plan: ReplayPlan, path: Path) -> str`** writes compressed NPZ plus JSON metadata atomically and returns payload hash. **`load_replay_plan(path: Path, expected: ReplayIdentity) -> ReplayPlan`** verifies the hash and identity. All four trajectories use the same plan file for a series/seed.

### 4.5 `objectives.py`: exact arm semantics

**`pretext_weight(iteration: int, total_iterations: int, initial_weight: float=1.0) -> float`** returns `1 - iteration/(total_iterations/5)` only for `iteration < total_iterations/5`, otherwise zero. With 100 iterations, nonzero weights occur at iterations 1-19.

**`official_negative_indices(z_anchor: Tensor, z_positive: Tensor) -> Tensor`** expects normalized `[B,256]`, computes `1-z_anchor@z_positive.T`, masks the matched diagonal to `-inf` for maximization, and returns the farthest positive-view row for each anchor. This exactly encodes the released pool and projection-space selection.

**`paper_negative_indices(h_anchor: Tensor) -> Tensor`** normalizes `[B,64]` encoder embeddings, computes pairwise cosine distance among minibatch anchors, masks the self diagonal, and returns the farthest anchor row. The selected `h_anchor[j]` is then passed through the projection head before the hinge is computed. This encodes the paper algorithm rather than a new mining rule.

**`compute_triplet_batch(model: nn.Module, h_anchor: Tensor, h_positive: Tensor, anchor_indices: Tensor, positive_indices: Tensor, trajectory: Trajectory, margin: float, divisor: float, temperature: float) -> TripletBatch`**:

1. Project and normalize anchors/positives to `[B,256]`.
2. Select negative indices by the exact arm rule above.
3. For `OFFICIAL`, use the already projected positive candidate; for both paper-negative arms, project the selected encoder anchor.
4. Compute `d_pos=1-cos(z_a,z_p)`, `d_neg=1-cos(z_a,z_n)`, and `hinge=relu(d_pos-d_neg+0.1)`.
5. Return unscaled mean, divided mean, per-sample distances/margins, active mask, negative candidate indices, temporal gaps, and index collisions.

**`compute_pretext_batch(model: nn.Module, h_anchor: Tensor, h_pretext: Tensor, valid_mask: Tensor, unadjacent_indices: Tensor, criterion: nn.Module) -> PretextBatch`** constructs valid adjacent pairs and five non-adjacent pairs per anchor, applies the official classification head and BCE-with-logits, and returns `loss_positive+loss_negative`, logits, labels, valid count, and accuracy. If the pretext weight is zero, the trainer skips this function and records zero loss/undefined accuracy.

**`encoder_gradient_diagnostics(triplet_scaled: Tensor, weighted_pretext: Tensor, encoder_parameters: Sequence[nn.Parameter]) -> GradDiagnostics`** uses `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` separately for the two terms. It returns L2 norms and cosine over the same encoder parameters without accumulating `.grad`; the subsequent normal `final_loss.backward()` is unchanged. When the weighted pretext term is zero, pretext norm is zero and cosine is JSON `null`.

### 4.6 `instrumentation.py`

**`quantile_summary(x: Tensor, probabilities: tuple[float,...]=(0,.25,.5,.75,1)) -> dict[str,float]`** detaches float64 CPU values, rejects non-finite input, and emits min/Q1/median/Q3/max.

**`build_iteration_record(...) -> dict[str,JSONScalar]`** accepts the iteration number, replay indices, objective outputs, gradient diagnostics, LR, pretext weight, runtime, and memory counters. It emits every per-iteration key frozen in YAML. `raw_overlap_ratio` is `max(0,1-abs(offset)/96)` summarized over the batch. `best_checkpoint_iteration` is the best-so-far iteration at record time.

**`IterationRecorder(path: Path, expected_iterations: int)`** opens one trajectory JSONL through a temporary file. `append(record: Mapping) -> None` validates keys and monotonic iterations and flushes each line. `close() -> str` requires exactly 100 records, atomically renames, and returns SHA256.

### 4.7 `trainer.py`: optimized trajectories and BN random control

**`cosine_learning_rate(iteration: int, total: int, initial_lr: float, final_ratio: float=.1) -> float`** exactly implements PaAno's cosine schedule from `lr` to `lr/10`.

**`train_trajectory(model: nn.Module, initial_state: Mapping[str,Tensor], store: PatchStore, replay: ReplayPlan, trajectory: Literal[OFFICIAL,PAPERNEG,PAPERNEG_NONOVERLAP], protocol: ProtocolConfig, device: torch.device, recorder: IterationRecorder) -> TrainingResult`** performs exactly 100 updates:

1. Load the shared initial state; assert its hash; enter train mode; create AdamW with LR `1e-4`, weight decay `1e-4`.
2. Read replayed anchor indices and materialize arm-specific positives. The anchor minibatches, local-pair random uniforms, and pretext random uniforms are shared across arms.
3. Build `[anchors,positives,pretext]` in one encoder forward while weight is nonzero and `[anchors,positives]` afterward, matching BN exposure in the released path.
4. Compute arm-specific triplet, pretext term, diagnostics, and `final_loss=triplet_after_divisor+lambda*pretext_loss`.
5. Zero gradients, backpropagate once, and step AdamW. Diagnostics must not alter `.grad`.
6. Compare the pre-update scalar `final_loss.item()` with best loss; if lower, clone the **post-update** complete state, matching released checkpoint timing. Record best iteration.
7. After iteration 100, clone `LAST`. Save both states only after the recorder validates 100 rows.

The return holds CPU `BEST`/`LAST` state dictionaries, hashes, best iteration/loss, final loss, runtime, and peak allocated VRAM. No score or label participates in checkpoint choice.

**`replay_rand_bn(model: nn.Module, initial_state: Mapping[str,Tensor], store: PatchStore, replay: ReplayPlan, protocol: ProtocolConfig, device: torch.device, recorder: IterationRecorder) -> TrainingResult`** loads the same initialization, keeps train mode, and executes 100 `torch.no_grad()` encoder forwards with the exact `OFFICIAL` anchor/positive/pretext concatenation and pretext schedule. It performs no projection loss, backward, optimizer creation, or parameter update. The test verifies that trainable tensors are byte-identical to initialization while BN running buffers/counters change. It saves only `BN_CALIBRATED`.

### 4.8 `memory.py` and `scoring.py`

**`encode_store(model: nn.Module, store: PatchStore, batch_size: int, device: torch.device) -> tuple[Tensor,Tensor]`** runs eval/inference mode and returns CPU embeddings `[N,64]` plus patch starts `[N]` in deterministic order.

**`effective_memory_count(n: int, requested_fraction: float) -> int`** computes `k=round(fraction*n)`, `minimum=min(500,max(1,n-1))`, and `max(minimum,min(k,n-1))`, exactly matching the baseline.

**`create_memory_bank(model: nn.Module, train_store: PatchStore, requested_fraction: float, batch_size: int, device: torch.device, random_state: int=42) -> MemoryResult`** normalizes encoder embeddings, applies `MiniBatchKMeans(n_clusters=M, init='k-means++', batch_size=max(8192,M), max_iter=50, n_init=1, reassignment_ratio=.01)`, selects the nearest real embedding per center, and returns unnormalized memory vectors, source indices, requested/effective fractions, and hash. It never uses labels.

**`score_patch_store(model: nn.Module, full_store: PatchStore, memory: Tensor, top_k: int, batch_size: int, device: torch.device) -> NDArray[np.float32]`** reproduces official NaN handling, L2 normalization, cosine similarity, largest top-3 similarity, mean `1-similarity`, and ordered CPU output `[N_full]`.

**`distribute_patch_scores(patch_scores: NDArray[np.float32], patch_size: int, num_points: int) -> NDArray[np.float32]`** reproduces the official convolution of sums and counts and returns `[T]`.

**`score_checkpoint(model: nn.Module, checkpoint_state: Mapping[str,Tensor], train_store: PatchStore, full_store: PatchStore, protocol: ProtocolConfig, device: torch.device) -> ScoreResult`** loads one checkpoint, builds a new identically seeded memory, scores the full store, distributes scores, and reports runtime, peak VRAM, memory ratios/hash, and score hash. BEST and LAST never share an incorrectly constructed memory bank.

### 4.9 `artifacts.py`: transactional score commit

**`sha256_file(path: Path, chunk_bytes: int=8_388_608) -> str`** streams a file hash.

**`atomic_write_json(path: Path, payload: Mapping) -> None`**, **`atomic_save_numpy(path: Path, array: NDArray) -> None`**, and **`atomic_save_checkpoint(path: Path, state: Mapping[str,Tensor]) -> None`** write a sibling temporary file, flush/fsync, then `os.replace`.

**`commit_score_artifact(run_dir: Path, scores: NDArray[np.float32], manifest: ScoreManifest) -> Path`** writes `scores.npy`, verifies its SHA256, writes `score_manifest.json` with that hash, then creates `_SUCCESS` last. A failed run keeps diagnostic files but never creates `_SUCCESS`.

**`verify_committed_score(run_dir: Path) -> tuple[NDArray[np.float32],ScoreManifest]`** requires `_SUCCESS`, validates schema/provenance/hash/length, and refuses partial or mutated results before labels can be loaded.

### 4.10 `run_series.py`: label-free runner

**`parse_args(argv: Sequence[str]|None=None) -> argparse.Namespace`** exposes only `--config`, `--manifest`, `--series-id`, `--trajectory`, `--seed`, `--vendor-root`, `--output-root`, and `--device`. There is no label path, metric, gate, checkpoint-choice, or family-parameter argument.

**`run_job(job: RunJob, protocol: ProtocolConfig) -> tuple[Path,...]`**:

1. Verify protocol, manifest row, dataset hash, and vendor SHA.
2. Load features only; build normal/full stores and baseline sliding window.
3. Derive shared initialization and replay identities; atomically create or verify replay plan.
4. Run the selected trajectory. Trained trajectories produce BEST and LAST; RAND_BN produces one checkpoint.
5. Score every registered checkpoint, commit each score artifact, then write trajectory `training_summary.json` and trajectory `_SUCCESS`.
6. On exception, write `_FAILED.json` with type/message/traceback and re-raise. Do not skip or substitute a series.

**`main() -> int`** constructs exactly one job and returns nonzero on any failure. It does not import `label_data` or `evaluate_scores`.

### 4.11 `evaluate_scores.py`: evaluator-only labels

**`compute_threshold_free_metrics(scores: NDArray[np.float32], labels: NDArray[np.int8], sliding_window: int, vendor: VendorSymbols, thresholds: int=250) -> dict[str,float]`** calls frozen `basic_metricor.metric_PR`, `metric_ROC`, and `generate_curve` to return only `auprc`, `auroc`, `vus_pr`, and `vus_roc`. AUROC is descriptive; no F1 or oracle threshold is computed.

**`evaluate_score_artifact(run_dir: Path, spec: SeriesSpec, vendor: VendorSymbols) -> MetricRow`** verifies the committed score and score hash **before** calling `read_labels`, checks alignment, computes metrics, and atomically writes `metrics.json`. Metric provenance includes score SHA, data SHA, config SHA, vendor SHA, and evaluator version.

**`parse_args`/`main`** accept score roots, frozen manifest/config, and vendor root; discover only registered successful runs and fail if any expected artifact is missing. Labels cannot influence retries or runner configuration.

### 4.12 `aggregate.py`: pre-registered contrasts and outcome

**`load_metric_rows(metrics_root: Path, expected_jobs: Sequence[RunJob]) -> tuple[MetricRow,...]`** requires exactly 42 primary metric rows (six files times seven scored arms), unique provenance, and no extras.

**`family_macro(rows: Sequence[MetricRow], arm: str, metric: str) -> float`** averages one file per frozen family equally. **`contrast_rows(rows, treatment, control) -> tuple[ContrastRow,...]`** performs paired family differences only.

**`performance_gate(contrasts: Sequence[ContrastRow]) -> GateResult`** applies all frozen conditions: macro VUS-PR delta at least `+0.010`, macro AUPRC delta greater than zero, at least three of six positive VUS-PR deltas, and worst delta at least `-0.010`.

The primary, non-selectable contrasts are fixed as:

```text
checkpoint: OFFICIAL_LAST - OFFICIAL_BEST
paper negative: PAPERNEG_LAST - OFFICIAL_LAST
overlap: PAPERNEG_NONOVERLAP_LAST - PAPERNEG_LAST
```

BEST-matched negative/overlap contrasts are reported as robustness diagnostics but cannot change the decision.

**`activity_gate(iteration_logs: Sequence[Path], series_to_family: Mapping[str,str]) -> GateResult`** uses `OFFICIAL` only, maps each logged `series_id` through the frozen manifest, computes the per-family median active-hinge fraction for iterations 20-100, and passes if it is at most 0.10 in at least four families.

**`checkpoint_gate(summaries: Sequence[TrainingSummary]) -> GateResult`** reports early checkpointing if `OFFICIAL.best_iteration<=30` in at least four families; it is diagnostic in addition to the checkpoint performance contrast.

**`decide_k0(...) -> DecisionRecord`** uses this fixed precedence:

1. `GO_METHOD_DESIGN` only if the incremental overlap contrast passes and low activity passes.
2. Otherwise `SIMPLE_PAPER_PARITY_FIX` if the paper-negative contrast passes.
3. Otherwise `SIMPLE_CHECKPOINT_FIX` if the checkpoint contrast passes.
4. Otherwise `STOP_NO_ACTIVITY_FAILURE` when the low-activity gate fails.
5. Otherwise `STOP_NO_PERFORMANCE_HEADROOM`.

No failed outcome creates a new arm. `RAND_BN` and BEST robustness results are diagnostic and cannot independently produce a GO.

**`write_aggregate_outputs(...) -> None`** writes the schemas below, including every negative family result. **`main() -> int`** fails on incomplete coverage rather than averaging available files.

## 5 Data and Label-Isolation Flow

```text
Frozen K0 manifest + verified TSB CSV
  -> read CSV header and feature columns only
  -> x_full [T,C]
      -> x_train = x_full[:train_end]
      -> train/full PatchStore views [N,C,96]
      -> shared initialization + replay plan
      -> train or RAND_BN replay
      -> checkpoint-specific memory and scores
      -> scores.npy + manifest/hash
      -> _SUCCESS written last

Only after _SUCCESS and hash verification:
  evaluator -> reads label column [T]
            -> VUS-PR/AUPRC/VUS-ROC/AUROC
            -> metrics.json
  aggregator -> paired contrasts and frozen decision
```

Static dependency rule:

```text
run_series -> feature_data, never label_data/evaluate_scores/aggregate
trainer/memory/scoring -> no pandas and no label type
evaluate_scores -> label_data only after verify_committed_score
```

## 6 Windows PowerShell Scripts

All scripts start with `$ErrorActionPreference='Stop'`, resolve repository paths, invoke `D:/Anaconda/envs/paano_msn/python.exe` directly, and tee timestamped logs. They do not activate an old PrefixCal/TSPulse environment.

**`00_smoke.ps1`** runs `pytest -q`, then one shortest frozen series with `OFFICIAL`, evaluates both checkpoints, verifies 100 instrumentation rows, score length/hash, and CUDA use. It deletes only its dedicated `results/smoke` directory before a rerun.

**`01_run_primary_k0.ps1`** loops frozen manifest order and trajectories `OFFICIAL`, `PAPERNEG`, `PAPERNEG_NONOVERLAP`, `RAND_BN`, seed 2027. It resumes only jobs with valid trajectory `_SUCCESS`; `_FAILED.json` causes immediate nonzero exit. It launches no concurrent GPU training jobs on the single 4090.

**`02_evaluate_primary_k0.ps1`** invokes the evaluator only after all 24 trajectory success markers exist and requires all 42 score artifacts.

**`03_aggregate_decision.ps1`** invokes aggregation, prints the single allowed outcome, and never launches confirmation after a stop.

**`04_run_confirmation.ps1`** reads `decision.json`; for a simple fix or GO only, it runs OFFICIAL and the winning branch at seeds 2028/2029, evaluates, and produces mean/std plus paired family-blocked bootstrap. It cannot run for a stop outcome.

**`monitor_k0.ps1`** is read-only and prints: runner alive, completed/failed trajectory counts, current series/trajectory, GPU utilization, VRAM used/total, temperature, C/D free GiB, and latest log timestamp. It is the command used by the requested 15-minute automation once the long GPU run begins.

## 7 Tests

| Test file/function | Required assertion |
|---|---|
| `test_vendor.py::test_vendor_sha_guard` | correct SHA passes; a wrong SHA fails before import |
| `test_vendor.py::test_encoder_forward_contract` | official model emits `[B,64]` embeddings and `[B,256]` projections |
| `test_label_isolation.py::test_feature_reader_never_loads_label` | monkeypatched `read_csv` observes feature-only `usecols` |
| `test_label_isolation.py::test_runner_has_no_label_surface` | runner imports/CLI/signatures contain no label argument or label module |
| `test_label_isolation.py::test_evaluator_reads_label_after_hash` | corrupt/missing score fails before label reader is called |
| `test_replay.py::test_replay_is_bitwise_repeatable` | identical series/seed gives identical NPZ/hash |
| `test_replay.py::test_all_arms_share_anchor_batches` | all trajectories consume the same anchor plan |
| `test_replay.py::test_local_arms_share_positive_indices` | OFFICIAL and PAPERNEG positives are identical |
| `test_replay.py::test_nonoverlap_offsets_have_zero_overlap` | every offset magnitude is 96 and no fallback occurs |
| `test_objectives.py::test_official_triplet_numerical_parity` | output matches a direct transcription of vendor lines 116-132 within `1e-7` |
| `test_objectives.py::test_paper_negative_uses_anchor_encoder_space` | hand-built embeddings select expected farthest anchors before projection |
| `test_objectives.py::test_pretext_schedule_exact` | iterations 1-19 positive, 20-100 zero |
| `test_objectives.py::test_gradient_diagnostics_do_not_accumulate_grad` | `.grad` remains empty before final backward and final gradients match no-instrumentation run |
| `test_trainer.py::test_best_is_post_update_minibatch_loss_state` | synthetic trajectory preserves exact BEST/LAST timing |
| `test_trainer.py::test_rand_bn_changes_only_bn_buffers` | parameters unchanged; BN counters/statistics updated |
| `test_scoring_parity.py::test_memory_count_and_centers_parity` | count/indices/memory match frozen helper on a fixture |
| `test_scoring_parity.py::test_score_and_distribution_parity` | patch and point scores match vendor functions within `1e-6` |
| `test_artifact_contract.py::test_success_marker_is_last` | evaluator rejects partial artifacts |
| `test_artifact_contract.py::test_score_hash_detects_mutation` | one-byte mutation is rejected |
| `test_aggregate_decision.py::test_gate_boundaries` | exact threshold edges and precedence match YAML |
| `test_aggregate_decision.py::test_incomplete_family_fails` | 41/42 rows cannot aggregate |
| `test_experiment_coverage.py::test_primary_job_matrix` | six files x four trajectories and 42 scored arms are exact |
| `test_experiment_coverage.py::test_forbidden_arm_rejected` | unregistered rescue arm/config key fails validation |

Smoke acceptance additionally requires two independent OFFICIAL runs to have equal initial/replay/checkpoint/score hashes on the same device and environment.

## 8 Results File Format

### 8.1 Directory layout

```text
results/k0/
├── replay/{series_id}/seed_{seed}.{npz,json}
├── runs/{series_id}/seed_{seed}/{trajectory}/
│   ├── iteration_metrics.jsonl
│   ├── training_summary.json
│   ├── checkpoints/{BEST|LAST|BN_CALIBRATED}.pt
│   ├── scores/{checkpoint}/
│   │   ├── scores.npy
│   │   ├── score_manifest.json
│   │   ├── metrics.json                 # evaluator creates later
│   │   └── _SUCCESS
│   ├── _SUCCESS
│   └── _FAILED.json                     # only on failure; retained
└── aggregate/
    ├── file_metrics.csv
    ├── family_metrics.csv
    ├── paired_contrasts.csv
    ├── activity_summary.csv
    ├── runtime_summary.csv
    └── decision.json
```

### 8.2 `score_manifest.json`

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | fixed artifact schema version |
| `run_id` | string | deterministic series/seed/trajectory/checkpoint ID |
| `series_id`, `family`, `track` | string | frozen manifest identity |
| `data_sha256`, `config_sha256`, `vendor_sha` | string | complete provenance |
| `seed`, `trajectory`, `checkpoint` | int/string | arm identity |
| `initial_state_sha256`, `replay_sha256`, `checkpoint_sha256` | string | causal pairing provenance |
| `num_points`, `num_train_patches`, `num_full_patches`, `channels` | int | shape audit |
| `patch_size`, `stride`, `top_k` | int | scorer configuration |
| `requested_memory_fraction`, `effective_memory_fraction`, `memory_count` | number | memory audit |
| `memory_sha256`, `score_sha256` | string | payload hashes |
| `runtime_seconds`, `peak_vram_mib` | number | resource measurements |
| `sliding_window` | int | feature-only ACF result for VUS |
| `labels_read` | bool | always `false` in runner artifact |

### 8.3 `iteration_metrics.jsonl`

One row per iteration (100 rows) contains every frozen instrumentation field plus series, trajectory, seed, batch size, learning rate, total/final loss, gradient cosine (nullable), iteration runtime, allocated VRAM, and best-so-far iteration. Distance/margin fields are five-number dictionaries. Units are cosine distance, L2 gradient norm, seconds, and MiB.

### 8.4 Evaluator and aggregate schemas

`file_metrics.csv` columns are: `run_id,series_id,family,track,seed,trajectory,checkpoint,vus_pr,auprc,vus_roc,auroc,score_sha256,data_sha256,config_sha256,vendor_sha`.

`paired_contrasts.csv` columns are: `contrast,series_id,family,seed,treatment,control,delta_vus_pr,delta_auprc,delta_vus_roc`.

`training_summary.json` wraps the typed `TrainingSummary` fields with runner provenance `series_id`, `family`, `track`, `data_sha256`, `config_sha256`, and `vendor_sha` so aggregation never infers family identity from a path string.

`activity_summary.csv` columns are: `series_id,family,trajectory,seed,median_active_hinge_20_100,median_triplet_grad_norm_20_100,median_pretext_grad_norm_1_19,best_iteration,low_activity,early_checkpoint`.

`decision.json` records all gate inputs, pass booleans, missing count (must be zero), outcome, protocol/config/data/vendor hashes, timestamp, and explicit `method_frozen:false`.

## 9 Environment and Pre-Coding Checklist

`code/requirements.txt` will contain library names only for `numpy`, `pandas`, `scikit-learn`, `scipy`, `statsmodels`, `matplotlib`, `tqdm`, `PyYAML`, `psutil`, and `pytest`. Exact installed versions are recorded in the environment manifest instead. It must not contain `torch`, `torchvision`, or `torchaudio`.

Recommended setup commands for `code/README.md`:

```powershell
conda create -p D:\Anaconda\envs\paano_msn python=3.11 -y
D:\Anaconda\envs\paano_msn\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
D:\Anaconda\envs\paano_msn\python.exe -m pip install -r C:\Users\qintian\Desktop\msn\msnmsn\code\requirements.txt
D:\Anaconda\envs\paano_msn\python.exe -m pip install -e C:\Users\qintian\Desktop\msn\msnmsn\code
```

| Check | Required state before coding/run | Status at design time |
|---|---|---|
| Experiment design | Part 3 and YAML frozen | Ready |
| Vendor | path exists; SHA guard must pass | Ready, recheck in code |
| Data | six files and hashes in frozen manifest | Ready, recheck in code |
| Device | one RTX 4090; CUDA-visible | Verify in fresh environment |
| Conda | fresh `paano_msn`; failed PrefixCal/TSPulse env removed independently | Ready |
| PyTorch | 2.7.1 installed separately, CUDA smoke passes | Pending environment task |
| Requirements | torch family excluded | Enforced by test/review |
| Output storage | results/logs gitignored; adequate D/C free space | Verify before long run |
| Label isolation | runner API contains no label surface | Enforced by design/tests |
| Reproducibility | deterministic algorithms and replay hashes | Enforced by design/tests |
| Run strategy | one GPU job at a time; 15-minute read-only monitor | Ready |
| Baseline comparison | paper headline external; no full reproduction | Ready |

## 10 Implementation Order

```text
1. code/requirements.txt + pyproject.toml + README environment section
2. schemas.py + config.py + vendor.py
3. feature_data.py + label_data.py
4. replay.py and replay/label-isolation unit tests
5. objectives.py + instrumentation.py and numerical parity tests
6. trainer.py and BEST/LAST/RAND_BN tests
7. memory.py + scoring.py and vendor parity tests
8. artifacts.py + transaction tests
9. run_series.py, then one CPU fixture and one CUDA smoke
10. evaluate_scores.py + aggregate.py + synthetic gate tests
11. PowerShell scripts and coverage test
12. pytest full suite -> 00_smoke.ps1
13. primary six-family K0 -> evaluator -> frozen decision
14. conditional three-seed confirmation only after a passing outcome
```

After each file is implemented, `docs/dev_log.md` must record implementation, tests, issues, and exact command. A file is not marked done until its focused tests pass. Vendor, data, and scientific YAML remain untouched throughout.

## 11 Implementation-Plan Validation

### 11.1 Experiment coverage

| Part 3 requirement | Supporting implementation |
|---|---|
| OFFICIAL/PAPERNEG/PAPERNEG_NONOVERLAP trajectories | `objectives.py`, `trainer.py`, replay plan |
| BEST/LAST scoring | post-update checkpoint logic in `trainer.py`; checkpoint-specific `score_checkpoint` |
| RAND_BN | `replay_rand_bn` and buffer-only test |
| Activity/gradient/geometry diagnostics | `objectives.py`, `instrumentation.py`, activity aggregator |
| Same initialization and batch randomness | `build_initial_state`, `ReplayPlan`, hashes/tests |
| Identical memory/scorer | `memory.py`, `scoring.py`, numerical parity tests |
| Six-family threshold-free evaluation | frozen manifest, evaluator, coverage test |
| Label isolation | split feature/label modules and transactional evaluator boundary |
| Frozen gates/outcomes | `aggregate.py` plus boundary/precedence tests |
| Runtime/VRAM/hash reporting | trainer/scorer summaries and manifests |
| Conditional three-seed confirmation | `expand_confirmation_jobs`, `04_run_confirmation.ps1` |

**Experiment coverage: PASS.** Every frozen Part 3 experiment and endpoint has a named module, function, result field, and test.

### 11.2 Logic consistency

- Patch geometry is consistently `[N,C,96]`; encoder output is `[N,64]`; projection output is `[N,256]`.
- OFFICIAL and PAPERNEG share initialization, anchor batches, local positives, pretext pairs, optimizer, schedule, memory, and scorer; only the negative pool/space changes.
- PAPERNEG_NONOVERLAP changes only positive offsets relative to PAPERNEG and asserts zero overlap.
- BEST is the post-update state selected by pre-update minibatch loss; LAST is iteration 100; both build their own memory.
- RAND_BN performs the same encoder-forward BN exposure but cannot update trainable parameters.
- Scores are length `[T]` before labels are read; evaluator validates hashes and alignment first.
- Decision contrasts are fixed to LAST for execution factors, so BEST cannot become a post-hoc favorable selector.

**Logic consistency: PASS.** No tensor, checkpoint, RNG, label, or contrast mismatch remains in the design.

### 11.3 Completeness

Every file in the proposed tree has a responsibility in Section 2; every Python module has exact public signatures and logic in Section 4; scripts, tests, environment, artifacts, and run order are specified in Sections 6-10. Data, results, checkpoints, logs, and commit exclusions are explicit.

**Completeness: PASS.** No proposed implementation file lacks a specification, and no specification refers to an unlisted file.

### 11.4 Phase outcome

```text
PHASE_D_COMPLETE
IMPLEMENTATION_PLAN_VALIDATED
READY_FOR_PHASE_E_CODING
METHOD_NOT_FROZEN
```

## 12 Phase F Full-Benchmark Extension

### 12.1 Scope

The user-confirmed Phase F extension changes experiment coverage only. It reuses the frozen protocol, `run_job`, transactional score artifacts, evaluator-only label boundary, and vendor metrics. It must not modify the PaAno model, objective implementations, memory, scorer, or six-file K0 artifacts.

### 12.2 Additional files and functions

| File | Public surface | Responsibility |
|---|---|---|
| `code/src/paano_k0/benchmark_manifest.py` | `load_benchmark_manifest(path, expected_tracks)`, `build_manifest_from_eval_lists(...)` | Build and strictly verify the complete 350/180 manifest from fixed Eval filename lists and local official CSV roots; hash every file without reading labels into model code. |
| `code/src/paano_k0/run_benchmark_series.py` | `main(argv=None)` | Resolve exactly one manifest series, construct a registered `RunJob`, and call the already tested label-free `run_job`. |
| `code/src/paano_k0/confirmation_guard.py` | `validate_confirmation_authorization(...)`, `validate_existing_confirmation_run(...)`, `main(argv=None)` | Before confirmation compute, bind the positive decision to the current config/vendor and exact fixed seed gate; before every resume skip, hash-check the LAST score and validate its run identity, dataset/config/vendor provenance, trajectory summary, and trajectory marker without loading labels. |
| `code/src/paano_k0/evaluate_benchmark.py` | `evaluate_registered_benchmark(...)`, `main(argv=None)` | Verify every LAST score commit before evaluator-only label loading; require exact series/arm coverage. |
| `code/src/paano_k0/aggregate_benchmark.py` | `aggregate_full_benchmark(...)`, `main(argv=None)` | Produce file-, family-, track-, and overall metrics; compare the full arm to fixed paper-reported U/M values and write the terminal full-main decision. |
| `code/src/paano_k0/report_benchmark.py` | `render_full_benchmark_report(...)`, `main(argv=None)` | Consume only compact aggregate CSV/JSON outputs, validate their registered rows and external-reference provenance, and render the English manuscript-facing numeric report with only the registered VUS-PR, AUPRC, and VUS-ROC endpoints, without reopening labels, raw scores, or datasets. AUROC may remain in internal compatibility artifacts but cannot enter the manuscript-facing report. |
| `code/src/paano_k0/aggregate_confirmation.py` | `aggregate_confirmation(...)`, `main(argv=None)` | Conditional-only aggregation of the fixed full arm across seeds 2027/2028/2029; require exact 530-file LAST coverage per seed and emit U/M seed means and standard deviations without retuning or dropping results. |
| `code/scripts/05_run_full_main.ps1` | PowerShell entry point | Run all 530 `PAPERNEG_NONOVERLAP` seed-2027 jobs fail-fast, resuming only already validated `_SUCCESS` runs. |
| `code/scripts/06_run_full_ablations.ps1` | PowerShell entry point | Run `PAPERNEG` and `OFFICIAL` seed-2027 component ablations on the same 530 files. |
| `code/scripts/07_evaluate_full.ps1` | PowerShell entry point | Evaluate exact full coverage and aggregate the manuscript-facing numeric tables. |
| `code/scripts/08_finalize_full.ps1` | PowerShell entry point | Run the registered full evaluator/aggregator, render the numeric Markdown report, execute the complete unit suite, and verify the compact Git-facing result set; it performs no automatic method change or hidden result filtering. |
| `code/scripts/09_run_full_confirmation.ps1` | Conditional PowerShell entry point | Only when the frozen seed-2027 decision is `CONTINUE_FULL_CONFIRMATION` and its schema, exact seed list, both-track gate, config SHA, and vendor SHA match the current frozen state, run the same `PAPERNEG_NONOVERLAP-LAST` arm on all 530 files for seeds 2028 and 2029. Resume is allowed only after label-free score-hash and full-provenance validation. |
| `code/scripts/10_evaluate_confirmation.ps1` | Conditional PowerShell entry point | Require complete main-arm LAST coverage for all three registered seeds, evaluate seeds 2028/2029 after global score preflight, and aggregate the fixed three-seed U/M mean and standard deviation. |
| `code/scripts/monitor_full.ps1` | read-only monitor | Report runner state, exact completion/failure count, current file/arm, GPU, disk, available physical RAM, and log freshness at 15-minute intervals for the main arm, ablations, or conditional confirmation seeds. Progress is enumerated from the frozen 530-series manifest, so stale or extra run directories cannot inflate counts; confirmation mode includes only seeds 2028/2029. The monitor remains observational: a free-RAM reading below the user floor stops unrelated parallel work but never mutates or terminates the active experiment. |

### 12.3 Data and split provenance

The canonical Eval filename lists contain exactly 350 U and 180 M files. Only the filename columns are used; scores stored in external benchmark CSVs are ignored. U files resolve under the verified TSB-AD-U extraction. M files resolve under a fresh extraction of the SHA-256-verified `TSB-AD-M.zip`. The generated manifest records family, track, filename, SHA-256, byte count, row count, channel count, and filename-derived `train_end`.

Any missing, duplicate, shape-invalid, hash-mismatched, or short-prefix file is a hard failure. No backup series is allowed.

### 12.4 Coverage and outputs

Primary main coverage is exactly `530 x 1 trajectory x 1 seed = 530` jobs. The existing runner may commit BEST and LAST, but evaluation consumes only LAST. Ablation coverage is exactly `530 x 2 trajectories x 1 seed = 1060` jobs. Large results live under `D:/qintian_experiments/paano_full/`; compact tables and decisions are copied to `artifacts/paano_full/`.

Required compact outputs are:

```text
artifacts/paano_full/main_file_metrics.csv
artifacts/paano_full/main_family_metrics.csv
artifacts/paano_full/main_track_metrics.csv
artifacts/paano_full/ablation_track_metrics.csv
artifacts/paano_full/paper_reference_comparison.csv
artifacts/paano_full/runtime_summary.csv
artifacts/paano_full/decision.json
docs/experiments/PAANO_FULL_MAIN_RESULTS.md
```

The result report must contain the exact 350/180 coverage, all three registered
LAST arms, U/M VUS-PR/AUPRC/VUS-ROC values, the external Table 15 comparison,
runtime/VRAM context, the terminal decision, and the negative six-file K0 caveat.
It is generated only from the compact aggregate outputs and is not permitted to
drop a track, family, arm, or unfavorable result.

### 12.5 Implementation and execution order

```text
1. Verify/extract official data and build 350/180 hash manifest.
2. Implement generic benchmark manifest/runner while leaving K0 loaders unchanged.
3. Add tests for 530-file coverage, duplicate/missing rejection, LAST-only evaluation, and label ordering.
4. Run the full method on U as soon as its manifest is ready; M may extract in parallel.
5. Continue the full method on M, then run the two ablation arms.
6. Evaluate only after score hashes are committed; aggregate against fixed paper references.
7. Render the compact English numeric report, run the full unit suite, and verify the Git-facing outputs before committing them.
8. If both tracks exceed, run main-only seeds 2028/2029; otherwise stop and retain failure evidence.
9. After conditional confirmation, report every registered seed and the three-seed mean/standard deviation; no confirmation result may trigger another variant, seed replacement, or per-track selection.
10. While conditional confirmation is active, invoke `monitor_full.ps1 -Mode confirmation`; this is read-only and reports progress over the exact 1,060 registered seed-2028/2029 runs.
11. The legacy rounded K0 reference fields in `configs/k0_protocol.yaml` remain byte-frozen for active-run artifact compatibility and are not the full-benchmark gate authority. The exact full gate is independently fixed and tested as U `0.5296` and M `0.4263` in the full aggregator/report path; do not edit the active config after scores exist.
```

**Phase F design validation:** the extension changes only coverage and reporting. Model semantics, labels, hyperparameters, checkpoint endpoint, and external comparison values are fixed before full-run labels are evaluated.
