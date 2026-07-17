# PaAno Full-Benchmark Main Results

> Numeric report rendered exclusively from the seven registered compact aggregate outputs; no raw score, label, or dataset file was reopened.
> Report code Git SHA: `319446f28834868d36b2353532385845d76bf5de`

## Protocol and complete coverage

All results use the frozen seed 2027 endpoint, file-weighted aggregation, and the `LAST` checkpoint. The same complete Eval lists are used for every registered arm.

| Track | Eval series | Main-arm families |
|---|---:|---:|
| TSB-AD-U | 350 | 23 |
| TSB-AD-M | 180 | 17 |
| **Total** | **530** | **40** |

The registered arms are `PAPERNEG_NONOVERLAP-LAST` (full arm), `PAPERNEG-LAST` (remove non-overlap positives), and `OFFICIAL-LAST` (remove both registered execution changes). No arm, track, family, or result is selected after evaluation.

## Main results

| Arm | Track | Files | VUS-PR | AUPRC | VUS-ROC |
|---|---|---:|---:|---:|---:|
| `PAPERNEG_NONOVERLAP-LAST` | U | 350 | 0.519191 | 0.455312 | 0.882412 |
| `PAPERNEG_NONOVERLAP-LAST` | M | 180 | 0.410214 | 0.362830 | 0.779854 |

### Complete main-arm family results

| Track | Family | Files | VUS-PR | AUPRC | VUS-ROC |
|---|---|---:|---:|---:|---:|
| U | CATSv2 | 1 | 0.321767 | 0.490381 | 0.754152 |
| U | Daphnet | 1 | 0.460816 | 0.492015 | 0.932702 |
| U | Exathlon | 30 | 0.823245 | 0.812400 | 0.968146 |
| U | IOPS | 15 | 0.344315 | 0.280415 | 0.889901 |
| U | LTDB | 8 | 0.643947 | 0.562222 | 0.796125 |
| U | MGAB | 8 | 0.281887 | 0.272933 | 0.966801 |
| U | MITDB | 7 | 0.456962 | 0.408041 | 0.909170 |
| U | MSL | 7 | 0.244086 | 0.214944 | 0.706155 |
| U | NAB | 23 | 0.481629 | 0.460249 | 0.789136 |
| U | NEK | 8 | 0.548146 | 0.585730 | 0.727305 |
| U | OPPORTUNITY | 27 | 0.156675 | 0.149592 | 0.618562 |
| U | Power | 1 | 0.159199 | 0.148562 | 0.656296 |
| U | SED | 2 | 0.954701 | 0.817956 | 0.998517 |
| U | SMAP | 17 | 0.779445 | 0.777710 | 0.919582 |
| U | SMD | 33 | 0.429173 | 0.390919 | 0.903141 |
| U | SVDB | 18 | 0.696352 | 0.632811 | 0.981894 |
| U | SWaT | 1 | 0.096879 | 0.096614 | 0.182208 |
| U | Stock | 8 | 0.723322 | 0.082176 | 0.844609 |
| U | TAO | 2 | 0.876680 | 0.112903 | 0.934177 |
| U | TODS | 13 | 0.727190 | 0.319980 | 0.869443 |
| U | UCR | 70 | 0.441373 | 0.451123 | 0.933584 |
| U | WSD | 20 | 0.527928 | 0.459028 | 0.943521 |
| U | YAHOO | 30 | 0.616809 | 0.475022 | 0.952948 |
| M | CATSv2 | 5 | 0.068489 | 0.071534 | 0.688032 |
| M | CreditCard | 1 | 0.014325 | 0.001305 | 0.381754 |
| M | Daphnet | 1 | 0.246185 | 0.230736 | 0.876590 |
| M | Exathlon | 25 | 0.786337 | 0.753118 | 0.956847 |
| M | GECCO | 1 | 0.129097 | 0.178312 | 0.833233 |
| M | GHL | 23 | 0.008308 | 0.007243 | 0.336343 |
| M | Genesis | 1 | 0.561298 | 0.549547 | 0.992317 |
| M | LTDB | 4 | 0.566238 | 0.539651 | 0.801267 |
| M | MITDB | 11 | 0.398092 | 0.461080 | 0.884320 |
| M | MSL | 14 | 0.200649 | 0.181795 | 0.743814 |
| M | OPPORTUNITY | 7 | 0.120095 | 0.103842 | 0.576097 |
| M | PSM | 1 | 0.195596 | 0.185247 | 0.623019 |
| M | SMAP | 25 | 0.496041 | 0.475467 | 0.884255 |
| M | SMD | 20 | 0.279160 | 0.296189 | 0.805857 |
| M | SVDB | 28 | 0.539741 | 0.522196 | 0.891795 |
| M | SWaT | 2 | 0.419708 | 0.446012 | 0.785085 |
| M | TAO | 11 | 0.751808 | 0.089604 | 0.856991 |

## Component-removal ablations

These are matched seed-2027 controls on the same files and frozen `LAST` endpoint; they are not the paper's external ablation rows.

| Registered arm | Track | Files | VUS-PR | AUPRC | VUS-ROC |
|---|---|---:|---:|---:|---:|
| `PAPERNEG_NONOVERLAP-LAST` | U | 350 | 0.519191 | 0.455312 | 0.882412 |
| `PAPERNEG_NONOVERLAP-LAST` | M | 180 | 0.410214 | 0.362830 | 0.779854 |
| `PAPERNEG-LAST` | U | 350 | 0.526176 | 0.462269 | 0.883637 |
| `PAPERNEG-LAST` | M | 180 | 0.406401 | 0.360328 | 0.779956 |
| `OFFICIAL-LAST` | U | 350 | 0.527147 | 0.462868 | 0.883174 |
| `OFFICIAL-LAST` | M | 180 | 0.409121 | 0.362301 | 0.778763 |

## External paper-reported comparison

The fixed external reference is **PaAno Table 15 default full-Eval (k=3)**. These values are paper-reported ten-seed results, not a local reproduction or a paired same-seed baseline.

| Track | Our full arm VUS-PR | PaAno paper-reported VUS-PR | Delta | Strictly exceeds |
|---|---:|---:|---:|---|
| TSB-AD-U | 0.519191 | 0.5296 | -0.010409 | No |
| TSB-AD-M | 0.410214 | 0.4263 | -0.016086 | No |

## Runtime and peak VRAM

Runtime is reported separately from protocol alignment. Totals are sums over files; means are per file; peak VRAM is the maximum observed within each registered arm/track group.

| Arm | Track | Train total (s) | Train mean (s) | Score total (s) | Score mean (s) | Train peak (MiB) | Score peak (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `PAPERNEG_NONOVERLAP-LAST` | U | 1068.420 | 3.053 | 638.187 | 1.823 | 1165.275 | 198.505 |
| `PAPERNEG_NONOVERLAP-LAST` | M | 582.008 | 3.233 | 814.333 | 4.524 | 1589.054 | 237.605 |
| `PAPERNEG-LAST` | U | 1073.172 | 3.066 | 633.006 | 1.809 | 1165.275 | 198.505 |
| `PAPERNEG-LAST` | M | 615.035 | 3.417 | 853.921 | 4.744 | 1589.054 | 237.605 |
| `OFFICIAL-LAST` | U | 1060.948 | 3.031 | 685.863 | 1.960 | 1164.146 | 198.505 |
| `OFFICIAL-LAST` | M | 607.827 | 3.377 | 903.367 | 5.019 | 1587.925 | 237.605 |

## Frozen decision

**Outcome: `STOP_FULL_MAIN_FAILURE`.**

The full arm does not strictly exceed the fixed paper-reported VUS-PR reference on TSB-AD-U, TSB-AD-M. The frozen protocol stops without confirmation seeds or a post-hoc variant.

## Six-file K0 negative caveat

The earlier six-file same-code K0 established objective inactivity and early checkpointing, but the registered execution changes did not pass its matched performance gate. This full-coverage external comparison does not erase that negative result and, by itself, cannot establish the proposed causal mechanism.

## Compact provenance

- Frozen config SHA-256: `8414885694d7a346ca9b70251213b017791ade6611d04fb9e7f39ef6d238e824`
- Frozen PaAno vendor commit: `d4c67116190efa4592dc6a8a157ced0def68b6af`
- Evaluated metric rows: `1590` (530 series x 3 arms x 1 seed)
- Inputs: `main_file_metrics.csv`, `main_family_metrics.csv`, `main_track_metrics.csv`, `ablation_track_metrics.csv`, `paper_reference_comparison.csv`, `runtime_summary.csv`, and `decision.json`
