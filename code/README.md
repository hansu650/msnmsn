# Index-Consistent Harmonic Projection (IHP)

This package contains the final parameter-free projection correction for the
frozen VLM4TS/ViT4TS screening stage. IHP interprets released multiscale masks
as literal zero-based base-cell indices and applies a validity-normalized
harmonic projection. It uses no labels, learned weights, thresholds, or extra
backbone passes.

The paper-facing evidence covers 492 series and all 11 Table-1 subdatasets.
The external ViT4TS values are copied from the AAAI 2026 paper and are always
marked paper-reported; `REL_U` is the same-cache component-removal control.

## Environment

CUDA PyTorch is installed separately so that the requirements file never
silently replaces it with a CPU wheel.

```powershell
python -m pip install -r .\code\requirements.txt
python -m pip install --no-deps TSB_AD==1.5
python -m pip install -e .\code
```

## Minimal API

```python
from measure_vit4ts.ihp import index_consistent_harmonic_projection

score_map = index_consistent_harmonic_projection(
    large_scores,
    mid_scores,
    patch_scores,
    large_zero_based_mask,
    mid_zero_based_mask,
)
```

The module accepts only frozen token costs and zero-based pooling masks. It
does not accept anomaly labels.

Apply the same method to a frozen ViT4TS token cache and emit base-grid maps
plus the 224-column window scores used by the unchanged timestamp stitcher:

```powershell
python .\code\scripts\ihp_score_cache.py `
  --token-cache <clip_tokens.npz> `
  --output <ihp_scores.npz> `
  --device cuda
```

Run the public unit tests:

```powershell
python -m pytest `
  .\code\tests\test_ihp.py -q
```

Recreate the label-free structural certificate from any frozen ViT4TS token
cache:

```powershell
python .\code\scripts\ihp_structure_audit.py `
  --token-cache <clip_tokens.npz> `
  --output .\artifacts\ihp\ihp_structure_certificate.json
```

Generate the final figures from the frozen compact result package:

```powershell
python .\code\scripts\ihp_make_figures.py `
  --artifacts .\artifacts\ihp `
  --output .\docs\manuscripts\msn2026\figures
```

Large tokens, anomaly maps, scores, datasets, checkpoints, and logs remain
local. Only compact source, evidence tables, manuscript files, and checked
figures belong in Git.

## Full v3 experiment code

Download and verify the frozen external backbone (no project-trained weights
exist):

```powershell
python .\code\scripts\fetch_vittrace_model.py
```

The registered full experiment uses `configs/vittrace_ablation_full_v3.yaml`,
`configs/vlm4ts_11group_manifest.json`, and the frozen registry/plan/parity
records under `artifacts/vittrace_release`. Paths in the YAML intentionally
record the original run and must be changed to local dataset, vendor, cache,
and output roots before a new run. The vendor is
`https://github.com/ZLHe0/VLM4TS.git` at commit
`8ab8c16414eb2c1a861dfc3e76f458180035a879`.

Run the complete public v3 regression without downloading data:

```powershell
python -m pytest .\code\tests\test_vittrace_v3*.py -q
```
