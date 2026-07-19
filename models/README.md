# Frozen model identity

ViTTrace adds no learned checkpoint. Its only neural component is the frozen
OpenCLIP `ViT-B-16` / `openai` backbone described in
`vittrace_vit_b16_openai.json`.

The 570.79 MiB upstream checkpoint is deliberately not stored in Git. Download
and verify the exact model state with:

```powershell
python code/scripts/fetch_vittrace_model.py --cache-dir D:/models/open_clip
```

The command succeeds only when the loaded state matches the registered
canonical SHA256. The explicit `--cache-dir` is passed through to OpenCLIP, so
the upstream checkpoint is downloaded to that location even when OpenCLIP was
already imported. Dataset files, token caches, and model weights remain local.
