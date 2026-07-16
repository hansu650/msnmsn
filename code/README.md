# PaAno K0 Implementation

This overlay keeps the PaAno encoder and scorer semantics fixed while running matched execution-fidelity, checkpoint, objective-activity, and overlap controls. It does not define a final method.

## Environment

PyTorch is installed separately because the CUDA wheel is platform-specific:

```powershell
D:\Anaconda\envs\paano_msn\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
D:\Anaconda\envs\paano_msn\python.exe -m pip install -r C:\Users\qintian\Desktop\msn\msnmsn\code\requirements.txt
D:\Anaconda\envs\paano_msn\python.exe -m pip install -e C:\Users\qintian\Desktop\msn\msnmsn\code
```

The CUDA smoke was repeated from independent output directories with identical initialization, replay, BEST/LAST checkpoint, memory-bank, and score hashes. The primary K0 uses the frozen six-series manifest and seed 2027.

## Run the frozen K0

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn
powershell -ExecutionPolicy Bypass -File .\code\scripts\01_run_primary_k0.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\02_evaluate_primary_k0.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\03_aggregate_decision.ps1
```

The runner never reads labels. The evaluator first verifies committed score hashes, then loads labels to compute metrics. PaAno paper values are external headline references; they are not represented as matched-file reproductions.

## Verify

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn\code
D:\Anaconda\envs\paano_msn\python.exe -m pytest -q
```

Expected implementation result: `28 passed`.
