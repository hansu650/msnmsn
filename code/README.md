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

## Run the frozen full benchmark

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn
powershell -ExecutionPolicy Bypass -File .\code\scripts\05_run_full_main.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\06_run_full_ablations.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\08_finalize_full.ps1
```

Monitor the active stage without changing any scientific state:

```powershell
powershell -ExecutionPolicy Bypass -File .\code\scripts\monitor_full.ps1 -RunnerPid <PID> -Mode main
powershell -ExecutionPolicy Bypass -File .\code\scripts\monitor_full.ps1 -RunnerPid <PID> -Mode ablations
```

Finalization calls `07_evaluate_full.ps1`, which requires all 1,590 registered
trajectory and LAST-score commits before evaluator-only label loading. It then
requires the seven compact aggregate artifacts, renders
`docs/experiments/PAANO_FULL_MAIN_RESULTS.md`, runs the complete pytest suite,
and verifies that all eight result files are nonempty and not ignored by Git.
The script does not commit or push changes.

If and only if finalization emits `CONTINUE_FULL_CONFIRMATION`, run the same
frozen main arm for seeds 2028 and 2029 and then aggregate all three seeds:

```powershell
powershell -ExecutionPolicy Bypass -File .\code\scripts\09_run_full_confirmation.ps1
powershell -ExecutionPolicy Bypass -File .\code\scripts\monitor_full.ps1 -RunnerPid <PID> -Mode confirmation
powershell -ExecutionPolicy Bypass -File .\code\scripts\10_evaluate_confirmation.ps1
```

Confirmation monitoring counts only the 1,060 seed-2028/2029 runs. Seed 2027
is excluded from its progress count, while all three seeds remain required by
the final confirmation evaluator.

## Verify

```powershell
Set-Location C:\Users\qintian\Desktop\msn\msnmsn\code
D:\Anaconda\envs\paano_msn\python.exe -m pytest -q
```

Expected implementation result: the complete suite passes with no failures.
