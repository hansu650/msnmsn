# PaAno Route Environment Status

Last updated: 2026-07-16T13:18:00+08:00

## Scope

This manifest records the isolated Conda environment prepared for the PaAno-based MSN 2026 route and the explicitly authorized cleanup of the closed PrefixCal/TSPulse route.

## Closed-route cleanup

- Removed environment: `prefixcal`
- Removed path: `D:\Anaconda\envs\prefixcal`
- Size before removal: 8,393,245,974 bytes (7.817 GiB)
- Files before removal: 45,911
- Safety check: no running `python`, `pythonw`, `conda`, or `pip` process referenced the environment path before removal.
- Removal command: `D:\Anaconda\Scripts\conda.exe env remove -n prefixcal -y`
- Removal verification: the path no longer exists and `conda env list` no longer contains `prefixcal`.
- No separately named TSPulse environment was present, so no other environment was removed.
- Environments unrelated to this closed route were preserved, including `timercd-k0` and all pre-existing general/RGB-D environments.

## New active research environment

- Conda name: `paano_msn`
- Prefix: `D:\Anaconda\envs\paano_msn`
- Python: 3.11.15
- Python executable: `D:\Anaconda\envs\paano_msn\python.exe`
- Initial environment size: 168,064,646 bytes (0.157 GiB)
- PaAno source commit: `d4c67116190efa4592dc6a8a157ced0def68b6af`
- Compatibility basis: the official PaAno README specifies Python 3.11+ and its installation example creates a Python 3.11 Conda environment.

The implementation environment is complete. PyTorch was installed from the CUDA 12.8 wheel index and the official scientific stack was installed at the following verified versions:

```text
matplotlib==3.10.5
numpy==2.3.2
pandas==2.3.1
scikit-learn==1.7.1
scipy==1.16.1
statsmodels==0.14.5
torch==2.7.1
tqdm==4.66.5
PyYAML==6.0.3
psutil==7.2.2
pytest==9.1.1
```

Runtime verification: `torch==2.7.1+cu128`, `torch.version.cuda==12.8`, `torch.cuda.is_available()==True`, device `NVIDIA GeForce RTX 4090`, CUDA tensor arithmetic passed, and `pip check` reported no broken requirements.

## Base package inventory

```text
bzip2 1.0.8
ca-certificates 2026.5.14
libexpat 2.8.2
libffi 3.4.8
libzlib 1.3.2
openssl 3.5.7
packaging 26.0
pip 26.1.2
python 3.11.15
setuptools 83.0.0
sqlite 3.53.2
tk 8.6.15
tzdata 2026c
ucrt 10.0.22621.0
vc 14.3
vc14_runtime 14.44.35208
vs2015_runtime 14.44.35208
wheel 0.47.0
xz 5.8.2
zlib 1.3.2
```

## Disk-space effect

- D: free before cleanup: 834.395 GiB
- D: free after cleanup and new environment creation: 842.131 GiB
- Observed net free-space increase: 7.736 GiB
- Environment-only estimate: 7.817 GiB removed minus 0.157 GiB created = 7.660 GiB net recovered. The small difference from the drive-level measurement reflects concurrent filesystem activity.
- C: free after completion: 97.737 GiB

## Activation

```powershell
& 'D:\Anaconda\Scripts\activate.ps1' paano_msn
```

For non-interactive runs, prefer the absolute interpreter path:

```powershell
& 'D:\Anaconda\envs\paano_msn\python.exe' --version
```
