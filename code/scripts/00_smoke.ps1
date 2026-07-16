[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Python = 'D:\Anaconda\envs\paano_msn\python.exe'
$CodeRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $CodeRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$Config = Join-Path $RepoRoot 'configs\k0_protocol.yaml'
$Manifest = Join-Path $RepoRoot 'docs\K0_DATA_MANIFEST.csv'
$VendorRoot = Join-Path $WorkspaceRoot 'vendor\PaAno'
$OutputRoot = Join-Path $RepoRoot 'results\smoke'
$LogRoot = Join-Path $RepoRoot 'logs\smoke'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

# The smoke owns this directory exclusively; no primary result is touched.
if (Test-Path -LiteralPath $OutputRoot) {
    Remove-Item -LiteralPath $OutputRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

function Invoke-LoggedPython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE. Log: $LogPath"
    }
}

$rows = @(Import-Csv -LiteralPath $Manifest)
if ($rows.Count -ne 6) {
    throw "Frozen manifest must contain exactly six rows; found $($rows.Count)"
}
$nab = $rows[0]
if ($nab.family -ne 'NAB' -or $nab.track -ne 'U') {
    throw 'The first frozen manifest row must be NAB-U.'
}
$seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$nab.file)
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

Invoke-LoggedPython -Arguments @('-m', 'pytest', '-q') `
    -LogPath (Join-Path $LogRoot "pytest_$stamp.log")

foreach ($replicate in @('run_a', 'run_b')) {
    $replicateRoot = Join-Path $OutputRoot $replicate
    $logPath = Join-Path $LogRoot ("{0}_{1}_{2}.log" -f $replicate, $seriesId, $stamp)
    Invoke-LoggedPython -Arguments @(
        '-m', 'paano_k0.run_series',
        '--config', $Config,
        '--manifest', $Manifest,
        '--series-id', $seriesId,
        '--trajectory', 'OFFICIAL',
        '--seed', '2027',
        '--vendor-root', $VendorRoot,
        '--output-root', $replicateRoot,
        '--device', 'cuda'
    ) -LogPath $logPath
}

$verification = @'
import json
import sys
from pathlib import Path

from paano_k0.artifacts import verify_committed_score
from paano_k0.config import load_protocol, load_series_manifest
from paano_k0.evaluate_scores import evaluate_score_artifact
from paano_k0.vendor import load_vendor_symbols

config_path, manifest_path, vendor_root, smoke_root, series_id = map(Path, sys.argv[1:])
protocol = load_protocol(config_path)
specs = [spec for spec in load_series_manifest(manifest_path) if spec.series_id == str(series_id)]
if len(specs) != 1:
    raise RuntimeError(f"smoke series lookup returned {len(specs)} entries")
spec = specs[0]
vendor = load_vendor_symbols(vendor_root, protocol.baseline.git_sha)
snapshots = []
for replicate in ("run_a", "run_b"):
    run_dir = smoke_root / replicate / "runs" / spec.series_id / "seed_2027" / "OFFICIAL"
    rows = run_dir.joinpath("iteration_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if len(rows) != 100:
        raise RuntimeError(f"{replicate} has {len(rows)} instrumentation rows, expected 100")
    summary = json.loads(run_dir.joinpath("training_summary.json").read_text(encoding="utf-8"))
    current = {
        "initial": summary["initial_state_sha256"],
        "replay": summary["replay_sha256"],
        "checkpoint": summary["checkpoint_sha256"],
        "scores": {},
    }
    for checkpoint in ("BEST", "LAST"):
        score_dir = run_dir / "scores" / checkpoint
        scores, manifest = verify_committed_score(score_dir)
        if scores.shape != (spec.rows,):
            raise RuntimeError(f"{replicate}/{checkpoint} score length mismatch")
        if manifest.peak_vram_mib <= 0:
            raise RuntimeError(f"{replicate}/{checkpoint} did not record CUDA VRAM use")
        evaluate_score_artifact(score_dir, spec, vendor)
        current["scores"][checkpoint] = manifest.score_sha256
    snapshots.append(current)
if snapshots[0] != snapshots[1]:
    raise RuntimeError("independent OFFICIAL smoke runs are not hash-identical")
print("SMOKE_PASS series=NAB trajectory=OFFICIAL iterations=100 checkpoints=2 replicates=2")
'@

Invoke-LoggedPython -Arguments @(
    '-c', $verification, $Config, $Manifest, $VendorRoot, $OutputRoot, $seriesId
) -LogPath (Join-Path $LogRoot "verify_$stamp.log")

