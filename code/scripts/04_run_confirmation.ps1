[CmdletBinding()]
param(
    [ValidateSet('cuda')][string]$Device = 'cuda'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Python = 'D:\Anaconda\envs\paano_msn\python.exe'
$CodeRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $CodeRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$Config = Join-Path $RepoRoot 'configs\k0_protocol.yaml'
$Manifest = Join-Path $RepoRoot 'docs\K0_DATA_MANIFEST.csv'
$VendorRoot = Join-Path $WorkspaceRoot 'vendor\PaAno'
$ResultsRoot = Join-Path $RepoRoot 'results\k0'
$DecisionPath = Join-Path $ResultsRoot 'aggregate\decision.json'
$LogRoot = Join-Path $RepoRoot 'logs\confirmation'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot, $DecisionPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$decision = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
$winnerByOutcome = @{
    SIMPLE_CHECKPOINT_FIX = 'OFFICIAL'
    SIMPLE_PAPER_PARITY_FIX = 'PAPERNEG'
    GO_METHOD_DESIGN = 'PAPERNEG_NONOVERLAP'
}
$outcome = [string]$decision.outcome
if (-not $winnerByOutcome.ContainsKey($outcome)) {
    throw "Confirmation is forbidden for terminal outcome: $outcome"
}
$winner = $winnerByOutcome[$outcome]
$trajectories = if ($winner -eq 'OFFICIAL') { @('OFFICIAL') } else { @('OFFICIAL', $winner) }
$rows = @(Import-Csv -LiteralPath $Manifest)
if ($rows.Count -ne 6) {
    throw "Frozen manifest must contain exactly six rows; found $($rows.Count)"
}

$jobIndex = 0
$totalJobs = $rows.Count * 2 * $trajectories.Count
foreach ($row in $rows) {
    $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
    foreach ($seed in @(2028, 2029)) {
        foreach ($trajectory in $trajectories) {
            $jobIndex++
            $runDirectory = Join-Path $ResultsRoot ("runs\$seriesId\seed_$seed\$trajectory")
            $failurePath = Join-Path $runDirectory '_FAILED.json'
            if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
                throw "Retained failure record blocks silent confirmation rerun: $failurePath"
            }
            if (Test-Path -LiteralPath (Join-Path $runDirectory '_SUCCESS') -PathType Leaf) {
                Write-Output ("SKIP_VALID_CONFIRMATION {0}/{1} series={2} seed={3} trajectory={4}" -f $jobIndex, $totalJobs, $seriesId, $seed, $trajectory)
                continue
            }
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            $logPath = Join-Path $LogRoot ("{0:D2}_{1}_seed{2}_{3}_{4}.log" -f $jobIndex, $seriesId, $seed, $trajectory, $stamp)
            Write-Output ("RUN_CONFIRMATION {0}/{1} series={2} seed={3} trajectory={4}" -f $jobIndex, $totalJobs, $seriesId, $seed, $trajectory)
            & $Python @(
                '-m', 'paano_k0.run_series',
                '--config', $Config,
                '--manifest', $Manifest,
                '--series-id', $seriesId,
                '--trajectory', $trajectory,
                '--seed', [string]$seed,
                '--vendor-root', $VendorRoot,
                '--output-root', $ResultsRoot,
                '--device', $Device
            ) 2>&1 | Tee-Object -FilePath $logPath
            if ($LASTEXITCODE -ne 0) {
                throw "Confirmation job failed with exit code $LASTEXITCODE. Log: $logPath"
            }
            if (-not (Test-Path -LiteralPath (Join-Path $runDirectory '_SUCCESS') -PathType Leaf)) {
                throw "Confirmation runner did not commit trajectory success: $runDirectory"
            }
        }
    }
}

$evaluateConfirmation = @'
import sys
from pathlib import Path

from paano_k0.config import expand_confirmation_jobs, load_protocol, load_series_manifest
from paano_k0.evaluate_scores import evaluate_score_artifact, score_directory
from paano_k0.schemas import scored_checkpoints
from paano_k0.vendor import load_vendor_symbols

config_path, manifest_path, vendor_root, results_root, decision_path = map(Path, sys.argv[1:])
config = load_protocol(config_path)
series = load_series_manifest(manifest_path)
vendor = load_vendor_symbols(vendor_root, config.baseline.git_sha)
jobs = expand_confirmation_jobs(config, decision_path, series, vendor_root, results_root)
evaluated = 0
for job in jobs:
    for checkpoint in scored_checkpoints(job.trajectory):
        evaluate_score_artifact(score_directory(job, checkpoint), job.series, vendor)
        evaluated += 1
expected = sum(len(scored_checkpoints(job.trajectory)) for job in jobs)
if evaluated != expected:
    raise RuntimeError(f"confirmation coverage mismatch: evaluated={evaluated}, expected={expected}")
print(f"CONFIRMATION_EVALUATION_COMPLETE metrics={evaluated}")
'@

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$evaluationLog = Join-Path $LogRoot "evaluate_$stamp.log"
& $Python -c $evaluateConfirmation $Config $Manifest $VendorRoot $ResultsRoot $DecisionPath 2>&1 |
    Tee-Object -FilePath $evaluationLog
if ($LASTEXITCODE -ne 0) {
    throw "Confirmation evaluation failed with exit code $LASTEXITCODE. Log: $evaluationLog"
}
Write-Output ("CONFIRMATION_RUNS_COMPLETE outcome={0} winner={1} trajectories={2}" -f $outcome, $winner, $totalJobs)

