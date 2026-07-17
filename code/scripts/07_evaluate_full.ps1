[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Python = 'D:\Anaconda\envs\paano_msn\python.exe'
$CodeRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $CodeRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot
$Config = Join-Path $RepoRoot 'configs\k0_protocol.yaml'
$Manifest = Join-Path $RepoRoot 'docs\TSB_AD_FULL_EVAL_MANIFEST.csv'
$VendorRoot = Join-Path $WorkspaceRoot 'vendor\PaAno'
$ResultsRoot = 'D:\qintian_experiments\paano_full'
$MetricsDirectory = Join-Path $ResultsRoot 'evaluation'
$OutputDirectory = Join-Path $RepoRoot 'artifacts\paano_full'
$LogRoot = Join-Path $RepoRoot 'logs\full_evaluation'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot, $ResultsRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

$rows = @(Import-Csv -LiteralPath $Manifest)
$uCount = @($rows | Where-Object track -eq 'U').Count
$mCount = @($rows | Where-Object track -eq 'M').Count
if ($rows.Count -ne 530 -or $uCount -ne 350 -or $mCount -ne 180) {
    throw "Full manifest coverage changed: total=$($rows.Count), U=$uCount, M=$mCount"
}

$trajectories = @('PAPERNEG_NONOVERLAP', 'PAPERNEG', 'OFFICIAL')
$seed = 2027
$trajectoryCount = 0
$lastScoreCount = 0
foreach ($row in $rows) {
    $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
    foreach ($trajectory in $trajectories) {
        $runDirectory = Join-Path $ResultsRoot ("runs\$seriesId\seed_$seed\$trajectory")
        if (Test-Path -LiteralPath (Join-Path $runDirectory '_FAILED.json') -PathType Leaf) {
            throw "Failure record present; full evaluation is forbidden: $runDirectory"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $runDirectory '_SUCCESS') -PathType Leaf)) {
            throw "Missing trajectory success marker: $runDirectory"
        }
        $trajectoryCount++
        $lastDirectory = Join-Path $runDirectory 'scores\LAST'
        if (-not (Test-Path -LiteralPath (Join-Path $lastDirectory '_SUCCESS') -PathType Leaf)) {
            throw "Missing frozen LAST score success marker: $lastDirectory"
        }
        $lastScoreCount++
    }
}
if ($trajectoryCount -ne 1590 -or $lastScoreCount -ne 1590) {
    throw "Full coverage mismatch: trajectories=$trajectoryCount LAST_scores=$lastScoreCount"
}

New-Item -ItemType Directory -Path $MetricsDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$evaluateLog = Join-Path $LogRoot "evaluate_$stamp.log"
& $Python @(
    '-m', 'paano_k0.evaluate_benchmark',
    '--config', $Config,
    '--manifest', $Manifest,
    '--vendor-root', $VendorRoot,
    '--results-root', $ResultsRoot,
    '--output-dir', $MetricsDirectory,
    '--seed', [string]$seed,
    '--trajectories', 'PAPERNEG_NONOVERLAP', 'PAPERNEG', 'OFFICIAL',
    '--checkpoint', 'LAST',
    # Windows process spawning reloads the artifact module and the full
    # PyTorch DLL set in every worker.  On this registered host that exhausts
    # page-file commit before metric computation (WinError 1455).  Exact-VUS
    # makes serial evaluation fast while preserving every scientific input.
    '--workers', '1',
    '--resume-existing'
) 2>&1 | Tee-Object -FilePath $evaluateLog
if ($LASTEXITCODE -ne 0) {
    throw "Full benchmark evaluator failed with exit code $LASTEXITCODE. Log: $evaluateLog"
}

$aggregateLog = Join-Path $LogRoot "aggregate_$stamp.log"
& $Python @(
    '-m', 'paano_k0.aggregate_benchmark',
    '--config', $Config,
    '--manifest', $Manifest,
    '--metrics-dir', $MetricsDirectory,
    '--results-root', $ResultsRoot,
    '--output-dir', $OutputDirectory,
    '--seed', [string]$seed
) 2>&1 | Tee-Object -FilePath $aggregateLog
if ($LASTEXITCODE -ne 0) {
    throw "Full benchmark aggregation failed with exit code $LASTEXITCODE. Log: $aggregateLog"
}

$requiredOutputs = @(
    'main_file_metrics.csv',
    'main_family_metrics.csv',
    'main_track_metrics.csv',
    'ablation_track_metrics.csv',
    'paper_reference_comparison.csv',
    'runtime_summary.csv',
    'decision.json'
)
foreach ($name in $requiredOutputs) {
    $path = Join-Path $OutputDirectory $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Full benchmark output is missing: $path"
    }
}

$decisionPath = Join-Path $OutputDirectory 'decision.json'
$decision = Get-Content -LiteralPath $decisionPath -Raw | ConvertFrom-Json
$allowed = @('CONTINUE_FULL_CONFIRMATION', 'STOP_FULL_MAIN_FAILURE')
if ([string]$decision.outcome -notin $allowed) {
    throw "Full benchmark aggregator returned an unregistered outcome: $($decision.outcome)"
}
Write-Output (
    "FULL_BENCHMARK_DECISION outcome={0} evaluated=1590 checkpoint=LAST decision={1}" -f
    $decision.outcome, $decisionPath
)
