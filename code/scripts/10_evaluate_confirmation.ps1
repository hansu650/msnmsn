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
$OutputDirectory = Join-Path $RepoRoot 'artifacts\paano_full'
$DecisionPath = Join-Path $OutputDirectory 'decision.json'
$Seed2027Metrics = Join-Path $OutputDirectory 'main_file_metrics.csv'
$ConfirmationEvaluationRoot = Join-Path $ResultsRoot 'confirmation_evaluation'
$LogRoot = Join-Path $RepoRoot 'logs\full_confirmation_evaluation'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @(
    $Python, $Config, $Manifest, $VendorRoot, $ResultsRoot,
    $DecisionPath, $Seed2027Metrics
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

$decision = Get-Content -LiteralPath $DecisionPath -Raw | ConvertFrom-Json
$requiredDecisionFields = @(
    'outcome', 'main_trajectory', 'checkpoint', 'seed', 'series_count', 'metric_count'
)
foreach ($field in $requiredDecisionFields) {
    if ($field -notin @($decision.PSObject.Properties.Name)) {
        throw "Full-main decision is missing frozen field: $field"
    }
}
if ([string]$decision.outcome -ne 'CONTINUE_FULL_CONFIRMATION') {
    throw "Full confirmation evaluation is forbidden for outcome: $($decision.outcome)"
}
if (
    [string]$decision.main_trajectory -ne 'PAPERNEG_NONOVERLAP' -or
    [string]$decision.checkpoint -ne 'LAST' -or
    [int]$decision.seed -ne 2027 -or
    [int]$decision.series_count -ne 530 -or
    [int]$decision.metric_count -ne 1590
) {
    throw "Full-main decision provenance does not authorize confirmation evaluation"
}

$rows = @(Import-Csv -LiteralPath $Manifest)
$uCount = @($rows | Where-Object track -eq 'U').Count
$mCount = @($rows | Where-Object track -eq 'M').Count
$seriesIds = @(
    $rows | ForEach-Object {
        [IO.Path]::GetFileNameWithoutExtension([string]$_.file)
    }
)
$uniqueSeriesCount = @($seriesIds | Sort-Object -Unique).Count
if (
    $rows.Count -ne 530 -or
    $uCount -ne 350 -or
    $mCount -ne 180 -or
    $uniqueSeriesCount -ne 530
) {
    throw (
        "Full manifest coverage changed: total={0}, U={1}, M={2}, unique={3}" -f
        $rows.Count, $uCount, $mCount, $uniqueSeriesCount
    )
}

$trajectory = 'PAPERNEG_NONOVERLAP'
$seeds = @(2027, 2028, 2029)
$lastCoverage = 0
foreach ($seed in $seeds) {
    foreach ($row in $rows) {
        $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
        $runDirectory = Join-Path $ResultsRoot (
            "runs\$seriesId\seed_$seed\$trajectory"
        )
        if (Test-Path -LiteralPath (Join-Path $runDirectory '_FAILED.json') -PathType Leaf) {
            throw "Failure record present; confirmation evaluation is forbidden: $runDirectory"
        }
        $requiredArtifacts = @(
            (Join-Path $runDirectory '_SUCCESS'),
            (Join-Path $runDirectory 'scores\LAST\_SUCCESS'),
            (Join-Path $runDirectory 'scores\LAST\scores.npy'),
            (Join-Path $runDirectory 'scores\LAST\score_manifest.json')
        )
        foreach ($artifact in $requiredArtifacts) {
            if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
                throw "Missing frozen LAST artifact: $artifact"
            }
        }
        $lastCoverage++
    }
}
if ($lastCoverage -ne 1590) {
    throw "Full confirmation LAST coverage mismatch: $lastCoverage != 1590"
}

$seed2027Rows = @(Import-Csv -LiteralPath $Seed2027Metrics)
$seed2027RunIds = @($seed2027Rows | ForEach-Object { [string]$_.run_id })
$expectedSeed2027RunIds = @(
    $seriesIds | ForEach-Object {
        "{0}__seed_2027__PAPERNEG_NONOVERLAP__LAST" -f $_
    }
)
$seed2027IdDifference = @(
    Compare-Object -ReferenceObject $expectedSeed2027RunIds -DifferenceObject $seed2027RunIds
)
if (
    $seed2027Rows.Count -ne 530 -or
    @($seed2027RunIds | Sort-Object -Unique).Count -ne 530 -or
    $seed2027IdDifference.Count -ne 0 -or
    @($seed2027Rows | Where-Object seed -ne '2027').Count -ne 0 -or
    @($seed2027Rows | Where-Object trajectory -ne $trajectory).Count -ne 0 -or
    @($seed2027Rows | Where-Object checkpoint -ne 'LAST').Count -ne 0
) {
    throw "Existing seed-2027 main metrics are not exact 530-file LAST coverage"
}

New-Item -ItemType Directory -Path $ConfirmationEvaluationRoot -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$metricSources = @{ 2027 = $Seed2027Metrics }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($seed in @(2028, 2029)) {
    $seedOutput = Join-Path $ConfirmationEvaluationRoot "seed_$seed"
    New-Item -ItemType Directory -Path $seedOutput -Force | Out-Null
    $evaluateLog = Join-Path $LogRoot "evaluate_seed${seed}_$stamp.log"
    & $Python @(
        '-m', 'paano_k0.evaluate_benchmark',
        '--config', $Config,
        '--manifest', $Manifest,
        '--vendor-root', $VendorRoot,
        '--results-root', $ResultsRoot,
        '--output-dir', $seedOutput,
        '--seed', [string]$seed,
        '--trajectories', $trajectory,
        '--checkpoint', 'LAST'
    ) 2>&1 | Tee-Object -FilePath $evaluateLog
    if ($LASTEXITCODE -ne 0) {
        throw "Seed-$seed confirmation evaluator failed. Log: $evaluateLog"
    }
    $metricFiles = @(Get-ChildItem -LiteralPath (Join-Path $seedOutput 'metrics') -File -Filter '*.json')
    if ($metricFiles.Count -ne 530) {
        throw "Seed-$seed evaluator metric coverage mismatch: $($metricFiles.Count) != 530"
    }
    $metricSources[$seed] = $seedOutput
}

$aggregateLog = Join-Path $LogRoot "aggregate_$stamp.log"
& $Python @(
    '-m', 'paano_k0.aggregate_confirmation',
    '--config', $Config,
    '--manifest', $Manifest,
    '--seed-2027-metrics', $metricSources[2027],
    '--seed-2028-metrics', $metricSources[2028],
    '--seed-2029-metrics', $metricSources[2029],
    '--output-dir', $OutputDirectory
) 2>&1 | Tee-Object -FilePath $aggregateLog
if ($LASTEXITCODE -ne 0) {
    throw "Full confirmation aggregation failed. Log: $aggregateLog"
}

$requiredOutputs = @(
    'confirmation_seed_track_metrics.csv',
    'confirmation_track_summary.csv',
    'confirmation_summary.json'
)
foreach ($name in $requiredOutputs) {
    $path = Join-Path $OutputDirectory $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Full confirmation output is missing: $path"
    }
}

$summaryPath = Join-Path $OutputDirectory 'confirmation_summary.json'
$summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
$summarySeeds = @($summary.seeds | ForEach-Object { [string]$_ }) -join ','
if (
    [string]$summary.schema_version -ne 'paano-full-confirmation-v1' -or
    $summarySeeds -ne '2027,2028,2029' -or
    [int]$summary.metric_count -ne 1590 -or
    [int]$summary.seed_track_row_count -ne 6 -or
    'outcome' -in @($summary.PSObject.Properties.Name)
) {
    throw "Full confirmation summary violated the fixed no-selection contract"
}
Write-Output (
    "FULL_CONFIRMATION_EVALUATION_COMPLETE LAST_scores=1590 " +
    "evaluated_new=1060 reused_seed_2027=530 seeds=$summarySeeds summary=$summaryPath"
)
