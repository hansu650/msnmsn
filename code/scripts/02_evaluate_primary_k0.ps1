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
$ResultsRoot = Join-Path $RepoRoot 'results\k0'
$LogRoot = Join-Path $RepoRoot 'logs\primary_k0'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot, $ResultsRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$rows = @(Import-Csv -LiteralPath $Manifest)
$trajectories = @('OFFICIAL', 'PAPERNEG', 'PAPERNEG_NONOVERLAP', 'RAND_BN')
$trajectoryCount = 0
$scoreCount = 0
foreach ($row in $rows) {
    $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
    foreach ($trajectory in $trajectories) {
        $runDirectory = Join-Path $ResultsRoot ("runs\$seriesId\seed_2027\$trajectory")
        if (Test-Path -LiteralPath (Join-Path $runDirectory '_FAILED.json') -PathType Leaf) {
            throw "Failure record present; evaluation is forbidden: $runDirectory"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $runDirectory '_SUCCESS') -PathType Leaf)) {
            throw "Missing trajectory success marker: $runDirectory"
        }
        $trajectoryCount++
        $checkpoints = if ($trajectory -eq 'RAND_BN') { @('BN_CALIBRATED') } else { @('BEST', 'LAST') }
        foreach ($checkpoint in $checkpoints) {
            $scoreDirectory = Join-Path $runDirectory ("scores\$checkpoint")
            if (-not (Test-Path -LiteralPath (Join-Path $scoreDirectory '_SUCCESS') -PathType Leaf)) {
                throw "Missing score success marker: $scoreDirectory"
            }
            $scoreCount++
        }
    }
}
if ($trajectoryCount -ne 24 -or $scoreCount -ne 42) {
    throw "Primary coverage mismatch: trajectories=$trajectoryCount scores=$scoreCount"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $LogRoot "evaluate_$stamp.log"
& $Python @(
    '-m', 'paano_k0.evaluate_scores',
    '--config', $Config,
    '--manifest', $Manifest,
    '--vendor-root', $VendorRoot,
    '--results-root', $ResultsRoot
) 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    throw "Primary evaluator failed with exit code $LASTEXITCODE. Log: $logPath"
}

$metricCount = @(Get-ChildItem -LiteralPath (Join-Path $ResultsRoot 'runs') -Filter 'metrics.json' -Recurse -File).Count
if ($metricCount -ne 42) {
    throw "Expected 42 primary metrics.json artifacts; found $metricCount"
}
Write-Output 'PRIMARY_K0_EVALUATION_COMPLETE metrics=42'

