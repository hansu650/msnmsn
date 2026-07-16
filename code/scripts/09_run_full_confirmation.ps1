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
$Manifest = Join-Path $RepoRoot 'docs\TSB_AD_FULL_EVAL_MANIFEST.csv'
$VendorRoot = Join-Path $WorkspaceRoot 'vendor\PaAno'
$ResultsRoot = 'D:\qintian_experiments\paano_full'
$DecisionPath = Join-Path $RepoRoot 'artifacts\paano_full\decision.json'
$LogRoot = Join-Path $RepoRoot 'logs\full_confirmation'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot, $DecisionPath)) {
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
    throw "Full confirmation is forbidden for outcome: $($decision.outcome)"
}
if (
    [string]$decision.main_trajectory -ne 'PAPERNEG_NONOVERLAP' -or
    [string]$decision.checkpoint -ne 'LAST' -or
    [int]$decision.seed -ne 2027 -or
    [int]$decision.series_count -ne 530 -or
    [int]$decision.metric_count -ne 1590
) {
    throw "Full-main decision provenance does not authorize the frozen confirmation"
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

New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$trajectory = 'PAPERNEG_NONOVERLAP'
$seeds = @(2028, 2029)
$totalJobs = 1060
$jobIndex = 0
foreach ($seed in $seeds) {
    foreach ($row in $rows) {
        $jobIndex++
        $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
        $runDirectory = Join-Path $ResultsRoot (
            "runs\$seriesId\seed_$seed\$trajectory"
        )
        $failurePath = Join-Path $runDirectory '_FAILED.json'
        $trajectorySuccess = Join-Path $runDirectory '_SUCCESS'
        $lastDirectory = Join-Path $runDirectory 'scores\LAST'
        $lastSuccess = Join-Path $lastDirectory '_SUCCESS'
        $scorePath = Join-Path $lastDirectory 'scores.npy'
        $scoreManifestPath = Join-Path $lastDirectory 'score_manifest.json'

        if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
            throw "Retained failure blocks a silent confirmation rerun: $failurePath"
        }
        if (Test-Path -LiteralPath $trajectorySuccess -PathType Leaf) {
            foreach ($committed in @($lastSuccess, $scorePath, $scoreManifestPath)) {
                if (-not (Test-Path -LiteralPath $committed -PathType Leaf)) {
                    throw "Trajectory success has incomplete LAST artifacts: $committed"
                }
            }
            Write-Output (
                "SKIP_VALID_FULL_CONFIRMATION {0}/{1} seed={2} track={3} series={4}" -f
                $jobIndex, $totalJobs, $seed, $row.track, $seriesId
            )
            continue
        }

        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $logPath = Join-Path $LogRoot (
            "{0:D4}_seed{1}_{2}_{3}.log" -f $jobIndex, $seed, $seriesId, $stamp
        )
        Write-Output (
            "RUN_FULL_CONFIRMATION {0}/{1} seed={2} track={3} series={4}" -f
            $jobIndex, $totalJobs, $seed, $row.track, $seriesId
        )
        & $Python @(
            '-m', 'paano_k0.run_benchmark_series',
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
            throw "Full confirmation job failed with exit code $LASTEXITCODE. Log: $logPath"
        }
        foreach ($committed in @(
            $trajectorySuccess, $lastSuccess, $scorePath, $scoreManifestPath
        )) {
            if (-not (Test-Path -LiteralPath $committed -PathType Leaf)) {
                throw "Full confirmation runner did not commit expected artifact: $committed"
            }
        }
    }
}

if ($jobIndex -ne $totalJobs) {
    throw "Full confirmation coverage changed: jobs=$jobIndex expected=$totalJobs"
}
Write-Output (
    'FULL_CONFIRMATION_RUNS_COMPLETE completed=1060 failed=0 ' +
    'trajectory=PAPERNEG_NONOVERLAP seeds=2028,2029 checkpoint=LAST'
)
