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
$LogRoot = Join-Path $RepoRoot 'logs\full_main'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}
New-Item -ItemType Directory -Path $ResultsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$rows = @(Import-Csv -LiteralPath $Manifest)
$uCount = @($rows | Where-Object track -eq 'U').Count
$mCount = @($rows | Where-Object track -eq 'M').Count
if ($rows.Count -ne 530 -or $uCount -ne 350 -or $mCount -ne 180) {
    throw "Full manifest coverage changed: total=$($rows.Count), U=$uCount, M=$mCount"
}

$trajectory = 'PAPERNEG_NONOVERLAP'
$seed = 2027
$jobIndex = 0
foreach ($row in $rows) {
    $jobIndex++
    $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
    $runDirectory = Join-Path $ResultsRoot ("runs\$seriesId\seed_$seed\$trajectory")
    $failurePath = Join-Path $runDirectory '_FAILED.json'
    $successPath = Join-Path $runDirectory '_SUCCESS'
    if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
        throw "Retained failure blocks a silent rerun: $failurePath"
    }
    if (Test-Path -LiteralPath $successPath -PathType Leaf) {
        Write-Output ("SKIP_VALID_MAIN {0}/530 track={1} series={2}" -f $jobIndex, $row.track, $seriesId)
        continue
    }
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logPath = Join-Path $LogRoot ("{0:D3}_{1}_{2}.log" -f $jobIndex, $seriesId, $stamp)
    Write-Output ("RUN_MAIN {0}/530 track={1} series={2}" -f $jobIndex, $row.track, $seriesId)
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
        throw "Full main job failed with exit code $LASTEXITCODE. Log: $logPath"
    }
    if (-not (Test-Path -LiteralPath $successPath -PathType Leaf)) {
        throw "Full main runner did not commit trajectory success: $runDirectory"
    }
}

Write-Output 'FULL_MAIN_RUNS_COMPLETE completed=530 failed=0 trajectory=PAPERNEG_NONOVERLAP checkpoint=LAST'
