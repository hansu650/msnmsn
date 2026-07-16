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
$OutputRoot = Join-Path $RepoRoot 'results\k0'
$LogRoot = Join-Path $RepoRoot 'logs\primary_k0'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

function Test-CommittedTrajectory {
    param(
        [Parameter(Mandatory = $true)][string]$RunDirectory,
        [Parameter(Mandatory = $true)][string]$Trajectory
    )
    $checkpoints = if ($Trajectory -eq 'RAND_BN') {
        @('BN_CALIBRATED')
    } else {
        @('BEST', 'LAST')
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RunDirectory '_SUCCESS') -PathType Leaf)) {
        return $false
    }
    foreach ($checkpoint in $checkpoints) {
        $scoreDirectory = Join-Path $RunDirectory ("scores\$checkpoint")
        & $Python -c 'import sys; from pathlib import Path; from paano_k0.artifacts import verify_committed_score; verify_committed_score(Path(sys.argv[1]))' $scoreDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "Existing trajectory marker is invalid: $RunDirectory ($checkpoint)"
        }
    }
    return $true
}

$rows = @(Import-Csv -LiteralPath $Manifest)
if ($rows.Count -ne 6) {
    throw "Frozen manifest must contain exactly six rows; found $($rows.Count)"
}
$expectedOrder = @('NAB', 'IOPS', 'Exathlon', 'SMD', 'SMAP', 'SWaT')
for ($index = 0; $index -lt $rows.Count; $index++) {
    if ([string]$rows[$index].family -ne $expectedOrder[$index]) {
        throw "Frozen manifest order changed at row $($index + 1)."
    }
}

$trajectories = @('OFFICIAL', 'PAPERNEG', 'PAPERNEG_NONOVERLAP', 'RAND_BN')
$jobIndex = 0
$totalJobs = $rows.Count * $trajectories.Count
foreach ($row in $rows) {
    $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
    foreach ($trajectory in $trajectories) {
        $jobIndex++
        $runDirectory = Join-Path $OutputRoot ("runs\$seriesId\seed_2027\$trajectory")
        $failurePath = Join-Path $runDirectory '_FAILED.json'
        if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
            throw "Retained failure record blocks silent rerun: $failurePath"
        }
        if (Test-CommittedTrajectory -RunDirectory $runDirectory -Trajectory $trajectory) {
            Write-Output ("SKIP_VALID {0}/{1} series={2} trajectory={3}" -f $jobIndex, $totalJobs, $seriesId, $trajectory)
            continue
        }

        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $logPath = Join-Path $LogRoot ("{0:D2}_{1}_{2}_{3}.log" -f $jobIndex, $seriesId, $trajectory, $stamp)
        Write-Output ("RUN {0}/{1} series={2} trajectory={3}" -f $jobIndex, $totalJobs, $seriesId, $trajectory)
        & $Python @(
            '-m', 'paano_k0.run_series',
            '--config', $Config,
            '--manifest', $Manifest,
            '--series-id', $seriesId,
            '--trajectory', $trajectory,
            '--seed', '2027',
            '--vendor-root', $VendorRoot,
            '--output-root', $OutputRoot,
            '--device', $Device
        ) 2>&1 | Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "Primary K0 job failed with exit code $LASTEXITCODE. Log: $logPath"
        }
        if (-not (Test-CommittedTrajectory -RunDirectory $runDirectory -Trajectory $trajectory)) {
            throw "Runner exited successfully without a valid trajectory commit: $runDirectory"
        }
    }
}

Write-Output "PRIMARY_K0_RUNS_COMPLETE trajectories=$totalJobs"

