[CmdletBinding()]
param(
    [int]$RunnerPid = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$CodeRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $CodeRoot
$Manifest = Join-Path $RepoRoot 'docs\K0_DATA_MANIFEST.csv'
$ResultsRoot = Join-Path $RepoRoot 'results\k0'
$LogRoot = Join-Path $RepoRoot 'logs\primary_k0'

if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Frozen manifest is missing: $Manifest"
}
$rows = @(Import-Csv -LiteralPath $Manifest)
$trajectories = @('OFFICIAL', 'PAPERNEG', 'PAPERNEG_NONOVERLAP', 'RAND_BN')
$completed = 0
$failed = 0
$current = 'none'
foreach ($row in $rows) {
    $seriesId = [IO.Path]::GetFileNameWithoutExtension([string]$row.file)
    foreach ($trajectory in $trajectories) {
        $runDirectory = Join-Path $ResultsRoot ("runs\$seriesId\seed_2027\$trajectory")
        if (Test-Path -LiteralPath (Join-Path $runDirectory '_FAILED.json') -PathType Leaf) {
            $failed++
        } elseif (Test-Path -LiteralPath (Join-Path $runDirectory '_SUCCESS') -PathType Leaf) {
            $completed++
        } elseif ($current -eq 'none') {
            $current = "$seriesId/$trajectory"
        }
    }
}

$runnerAlive = $false
if ($RunnerPid -gt 0) {
    $runnerAlive = $null -ne (Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue)
}

$gpuUtil = 'NA'
$vramUsed = 'NA'
$vramTotal = 'NA'
$temperature = 'NA'
$nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
if ($null -ne $nvidiaSmi) {
    $gpuLine = & $nvidiaSmi.Source '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu' '--format=csv,noheader,nounits' 2>$null |
        Select-Object -First 1
    # Some Windows WDDM driver builds return a nonzero process status despite
    # emitting a complete query row. Treat a four-field row as authoritative.
    if ($null -ne $gpuLine) {
        $gpu = @([string]$gpuLine -split ',\s*')
        if ($gpu.Count -eq 4) {
            $gpuUtil, $vramUsed, $vramTotal, $temperature = $gpu
        }
    }
}

function Get-FreeGiB {
    param([Parameter(Mandatory = $true)][string]$DriveName)
    $drive = Get-PSDrive -Name $DriveName -ErrorAction SilentlyContinue
    if ($null -eq $drive) { return 'NA' }
    return ('{0:F2}' -f ($drive.Free / 1GB))
}

$latestLog = 'none'
$latestLogTime = 'NA'
if (Test-Path -LiteralPath $LogRoot -PathType Container) {
    $log = Get-ChildItem -LiteralPath $LogRoot -Filter '*.log' -File |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -ne $log) {
        $latestLog = $log.Name
        $latestLogTime = $log.LastWriteTime.ToString('yyyy-MM-ddTHH:mm:sszzz')
    }
}

Write-Output (
    'runner_alive={0} pid={1} completed={2}/24 failed={3} current={4} gpu_util_pct={5} vram_mib={6}/{7} temp_c={8} c_free_gib={9} d_free_gib={10} latest_log={11} latest_log_time={12}' -f
    $runnerAlive, $RunnerPid, $completed, $failed, $current, $gpuUtil, $vramUsed, $vramTotal, $temperature,
    (Get-FreeGiB -DriveName 'C'), (Get-FreeGiB -DriveName 'D'), $latestLog, $latestLogTime
)
