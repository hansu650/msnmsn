[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$RunnerPid
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ResultsRoot = 'D:\qintian_experiments\paano_full'
$runsRoot = Join-Path $ResultsRoot 'runs'
$process = Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue
$success = @(
    Get-ChildItem -LiteralPath $runsRoot -Recurse -File -Filter '_SUCCESS' -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq 'PAPERNEG_NONOVERLAP' }
).Count
$failed = @(
    Get-ChildItem -LiteralPath $runsRoot -Recurse -File -Filter '_FAILED.json' -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq 'PAPERNEG_NONOVERLAP' }
).Count

$launcherLog = Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'logs\full_launcher') -File -Filter '*.stdout.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$current = 'none'
if ($null -ne $launcherLog) {
    $line = Get-Content -LiteralPath $launcherLog.FullName -Tail 30 |
        Where-Object { $_ -match '^(RUN_MAIN|SKIP_VALID_MAIN) ' } |
        Select-Object -Last 1
    if ($null -ne $line) { $current = [string]$line }
}

$gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1) -split ','
$c = Get-PSDrive -Name C
$d = Get-PSDrive -Name D
Write-Output (
    'runner_alive={0} pid={1} completed={2}/530 failed={3} current="{4}" gpu_util_pct={5} vram_mib={6}/{7} temp_c={8} c_free_gib={9:N2} d_free_gib={10:N2}' -f
    ($null -ne $process), $RunnerPid, $success, $failed, $current,
    $gpu[0].Trim(), $gpu[1].Trim(), $gpu[2].Trim(), $gpu[3].Trim(),
    ($c.Free / 1GB), ($d.Free / 1GB)
)
