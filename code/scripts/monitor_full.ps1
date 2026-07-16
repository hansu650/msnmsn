[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$RunnerPid,
    [ValidateSet('main', 'ablations', 'confirmation')][string]$Mode = 'main'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ResultsRoot = 'D:\qintian_experiments\paano_full'
$runsRoot = Join-Path $ResultsRoot 'runs'
$Manifest = Join-Path $RepoRoot 'docs\TSB_AD_FULL_EVAL_MANIFEST.csv'
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "Frozen full-Eval manifest is missing: $Manifest"
}
$rows = @(Import-Csv -LiteralPath $Manifest)
$seriesIds = @($rows | ForEach-Object {
    [IO.Path]::GetFileNameWithoutExtension([string]$_.file)
})
if (
    $rows.Count -ne 530 -or
    @($rows | Where-Object track -eq 'U').Count -ne 350 -or
    @($rows | Where-Object track -eq 'M').Count -ne 180 -or
    @($seriesIds | Sort-Object -Unique).Count -ne 530
) {
    throw 'Frozen full-Eval manifest coverage changed.'
}
$process = Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue
$trajectories = @(switch ($Mode) {
    'main' { @('PAPERNEG_NONOVERLAP') }
    'ablations' { @('PAPERNEG', 'OFFICIAL') }
    'confirmation' { @('PAPERNEG_NONOVERLAP') }
})
$seeds = @(if ($Mode -eq 'confirmation') { @(2028, 2029) } else { @(2027) })
$total = $seriesIds.Count * $trajectories.Count * $seeds.Count
$success = 0
$failed = 0
foreach ($seriesId in $seriesIds) {
    foreach ($seed in $seeds) {
        foreach ($trajectory in $trajectories) {
            $runDirectory = Join-Path $runsRoot (
                "$seriesId\seed_$seed\$trajectory"
            )
            if (Test-Path -LiteralPath (Join-Path $runDirectory '_SUCCESS') -PathType Leaf) {
                $success++
            }
            if (Test-Path -LiteralPath (Join-Path $runDirectory '_FAILED.json') -PathType Leaf) {
                $failed++
            }
        }
    }
}

$launcherDirectory = if ($Mode -eq 'main') {
    Join-Path $RepoRoot 'logs\full_launcher'
} elseif ($Mode -eq 'ablations') {
    Join-Path $RepoRoot 'logs\full_ablation_launcher'
} else {
    Join-Path $RepoRoot 'logs\full_confirmation'
}
$launcherLog = Get-ChildItem -LiteralPath $launcherDirectory -File -Filter '*.stdout.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$current = 'none'
if ($null -ne $launcherLog) {
    $statusPattern = if ($Mode -eq 'main') {
        '^(RUN_MAIN|SKIP_VALID_MAIN) '
    } elseif ($Mode -eq 'ablations') {
        '^(RUN_ABLATION|SKIP_VALID_ABLATION) '
    } else {
        '^(RUN_FULL_CONFIRMATION|SKIP_VALID_FULL_CONFIRMATION) '
    }
    $line = Get-Content -LiteralPath $launcherLog.FullName -Tail 30 |
        Where-Object { $_ -match $statusPattern } |
        Select-Object -Last 1
    if ($null -ne $line) { $current = [string]$line }
}

$gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1) -split ','
$c = Get-PSDrive -Name C
$d = Get-PSDrive -Name D
Write-Output (
    'runner_alive={0} pid={1} completed={2}/{3} failed={4} current="{5}" gpu_util_pct={6} vram_mib={7}/{8} temp_c={9} c_free_gib={10:N2} d_free_gib={11:N2} mode={12}' -f
    ($null -ne $process), $RunnerPid, $success, $total, $failed, $current,
    $gpu[0].Trim(), $gpu[1].Trim(), $gpu[2].Trim(), $gpu[3].Trim(),
    ($c.Free / 1GB), ($d.Free / 1GB), $Mode
)
