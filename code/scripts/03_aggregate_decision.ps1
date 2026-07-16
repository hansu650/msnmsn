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
$OutputDirectory = Join-Path $ResultsRoot 'aggregate'
$LogRoot = Join-Path $RepoRoot 'logs\primary_k0'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $Config, $Manifest, $VendorRoot, $ResultsRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}
$metricCount = @(Get-ChildItem -LiteralPath (Join-Path $ResultsRoot 'runs') -Filter 'metrics.json' -Recurse -File).Count
if ($metricCount -ne 42) {
    throw "Aggregation requires exactly 42 primary metrics; found $metricCount"
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $LogRoot "aggregate_$stamp.log"
& $Python @(
    '-m', 'paano_k0.aggregate',
    '--config', $Config,
    '--manifest', $Manifest,
    '--vendor-root', $VendorRoot,
    '--results-root', $ResultsRoot,
    '--output-dir', $OutputDirectory
) 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    throw "K0 aggregation failed with exit code $LASTEXITCODE. Log: $logPath"
}

$decisionPath = Join-Path $OutputDirectory 'decision.json'
if (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf)) {
    throw "Aggregator did not commit decision.json: $decisionPath"
}
$decision = Get-Content -LiteralPath $decisionPath -Raw | ConvertFrom-Json
$allowed = @(
    'STOP_NO_ACTIVITY_FAILURE',
    'STOP_NO_PERFORMANCE_HEADROOM',
    'SIMPLE_CHECKPOINT_FIX',
    'SIMPLE_PAPER_PARITY_FIX',
    'GO_METHOD_DESIGN'
)
if ([string]$decision.outcome -notin $allowed) {
    throw "Aggregator returned an unregistered outcome: $($decision.outcome)"
}
Write-Output ("K0_DECISION outcome={0} decision={1}" -f $decision.outcome, $decisionPath)

