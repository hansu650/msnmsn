[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Python = 'D:\Anaconda\envs\paano_msn\python.exe'
$CodeRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $CodeRoot
$EvaluateScript = Join-Path $PSScriptRoot '07_evaluate_full.ps1'
$ArtifactsDirectory = Join-Path $RepoRoot 'artifacts\paano_full'
$ReportPath = Join-Path $RepoRoot 'docs\experiments\PAANO_FULL_MAIN_RESULTS.md'
$LogRoot = Join-Path $RepoRoot 'logs\full_finalize'
$env:PYTHONPATH = (Join-Path $CodeRoot 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH

foreach ($required in @($Python, $CodeRoot, $RepoRoot, $EvaluateScript)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required finalization path is missing: $required"
    }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required for read-only Git-facing output verification.'
}

New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$FinalizeLog = Join-Path $LogRoot "finalize_$stamp.log"
$currentPhase = 'INITIALIZE'

function Write-PhaseEvent {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][ValidateSet('START', 'COMPLETE', 'FAILED')][string]$State,
        [string]$Message = ''
    )

    $suffix = if ($Message) { " message=$Message" } else { '' }
    $line = "{0} phase={1} state={2}{3}" -f (
        Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK'
    ), $Phase, $State, $suffix
    Write-Output $line
    Add-Content -LiteralPath $FinalizeLog -Value $line -Encoding UTF8
}

function Invoke-LoggedPhase {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    $script:currentPhase = $Name
    Write-PhaseEvent -Phase $Name -State START
    try {
        & $Action
        Write-PhaseEvent -Phase $Name -State COMPLETE
    }
    catch {
        Write-PhaseEvent -Phase $Name -State FAILED -Message $_.Exception.Message
        throw
    }
}

$compactNames = @(
    'main_file_metrics.csv',
    'main_family_metrics.csv',
    'main_track_metrics.csv',
    'ablation_track_metrics.csv',
    'paper_reference_comparison.csv',
    'runtime_summary.csv',
    'decision.json'
)
$compactPaths = @($compactNames | ForEach-Object { Join-Path $ArtifactsDirectory $_ })

try {
    Invoke-LoggedPhase -Name 'EVALUATE_FULL' -Action {
        # This is the first operation allowed to reach evaluator-only labels. Script 07
        # performs exact LAST coverage and score-commit preflight before loading them.
        & $EvaluateScript 2>&1 | Tee-Object -FilePath $FinalizeLog -Append
    }

    Invoke-LoggedPhase -Name 'VERIFY_COMPACT_ARTIFACTS' -Action {
        foreach ($path in $compactPaths) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Required compact full-benchmark artifact is missing: $path"
            }
            if ((Get-Item -LiteralPath $path).Length -le 0) {
                throw "Required compact full-benchmark artifact is empty: $path"
            }
        }
    }

    $gitSha = ''
    Invoke-LoggedPhase -Name 'RENDER_REPORT' -Action {
        $provenancePaths = @(
            'code/src/paano_k0',
            'code/scripts/05_run_full_main.ps1',
            'code/scripts/06_run_full_ablations.ps1',
            'code/scripts/07_evaluate_full.ps1',
            'code/scripts/08_finalize_full.ps1',
            'configs/k0_protocol.yaml',
            'docs/TSB_AD_FULL_EVAL_MANIFEST.csv'
        )
        $dirtyCore = @(
            & git -C $RepoRoot status --porcelain=v1 --untracked-files=all -- @provenancePaths 2>&1
        )
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to verify report-code cleanliness: $($dirtyCore -join ' ')"
        }
        if ($dirtyCore.Count -gt 0) {
            throw (
                'Report code/config/manifest differs from HEAD; commit the frozen ' +
                "provenance surface before finalization: $($dirtyCore -join '; ')"
            )
        }

        $gitShaOutput = @(& git -C $RepoRoot rev-parse HEAD 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to resolve repository HEAD: $($gitShaOutput -join ' ')"
        }
        $script:gitSha = ([string]$gitShaOutput[0]).Trim()
        if (-not $script:gitSha) {
            throw 'Repository HEAD resolved to an empty value.'
        }

        & $Python @(
            '-m', 'paano_k0.report_benchmark',
            '--artifacts-dir', $ArtifactsDirectory,
            '--output', $ReportPath,
            '--git-sha', $script:gitSha
        ) 2>&1 | Tee-Object -FilePath $FinalizeLog -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Full benchmark report renderer failed with exit code $LASTEXITCODE."
        }
        if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
            throw "Full benchmark report was not created: $ReportPath"
        }
        if ((Get-Item -LiteralPath $ReportPath).Length -le 0) {
            throw "Full benchmark report is empty: $ReportPath"
        }
    }

    Invoke-LoggedPhase -Name 'RUN_FULL_PYTEST' -Action {
        Push-Location $CodeRoot
        try {
            & $Python -m pytest -q 2>&1 | Tee-Object -FilePath $FinalizeLog -Append
            if ($LASTEXITCODE -ne 0) {
                throw "Complete pytest suite failed with exit code $LASTEXITCODE."
            }
        }
        finally {
            Pop-Location
        }
    }

    $decision = $null
    Invoke-LoggedPhase -Name 'VERIFY_GIT_FACING_OUTPUTS' -Action {
        $gitFacingPaths = @($compactPaths + $ReportPath)
        $relativePaths = @()
        foreach ($path in $gitFacingPaths) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Git-facing output is missing: $path"
            }
            if ((Get-Item -LiteralPath $path).Length -le 0) {
                throw "Git-facing output is empty: $path"
            }

            $relative = $path.Substring($RepoRoot.Length).TrimStart('\', '/').Replace('\', '/')
            & git -C $RepoRoot check-ignore --quiet -- $relative
            $ignoreExitCode = $LASTEXITCODE
            if ($ignoreExitCode -eq 0) {
                throw "Git-facing output is ignored by repository rules: $relative"
            }
            if ($ignoreExitCode -ne 1) {
                throw "git check-ignore failed for $relative with exit code $ignoreExitCode."
            }
            $relativePaths += $relative
        }

        $statusArguments = @(
            '-C', $RepoRoot, 'status', '--short', '--untracked-files=all', '--'
        ) + $relativePaths
        $gitStatus = @(& git @statusArguments 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to inspect Git-facing outputs: $($gitStatus -join ' ')"
        }
        if ($gitStatus.Count -gt 0) {
            $gitStatus | Tee-Object -FilePath $FinalizeLog -Append
        }

        $decisionPath = Join-Path $ArtifactsDirectory 'decision.json'
        $script:decision = Get-Content -LiteralPath $decisionPath -Raw | ConvertFrom-Json
        $allowedOutcomes = @('CONTINUE_FULL_CONFIRMATION', 'STOP_FULL_MAIN_FAILURE')
        if ([string]$script:decision.outcome -notin $allowedOutcomes) {
            throw "Unregistered full-benchmark outcome: $($script:decision.outcome)"
        }
    }

    Write-Output (
        "FULL_FINALIZATION_COMPLETE outcome={0} compact_artifacts=7 report={1} pytest=passed git_files=8 log={2}" -f
        $decision.outcome, $ReportPath, $FinalizeLog
    )
}
catch {
    Write-Output (
        "FULL_FINALIZATION_FAILED phase={0} error={1} log={2}" -f
        $currentPhase, $_.Exception.Message, $FinalizeLog
    )
    throw
}
