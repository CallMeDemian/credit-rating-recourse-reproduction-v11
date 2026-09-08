<#
.SYNOPSIS
    Canonical clean rebuild of every thesis post-freeze analysis.

.DESCRIPTION
    This is the only PowerShell entry point needed after Oracle, RL, and LLM
    runs have been archived under data\final_freeze.

    It performs no live API calls and does not retrain Oracle/RL or regenerate
    Stage7-9 outputs. It reads frozen artifacts and rebuilds all paper-used
    post-freeze analyses under exactly:

        data\analysis\paper_repro\

    Existing canonical output is moved to:

        data\reproduction\archive\paper_repro_rebuilds\previous_<timestamp>\

    If the rebuild fails, the partial output is preserved under the same
    archive root and the previous canonical output is restored.

.EXAMPLE
    powershell.exe -NoProfile -ExecutionPolicy Bypass `
      -File .\tools\run_postfreeze_analysis.ps1 `
      -ProjectRoot C:\Users\Demian\Desktop\thesis_repo

    Add -PlanOnly to check only the required analysis input files without moving
    or rebuilding data\analysis\paper_repro.
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$PythonExe = "",
    [Parameter(Mandatory = $true)][string]$PythonSourceRoot,
    [Parameter(Mandatory = $true)][string]$PythonSitePackagesRoot,
    [Parameter(Mandatory = $true)][string]$PythonEnvironmentConfig,
    [Parameter(Mandatory = $true)][string]$PythonEnvironmentContractId,
    [Parameter(Mandatory = $true)][string]$ExecutionRunRoot,
    [Parameter(Mandatory = $true)][string]$TemporaryRoot,
    [Parameter(Mandatory = $true)][string]$RawRoot,
    [string]$ResumePartialAnalysisPath = "",
    [string]$ResumeFailedAnalysisPath = "",
    [ValidateSet("current_comprehensive", "historical_20260715")]
    [string]$AnalysisProfile = "current_comprehensive",
    [string]$EligibleRunCatalog = "",
    [switch]$ReplaceOutput,
    [switch]$RunN5MAxisSwap,
    [string]$AxisSwapArm0p75 = "",
    [string]$AxisSwapArm1p27 = "",
    [string]$AxisSwapArm2p00 = "",
    [string]$AxisSwapArmUnbounded = "",
    [int]$AxisSwapShapleyPermutations = 64,
    [int]$AxisSwapSeed = 20260714,
    [string]$AxisSwapOutputDir = "data\analysis\n5m_axis_swap",
    [switch]$AxisSwapOatOnly,
    [switch]$PlanOnly
)
# V11 calls this preserved V10 analysis directly against its canonical
# data/final_freeze working copy. No mount or compatibility view is created.

. (Join-Path $PSScriptRoot '_repro_common.ps1')

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "`n==== $Message ====" -ForegroundColor Cyan
}

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label,
        [ValidateSet("Leaf", "Container")][string]$PathType
    )

    $exists = if ($PathType -eq "Leaf") {
        Test-Path -LiteralPath $Path -PathType Leaf
    } else {
        Test-Path -LiteralPath $Path -PathType Container
    }

    if (-not $exists) {
        throw "$Label is missing: $Path"
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Move-DirectoryRobust {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "$Label source directory is missing: $Source"
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "$Label destination already exists: $Destination"
    }

    $DestinationParent = Split-Path -Parent $Destination
    if (-not [string]::IsNullOrWhiteSpace($DestinationParent)) {
        New-Item -ItemType Directory -Path $DestinationParent -Force | Out-Null
    }

    $RoboCopy = Get-Command robocopy.exe -ErrorAction SilentlyContinue
    if ($null -eq $RoboCopy) {
        Move-Item -LiteralPath $Source -Destination $Destination
        return
    }

    # robocopy is used for generated analysis trees because Windows PowerShell
    # Move-Item can fail on deeply nested output paths and mask the original
    # Python exception during failure preservation/restoration.
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $RoboCopy.Source `
          $Source `
          $Destination `
          /E /MOVE /COPY:DT /DCOPY:DT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }

    if ($ExitCode -ge 8) {
        throw "$Label robocopy failed with exit=$ExitCode; source=$Source; destination=$Destination"
    }
    if (Test-Path -LiteralPath $Source) {
        Remove-Item -LiteralPath $Source -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $Destination -PathType Container)) {
        throw "$Label did not create destination: $Destination"
    }
    Set-ReproTreeWritable -Path $Destination -Label $Label
    Assert-ReproWritableDirectory -Path $Destination -Label $Label
}

function Resolve-CanonicalN5MArm {
    param(
        [Parameter(Mandatory = $true)][string]$BudgetToken,
        [string]$ExplicitPath = ""
    )
    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        return Resolve-RequiredPath -Path $ExplicitPath -Label "N5M axis-swap arm $BudgetToken" -PathType Container
    }
    $pattern = "N5M_C4C6_L1_${BudgetToken}_ICb_*"
    $candidates = @(
        Get-ChildItem -LiteralPath $script:LlmRuns -Directory -Filter $pattern -ErrorAction SilentlyContinue |
        Where-Object {
            $manifestPath = Join-Path $_.FullName "archive_manifest.json"
            if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return $false }
            try {
                $meta = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
                return ([string]$meta.run_role -eq "paper_n5_matched_budget_frontier_icb")
            }
            catch { return $false }
        }
    )
    if ($candidates.Count -eq 0) { throw "No canonical N5M axis-swap arm found for token=$BudgetToken. Pass the explicit -AxisSwapArm... path." }
    if ($candidates.Count -gt 1) {
        $paths = ($candidates | ForEach-Object { $_.FullName }) -join "`n  "
        throw "Ambiguous canonical N5M axis-swap arm for token=$BudgetToken. Pass an explicit path. Candidates:`n  $paths"
    }
    return $candidates[0].FullName
}

function Invoke-PythonCaptured {
    param(
        [Parameter(Mandatory = $true, ParameterSetName = "Arguments")]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true, ParameterSetName = "ScriptText")]
        [AllowEmptyString()][string]$ScriptText,
        [string]$LogPath = "",
        [switch]$EchoOutput
    )

    # Windows PowerShell 5.1 converts native stderr into ErrorRecord objects.
    # With ErrorActionPreference=Stop this used to terminate on the first
    # traceback line and hide the real Python error behind NativeCommandError.
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ($PSCmdlet.ParameterSetName -eq "ScriptText") {
            $ProcessResult = Invoke-ReproPythonStdinCaptured `
                -PythonExe $script:Py `
                -ScriptText $ScriptText
            $RawOutput = @($ProcessResult.Output)
            $ExitCode = [int]$ProcessResult.ExitCode
            $StandardOutput = @($ProcessResult.StandardOutput)
            $StandardError = @($ProcessResult.StandardError)
            $FirstStderr = $ProcessResult.FirstStderr
        }
        else {
            $RawOutput = @(& $script:Py -B @Arguments 2>&1)
            $ExitCode = $LASTEXITCODE
            $StandardOutput = @($RawOutput)
            $StandardError = @()
            $FirstStderr = $null
        }
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }

    $Lines = @(
        $RawOutput | ForEach-Object {
            if ($null -eq $_) { "" } else { $_.ToString() }
        }
    )

    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        $LogDirectory = Split-Path -Parent $LogPath
        if (-not [string]::IsNullOrWhiteSpace($LogDirectory)) {
            New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
        }
        $Lines | Set-Content -LiteralPath $LogPath -Encoding UTF8
    }

    if ($EchoOutput) {
        $Lines | ForEach-Object { Write-Host $_ }
    }

    return [pscustomobject]@{
        ExitCode = [int]$ExitCode
        Output = $Lines
        StandardOutput = $StandardOutput
        StandardError = $StandardError
        FirstStderr = $FirstStderr
        LogPath = $LogPath
    }
}

function Write-StructuredPythonFailure {
    param(
        [string]$FailureReport,
        [string]$LogPath
    )

    if (Test-Path -LiteralPath $FailureReport -PathType Leaf) {
        try {
            $Failure = Get-Content -LiteralPath $FailureReport -Raw -Encoding UTF8 | ConvertFrom-Json
            Write-Host "Python failure: $($Failure.exception_type): $($Failure.exception_message)" -ForegroundColor Red
        }
        catch {
            Write-Host "Python failure report exists but could not be parsed: $FailureReport" -ForegroundColor Red
        }
        Write-Host "Failure report: $FailureReport" -ForegroundColor Yellow
    }
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        Write-Host "Command log   : $LogPath" -ForegroundColor Yellow
    }
}

function Invoke-FinalPaperAssets {
    param(
        [Parameter(Mandatory = $true)][string]$AnalysisDir,
        [Parameter(Mandatory = $true)][string]$LabelSuffix
    )

    Set-ReproTreeWritable -Path $AnalysisDir -Label "Paper analysis output $LabelSuffix"

    $assetResult = Invoke-PythonCaptured `
        -Arguments @(
            "-m", "credit_recourse.analysis.paper_repro_assets",
            "--project-root", $script:Root,
            "--analysis-dir", $AnalysisDir,
            "--analysis-profile", $AnalysisProfile,
            "--strict"
        ) `
        -EchoOutput
    if ($assetResult.ExitCode -ne 0) {
        throw "Paper asset generation failed $LabelSuffix."
    }

}


trap {
    Write-Host "`nFAIL: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "The runner stopped without replacing a valid previous paper_repro output." -ForegroundColor Yellow
    exit 1
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$Root = Resolve-RequiredPath -Path $ProjectRoot -Label "Project root" -PathType Container
$script:Root = $Root

$PyCandidate = if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    Join-Path $Root ".venv\Scripts\python.exe"
} else {
    $PythonExe
}

$Py = Resolve-RequiredPath -Path $PyCandidate -Label "Virtualenv Python" -PathType Leaf
$script:Py = $Py

$RawRoot = Resolve-RequiredPath -Path $RawRoot -Label "Raw input root" -PathType Container
$RawNonfinancialRoot = Resolve-RequiredPath `
    -Path (Join-Path $RawRoot "raw_nonfinancial") `
    -Label "Raw nonfinancial input" `
    -PathType Container

$FinalFreeze = Join-Path $Root "data\final_freeze"
$LlmRuns = Join-Path $FinalFreeze "llm_runs"
$Target = Join-Path $Root "data\analysis\paper_repro"

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ArchiveRoot = Join-Path $Root "data\reproduction\archive\paper_repro_rebuilds"
$Backup = Join-Path $ArchiveRoot "previous_$RunStamp"
$Failed = Join-Path $ArchiveRoot "failed_$RunStamp"

Write-Step "Check required analysis inputs"

$RequiredFiles = @(
    "src\credit_recourse\analysis\paper_repro_analysis.py",
    "src\credit_recourse\analysis\paper_output_layout.py",
    "src\credit_recourse\analysis\paper_repro_assets.py",
    "src\credit_recourse\analysis\icc_probe_analysis.py",
    "src\credit_recourse\contracts\paper_reproduction.py",
    "src\credit_recourse\utils\writable_outputs.py",
    "src\credit_recourse\configs\paper_reproduction_profile.json"
)

if ($RunN5MAxisSwap) {
    $RequiredFiles += @(
        "src\credit_recourse\analysis\n5m_axis_swap_intervention.py"
    )
}

foreach ($relative in $RequiredFiles) {
    Resolve-RequiredPath `
        -Path (Join-Path $Root $relative) `
        -Label "Required source file" `
        -PathType Leaf | Out-Null
}

$RequiredDirectories = @(
    "data\final_freeze",
    "data\final_freeze\llm_runs"
)

foreach ($relative in $RequiredDirectories) {
    Resolve-RequiredPath `
        -Path (Join-Path $Root $relative) `
        -Label "Required data directory" `
        -PathType Container | Out-Null
}

if ((Get-ChildItem -LiteralPath $LlmRuns -Directory | Measure-Object).Count -eq 0) {
    throw "No LLM run directories were found under: $LlmRuns"
}

Set-Location -LiteralPath $Root

Initialize-ReproFrozenPythonEnvironment `
    -Root $Root `
    -PythonExe $Py `
    -PythonSourceRoot $PythonSourceRoot `
    -PythonSitePackagesRoot $PythonSitePackagesRoot `
    -PythonEnvironmentConfig $PythonEnvironmentConfig `
    -PythonEnvironmentContractId $PythonEnvironmentContractId `
    -ExecutionRunRoot $ExecutionRunRoot `
    -TemporaryRoot $TemporaryRoot
$AnalysisSelectionArguments = @("--analysis-profile", $AnalysisProfile)
$AnalysisSelectionArguments += @("--raw-root", $RawRoot)
$EligibleRunCatalogResolved = ""
if (-not [string]::IsNullOrWhiteSpace($EligibleRunCatalog)) {
    $EligibleRunCatalogResolved = Resolve-RequiredPath `
        -Path $EligibleRunCatalog `
        -Label "Eligible LLM run catalog" `
        -PathType Leaf
    $AnalysisSelectionArguments += @(
        "--eligible-run-catalog",
        $EligibleRunCatalogResolved
    )
}
$script:EligibleRunCatalogResolved = $EligibleRunCatalogResolved

$AxisSwapArguments = @()
$AxisSwapManifest = ""
if ($RunN5MAxisSwap) {
    if ($AxisSwapShapleyPermutations -le 0) { throw "AxisSwapShapleyPermutations must be positive." }
    $arm0p75 = Resolve-CanonicalN5MArm -BudgetToken "0p75" -ExplicitPath $AxisSwapArm0p75
    $armUnbounded = Resolve-CanonicalN5MArm -BudgetToken "unbounded" -ExplicitPath $AxisSwapArmUnbounded
    $axisArms = @(@("0p75", $arm0p75), @("unbounded", $armUnbounded))
    if (-not [string]::IsNullOrWhiteSpace($AxisSwapArm1p27)) {
        $axisArms += ,@("1p27", (Resolve-CanonicalN5MArm -BudgetToken "1p27" -ExplicitPath $AxisSwapArm1p27))
    }
    if (-not [string]::IsNullOrWhiteSpace($AxisSwapArm2p00)) {
        $axisArms += ,@("2p00", (Resolve-CanonicalN5MArm -BudgetToken "2p00" -ExplicitPath $AxisSwapArm2p00))
    }
    $axisOut = if ([System.IO.Path]::IsPathRooted($AxisSwapOutputDir)) { $AxisSwapOutputDir } else { Join-Path $Root $AxisSwapOutputDir }
    $AxisSwapArguments = @(
        "-m", "credit_recourse.analysis.n5m_axis_swap_intervention",
        "--project-root", $Root,
        "--out", $axisOut,
        "--shapley-permutations", "$AxisSwapShapleyPermutations",
        "--shapley-arms", "0p75,unbounded",
        "--seed", "$AxisSwapSeed",
        "--fidelity-tol", "1e-9",
        "--efficiency-tol", "1e-8"
    )
    foreach ($axisArm in $axisArms) {
        $AxisSwapArguments += @("--arm", "$($axisArm[0])=$($axisArm[1])")
    }
    if ($AxisSwapOatOnly) { $AxisSwapArguments += "--oat-only" }
    $AxisSwapManifest = Join-Path $axisOut "intervention_manifest.json"
}

$ImportSmoke = @'
import os
import pathlib

import credit_recourse
from credit_recourse.analysis.paper_repro_analysis import run_analysis

expected_source = pathlib.Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]).resolve()
actual_source = pathlib.Path(credit_recourse.__file__).resolve()
if expected_source not in actual_source.parents:
    raise RuntimeError(
        f"paper analysis import escaped source root: {actual_source} != {expected_source}"
    )
print("paper analysis import PASS", actual_source)
'@
$ImportResult = Invoke-PythonCaptured -ScriptText $ImportSmoke
if ($null -eq $ImportResult -or $ImportResult.ExitCode -ne 0) {
    if ($null -ne $ImportResult) {
        $ImportResult.Output | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    }
    throw "Paper analysis import failed."
}

if (-not [string]::IsNullOrWhiteSpace($ResumePartialAnalysisPath)) {
    Write-Step "Resume preserved partial canonical analysis"

    $ResumeSource = Resolve-RequiredPath `
        -Path $ResumePartialAnalysisPath `
        -Label "Partial analysis directory" `
        -PathType Container
    $ResumeInPlace = [System.StringComparer]::OrdinalIgnoreCase.Equals(
        [System.IO.Path]::GetFullPath($ResumeSource),
        [System.IO.Path]::GetFullPath($Target)
    )
    $ResumeManifestPath = Join-Path $ResumeSource "00_manifest\paper_repro_analysis_manifest.json"
    $ResumeManifestPath = Resolve-RequiredPath `
        -Path $ResumeManifestPath `
        -Label "Partial analysis manifest" `
        -PathType Leaf
    $ResumeManifest = Get-Content -LiteralPath $ResumeManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$ResumeManifest.status -ne "FAIL") {
        throw "Partial analysis manifest is not a failed resumable run: status=$($ResumeManifest.status)"
    }
    if ([string]$ResumeManifest.analysis_profile -ne $AnalysisProfile) {
        throw "Partial analysis profile mismatch: partial=$($ResumeManifest.analysis_profile), requested=$AnalysisProfile"
    }

    $ResumeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ResumeLog = Join-Path $Root ("data\reproduction\diagnostics\paper_repro_partial_resume_" + $ResumeStamp + ".log")
    $ResumeFailure = Join-Path $Root ("data\reproduction\diagnostics\paper_repro_partial_resume_" + $ResumeStamp + "_failure.json")
    $ResumeArguments = @(
        "-m", "credit_recourse.analysis.paper_repro_analysis",
        "--project-root", $Root,
        "--analysis-dir", $ResumeSource,
        "--failure-report", $ResumeFailure,
        "--concise-errors"
    )
    $ResumeArguments += $AnalysisSelectionArguments

    Write-Host "Partial source: $ResumeSource"
    Write-Host "CMD> $Py $($ResumeArguments -join ' ')" -ForegroundColor DarkGray
    $ResumeResult = Invoke-PythonCaptured `
        -Arguments $ResumeArguments `
        -LogPath $ResumeLog `
        -EchoOutput
    if ($ResumeResult.ExitCode -ne 0) {
        Write-StructuredPythonFailure -FailureReport $ResumeFailure -LogPath $ResumeLog
        throw "Partial post-freeze analysis resume failed with exit=$($ResumeResult.ExitCode). The partial directory remains preserved."
    }
    Remove-Item -LiteralPath $ResumeFailure -Force -ErrorAction SilentlyContinue

    Write-Step "Run canonical remaining no-API thesis analyses after partial resume"
    $RemainingArguments = @(
        "-m", "credit_recourse.analysis.remaining_thesis_analyses",
        "--project-root", $Root,
        "--raw-root", $RawRoot,
        "--mode", "noapi",
        "--analysis-profile", $AnalysisProfile,
        "--allow-analysis-running"
    )
    if (-not [string]::IsNullOrWhiteSpace($script:EligibleRunCatalogResolved)) {
        $RemainingArguments += @(
            "--eligible-run-catalog",
            $script:EligibleRunCatalogResolved
        )
    }
    $RemainingResult = Invoke-PythonCaptured `
        -Arguments $RemainingArguments `
        -EchoOutput
    if ($RemainingResult.ExitCode -ne 0) {
        throw "Canonical remaining no-API analyses failed after partial resume with exit=$($RemainingResult.ExitCode)"
    }

    Write-Step "Generate complete paper assets after partial resume"
    Invoke-FinalPaperAssets `
        -AnalysisDir $ResumeSource `
        -LabelSuffix "[partial resume]"

    if (-not $ResumeInPlace) {
        New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
        if (Test-Path -LiteralPath $Target) {
            Move-DirectoryRobust -Source $Target -Destination $Backup -Label "Previous canonical output archive"
        }
        Move-DirectoryRobust -Source $ResumeSource -Destination $Target -Label "Partial analysis promotion"
    }

    Write-Host "`nPASS: preserved partial paper analysis completed without repeating finished shuffle draws." -ForegroundColor Green
    Write-Host "Canonical output: $Target"
    exit 0
}

$InputCheckLog = Join-Path `
    $Root `
    ("data\reproduction\diagnostics\paper_repro_input_check_" + $RunStamp + ".log")
$InputCheckFailure = Join-Path `
    $Root `
    ("data\reproduction\diagnostics\paper_repro_input_check_" + $RunStamp + "_failure.json")
$InputCheckResult = Invoke-PythonCaptured `
    -Arguments (@(
        "-m", "credit_recourse.analysis.paper_repro_analysis",
        "--project-root", $Root,
        "--analysis-dir", $Target,
        "--check-inputs-only",
        "--failure-report", $InputCheckFailure,
        "--concise-errors"
    ) + $AnalysisSelectionArguments) `
    -LogPath $InputCheckLog
if ($InputCheckResult.ExitCode -ne 0) {
    $InputCheckResult.Output | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    Write-StructuredPythonFailure -FailureReport $InputCheckFailure -LogPath $InputCheckLog
    throw "Required analysis input check failed."
}
Remove-Item -LiteralPath $InputCheckFailure -Force -ErrorAction SilentlyContinue

Write-Host "Required inputs: PASS" -ForegroundColor Green
Write-Host "Project root   : $Root"
Write-Host "Python         : $Py"
Write-Host "Input root     : $FinalFreeze"
Write-Host "Output         : $Target"
Write-Host "LLM API calls  : NONE" -ForegroundColor Green

if ($PlanOnly) {
    if (-not [string]::IsNullOrWhiteSpace($ResumeFailedAnalysisPath)) {
        throw "-PlanOnly cannot be combined with -ResumeFailedAnalysisPath."
    }
    Write-Host "`nPASS: all required analysis input files are present." -ForegroundColor Green
    if ($RunN5MAxisSwap) {
        Write-Host "N5M axis-swap plan (LLM API calls: NONE):" -ForegroundColor Green
        Write-Host "CMD> $Py $($AxisSwapArguments -join ' ')" -ForegroundColor DarkGray
    }
    Write-Host "No output was moved or rebuilt because -PlanOnly was specified." -ForegroundColor Yellow
    exit 0
}

if (-not [string]::IsNullOrWhiteSpace($ResumeFailedAnalysisPath)) {
    Write-Step "Resume failed canonical analysis after completed frontier"

    $ResumeSource = Resolve-RequiredPath `
        -Path $ResumeFailedAnalysisPath `
        -Label "Failed analysis directory" `
        -PathType Container

    Set-ReproTreeWritable -Path $ResumeSource -Label "Resumed paper analysis"
    Assert-ReproWritableDirectory -Path $ResumeSource -Label "Resumed paper analysis"

    $ResumeInPlace = [System.StringComparer]::OrdinalIgnoreCase.Equals(
        [System.IO.Path]::GetFullPath($ResumeSource),
        [System.IO.Path]::GetFullPath($Target)
    )

    $ResumeManifest = Join-Path $ResumeSource "00_manifest\paper_repro_analysis_manifest.json"
    $ResumeFrontierMetadata = Join-Path $ResumeSource "03_output_contract_diagnostics\budget_frontier\metadata.json"
    $ResumeFrontierStatus = Join-Path $ResumeSource "03_output_contract_diagnostics\budget_frontier\frontier_grid_status.csv"
    foreach ($required in @($ResumeManifest, $ResumeFrontierMetadata, $ResumeFrontierStatus)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Resume source is not a completed-frontier failed analysis directory. Missing: $required"
        }
    }

    $ResumeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ResumeLog = Join-Path $Root ("data\reproduction\diagnostics\paper_repro_resume_" + $ResumeStamp + ".log")
    $ResumeFailure = Join-Path $Root ("data\reproduction\diagnostics\paper_repro_resume_" + $ResumeStamp + "_failure.json")
    $ResumeArguments = @(
        "-m", "credit_recourse.analysis.paper_repro_analysis",
        "--project-root", $Root,
        "--analysis-dir", $ResumeSource,
        "--resume-after-frontier",
        "--failure-report", $ResumeFailure,
        "--concise-errors"
    )
    $ResumeArguments += $AnalysisSelectionArguments

    Write-Host "Resume source : $ResumeSource"
    Write-Host "Canonical out : $Target"
    Write-Host "CMD> $Py $($ResumeArguments -join ' ')" -ForegroundColor DarkGray

    $ResumeResult = Invoke-PythonCaptured `
        -Arguments $ResumeArguments `
        -LogPath $ResumeLog `
        -EchoOutput

    if ($ResumeResult.ExitCode -ne 0) {
        Write-StructuredPythonFailure -FailureReport $ResumeFailure -LogPath $ResumeLog
        throw "Post-freeze analysis resume failed with exit=$($ResumeResult.ExitCode). The failed archive was preserved in place."
    }
    Remove-Item -LiteralPath $ResumeFailure -Force -ErrorAction SilentlyContinue

    Write-Step "Generate complete paper assets after resume"
    Invoke-FinalPaperAssets `
        -AnalysisDir $ResumeSource `
        -LabelSuffix "[resume]"

    if ($ResumeInPlace) {
        Write-Host "`nPASS: canonical partial paper analysis resumed in place without rerunning pre-frontier work." -ForegroundColor Green
        Write-Host "Canonical output:"
        Write-Host "  $Target"
        exit 0
    }

    New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
    if (Test-Path -LiteralPath $Target) {
        if (Test-Path -LiteralPath $Backup) {
            throw "Backup path already exists: $Backup"
        }
        Move-DirectoryRobust -Source $Target -Destination $Backup -Label "Previous canonical output archive"
        Write-Host "Previous canonical output moved to:" -ForegroundColor Yellow
        Write-Host "  $Backup"
    }

    Move-DirectoryRobust -Source $ResumeSource -Destination $Target -Label "Resumed analysis promotion"
    Write-Host "`nPASS: archived failed paper analysis resumed without rerunning pre-frontier work." -ForegroundColor Green
    Write-Host "Canonical output:"
    Write-Host "  $Target"
    exit 0
}

Write-Step "Prepare clean canonical output"

New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null

$PreviousOutputMoved = $false
if (Test-Path -LiteralPath $Target) {
    Move-DirectoryRobust -Source $Target -Destination $Backup -Label "Previous canonical output archive"
    $PreviousOutputMoved = $true
    Write-Host "Previous canonical output moved to:" -ForegroundColor Yellow
    Write-Host "  $Backup"
} else {
    Write-Host "No previous canonical output found."
}

$RunLog = Join-Path `
    $Root `
    ("data\reproduction\diagnostics\paper_repro_run_" + $RunStamp + ".log")
$RunFailure = Join-Path `
    $Root `
    ("data\reproduction\diagnostics\paper_repro_run_" + $RunStamp + "_failure.json")
$Arguments = @(
    "-m", "credit_recourse.analysis.paper_repro_analysis",
    "--project-root", $Root,
    "--analysis-dir", $Target,
    "--failure-report", $RunFailure,
    "--concise-errors"
)
$Arguments += $AnalysisSelectionArguments

try {
    Write-Step "Run every paper post-freeze analysis"
    Write-Host "CMD> $Py $($Arguments -join ' ')" -ForegroundColor DarkGray

    $RunResult = Invoke-PythonCaptured `
        -Arguments $Arguments `
        -LogPath $RunLog `
        -EchoOutput

    if ($RunResult.ExitCode -ne 0) {
        Write-StructuredPythonFailure -FailureReport $RunFailure -LogPath $RunLog
        throw "Canonical post-freeze analysis failed with exit=$($RunResult.ExitCode)"
    }
    Remove-Item -LiteralPath $RunFailure -Force -ErrorAction SilentlyContinue

    Write-Step "Run canonical remaining no-API thesis analyses"
    $RemainingArguments = @(
        "-m", "credit_recourse.analysis.remaining_thesis_analyses",
        "--project-root", $Root,
        "--raw-root", $RawRoot,
        "--mode", "noapi",
        "--analysis-profile", $AnalysisProfile,
        "--allow-analysis-running"
    )
    if (-not [string]::IsNullOrWhiteSpace($script:EligibleRunCatalogResolved)) {
        $RemainingArguments += @(
            "--eligible-run-catalog",
            $script:EligibleRunCatalogResolved
        )
    }
    Write-Host "CMD> $Py $($RemainingArguments -join ' ')" -ForegroundColor DarkGray
    $RemainingResult = Invoke-PythonCaptured `
        -Arguments $RemainingArguments `
        -EchoOutput
    if ($RemainingResult.ExitCode -ne 0) {
        throw "Canonical remaining no-API thesis analyses failed with exit=$($RemainingResult.ExitCode)"
    }

    Write-Step "Generate complete paper assets"
    Invoke-FinalPaperAssets `
        -AnalysisDir $Target `
        -LabelSuffix "[canonical]"

    $ExpectedTopLevel = @(
        "00_manifest",
        "01_substrate_validation",
        "02_llm_hypothesis_tests",
        "03_output_contract_diagnostics",
        "04_paper_assets",
        "05_extension_e3_e4",
        "06_thesis_registry",
        "99_verification"
    )

    $MissingTopLevel = @(
        $ExpectedTopLevel |
        Where-Object {
            -not (
                Test-Path `
                    -LiteralPath (Join-Path $Target $_) `
                    -PathType Container
            )
        }
    )

    if ($MissingTopLevel.Count -gt 0) {
        throw "Canonical output is incomplete. Missing directories: $($MissingTopLevel -join ', ')"
    }

    Write-Host ""
    Write-Host "PASS: all paper-used post-freeze analyses were rebuilt." -ForegroundColor Green
    Write-Host "Canonical output:"
    Write-Host "  $Target"

    if ($PreviousOutputMoved) {
        Write-Host "Previous output backup:"
        Write-Host "  $Backup"
    }

    Write-Host ""
    Write-Host "Canonical sections:" -ForegroundColor Cyan
    foreach ($name in $ExpectedTopLevel) {
        Write-Host "  $name"
    }
}
catch {
    $OriginalFailure = $_
    $RecoveryErrors = New-Object System.Collections.Generic.List[string]

    try {
        if (Test-Path -LiteralPath $Target) {
            if (Test-Path -LiteralPath $Failed) {
                throw "Failure archive path already exists and cannot be overwritten: $Failed"
            }

            Move-DirectoryRobust -Source $Target -Destination $Failed -Label "Failed partial output archive"
            Write-Host "`nFailed partial output preserved at:" -ForegroundColor Red
            Write-Host "  $Failed" -ForegroundColor Red
        }
    }
    catch {
        $RecoveryErrors.Add("failed-output archive: $($_.Exception.Message)")
        Write-Host "WARNING: failed partial output could not be archived: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    try {
        if ($PreviousOutputMoved -and (Test-Path -LiteralPath $Backup)) {
            Move-DirectoryRobust -Source $Backup -Destination $Target -Label "Previous canonical output restore"
            Write-Host "Previous canonical output restored to:" -ForegroundColor Yellow
            Write-Host "  $Target"
        }
    }
    catch {
        $RecoveryErrors.Add("previous-output restore: $($_.Exception.Message)")
        Write-Host "WARNING: previous canonical output could not be restored: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    if ($RecoveryErrors.Count -gt 0) {
        $RecoveryText = $RecoveryErrors -join " | "
        throw "$($OriginalFailure.Exception.Message) | recovery_errors=$RecoveryText"
    }
    throw $OriginalFailure
}

if ($RunN5MAxisSwap) {
    Write-Step "Run integrated N5M axis-swap OAT/Shapley"
    Write-Host "LLM API calls: NONE" -ForegroundColor Green
    Write-Host "CMD> $Py $($AxisSwapArguments -join ' ')" -ForegroundColor DarkGray
    $AxisResult = Invoke-PythonCaptured -Arguments $AxisSwapArguments -EchoOutput
    if ($AxisResult.ExitCode -ne 0) { throw "Integrated N5M axis-swap analysis failed. Canonical paper_repro output remains valid." }
    if (-not (Test-Path -LiteralPath $AxisSwapManifest -PathType Leaf)) { throw "Integrated N5M axis-swap manifest missing: $AxisSwapManifest" }
    $AxisMeta = Get-Content -LiteralPath $AxisSwapManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$AxisMeta.status -ne "PASS") { throw "Integrated N5M axis-swap manifest status=$($AxisMeta.status)" }
    Write-Host "PASS: $AxisSwapManifest" -ForegroundColor Green
}
