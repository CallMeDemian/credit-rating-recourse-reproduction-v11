param(
  [string]$ProjectRoot = "",
  [ValidateSet("default", "calibrated")]
  [string]$SimBusinessPlanMode = "calibrated",
  [double]$MaxRelErrAssets = 0.05,
  [double]$MinCoverageShare = 0.80,
  [ValidateSet(50,65,75,85)]
  [int]$MagnitudeQuantile = 50,
  [double]$MaxIdentityRelErrAssets = 0.02,
  [bool]$PreserveCurrentNonCurrentResidual = $true
)

$ErrorActionPreference = "Stop"

function Write-Section([string]$Title) {
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "[$Title]"
  Write-Host "============================================================"
}

function Resolve-ProjectRoot([string]$ProvidedRoot) {
  if ($ProvidedRoot -and $ProvidedRoot.Trim().Length -gt 0) {
    return (Resolve-Path -LiteralPath $ProvidedRoot).Path
  }
  $ScriptDir = Split-Path -Parent $PSCommandPath
  $Candidate = Split-Path -Parent $ScriptDir
  if (Test-Path -LiteralPath (Join-Path $Candidate "src\credit_recourse") -PathType Container) {
    return (Resolve-Path -LiteralPath $Candidate).Path
  }
  throw "Cannot infer ProjectRoot. Put this script under <ProjectRoot>\tools or pass -ProjectRoot explicitly."
}

function Require-File([string]$Path, [string]$Label) {
  if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { throw "MISSING: $Label => $Path" }
}

function Require-Dir([string]$Path, [string]$Label) {
  if (!(Test-Path -LiteralPath $Path -PathType Container)) { throw "MISSING: $Label => $Path" }
}


function ConvertTo-BoolArg([bool]$Value, [string]$FlagName) {
  if ([string]::IsNullOrWhiteSpace($FlagName)) { throw "Internal runner error: empty boolean flag name." }
  if ($Value) { return @($FlagName) }
  return @()
}

function Invoke-Python([string]$Label, [string[]]$PyArgs, [switch]$AllowGateFail) {
  Write-Section $Label
  if ($null -eq $PyArgs -or $PyArgs.Count -eq 0) { throw "Internal runner error: empty python args for [$Label]." }
  Write-Host ("CMD: " + $PythonExe + " " + ($PyArgs -join " "))
  & $PythonExe @PyArgs
  $Code = $LASTEXITCODE
  if ($Code -eq 2 -and $AllowGateFail) {
    Write-Warning "$Label completed, but validation gate returned FAIL/WARN (exit=2). Artifacts were written; inspect the JSON report."
    return
  }
  if ($Code -ne 0) { throw "FAILED: $Label exit=$Code" }
}

$ProjectRoot = Resolve-ProjectRoot $ProjectRoot
$SrcDir = Join-Path $ProjectRoot "src"
$FinalFreeze = Join-Path $ProjectRoot "data\final_freeze"
$VenvPython = if ($env:REPRO_PYTHON_EXE) { $env:REPRO_PYTHON_EXE } else { Join-Path $ProjectRoot ".venv\Scripts\python.exe" }
$PythonExe = $VenvPython
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Virtualenv Python missing: $PythonExe" }

$env:PYTHONPATH = $SrcDir
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "Python: $PythonExe"
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host "SimBusinessPlanMode: $SimBusinessPlanMode"
Write-Host "MagnitudeQuantile: P$MagnitudeQuantile"
Write-Host "MaxIdentityRelErrAssets: $MaxIdentityRelErrAssets"
Write-Host "PreserveCurrentNonCurrentResidual: $PreserveCurrentNonCurrentResidual"

Require-Dir $SrcDir "src directory"
Require-Dir $FinalFreeze "final_freeze directory"
Require-File (Join-Path $SrcDir "credit_recourse\oracle\verification\verify_stage2_substrate_loopA_loopB2.py") "Loop A/B2 verifier"
# Stage boundary verifier: Loop B2 B-gate contract
Require-File (Join-Path $FinalFreeze "stage1_oracle_backends\alpha\oracle_alpha_params.json") "Stage1 alpha params for Loop B2"

$CanonicalRegistry = Join-Path $FinalFreeze "configs\oracle_backend_registry.yaml"
$LegacyRegistry = Join-Path $FinalFreeze "oracle_backend_registry.json"
if (Test-Path -LiteralPath $CanonicalRegistry -PathType Leaf) {
  Write-Host "Oracle backend registry: $CanonicalRegistry"
} elseif (Test-Path -LiteralPath $LegacyRegistry -PathType Leaf) {
  Write-Host "Oracle backend registry: $LegacyRegistry"
} else {
  Write-Warning "No oracle backend registry found. Patched verifier will fall back to stage1_oracle_backends\alpha\oracle_alpha_params.json."
}

Invoke-Python "Compile source" @("-m", "compileall", "-q", $SrcDir)

$LoopArgs = @(
  "-m", "credit_recourse.oracle.verification.verify_stage2_substrate_loopA_loopB2",
  "--project-root", $ProjectRoot,
  "--sim-business-plan-mode", $SimBusinessPlanMode,
  "--max-rel-err-assets", ([string]$MaxRelErrAssets),
  "--min-coverage-share", ([string]$MinCoverageShare),
  "--magnitude-quantile", ([string]$MagnitudeQuantile),
  "--max-identity-rel-err-assets", ([string]$MaxIdentityRelErrAssets)
)
$LoopArgs += ConvertTo-BoolArg $PreserveCurrentNonCurrentResidual "--preserve-current-non-current-residual"

Invoke-Python "Loop A / Loop B2 Stage2 extension" $LoopArgs -AllowGateFail

$Report = Join-Path $FinalFreeze "stage2_substrate_loopA_loopB2\substrate_loopA_loopB2_report.json"
Require-File $Report "Loop A/B2 report"

Write-Section "Loop A/B2 report summary"
$Meta = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
$Meta | Select-Object status, sim_business_plan_mode, preserve_current_non_current_residual, magnitude_quantile, loopA_rows, loopA_phase_rows, loopA_comparable_error_rows, loopA_dimensions | Format-List
Write-Host "loopA_contract_gate:"
$Meta.loopA_contract_gate | Format-List
Write-Host "loopA_historical_resimulation_stress:"
$Meta.loopA_historical_resimulation_stress | Format-List
Write-Host "loopB2:"
$Meta.loopB2 | Format-List
Write-Host "Report: $Report"
