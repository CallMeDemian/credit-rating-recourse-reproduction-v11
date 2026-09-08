# Universal Oracle Stage0/Stage1 runner for thesis_repo final-freeze infra.
# Supports full redevelopment and bounded partial reruns.
# Stage order:
#   stage0 -> stage00_01 -> stage00_02 -> stage00_03 -> stage00_04
#          -> backend_alpha -> backend_beta -> backend_gamma
#          -> bridge -> diagnostics -> substrate_validation
# Default: resume-safe full run. Use -ForceSelectedRange to delete outputs in the selected window first.

param(
  [string]$ProjectRoot = "",
  [string]$PythonExe = "",
  [string]$RawAllDir = "",
  [string]$RawRatingDir = "",

  [ValidateSet("stage0","stage00_01","stage00_02","stage00_03","stage00_04","backend_alpha","backend_beta","backend_gamma","bridge","diagnostics","substrate_validation")]
  [string]$StartAt = "stage0",

  [ValidateSet("stage0","stage00_01","stage00_02","stage00_03","stage00_04","backend_alpha","backend_beta","backend_gamma","bridge","diagnostics","substrate_validation")]
  [string]$StopAfter = "substrate_validation",

  [ValidateSet("","stage0","stage00_01","stage00_02","stage00_03","stage00_04","backend_alpha","backend_beta","backend_gamma","bridge","diagnostics","substrate_validation")]
  [string]$Only = "",

  [switch]$ForceSelectedRange,
  [switch]$CleanStage0,
  [switch]$CleanStage1All,
  [switch]$CleanBackendsOnly,
  [switch]$ReuseStage1Inputs,
  [switch]$RunVerifiers,
  [switch]$SkipCompile,
  [switch]$SkipNoRegression,

  [int]$ScoreEndYear = 2023,
  [int]$DevStartYear = 2002,
  [int]$DevEndYear = 2019,
  [int]$OotStartYear = 2020
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$StageOrder = @("stage0","stage00_01","stage00_02","stage00_03","stage00_04","backend_alpha","backend_beta","backend_gamma","bridge","diagnostics","substrate_validation")

function Get-StageIndex([string]$Stage) {
  $idx = [Array]::IndexOf($StageOrder, $Stage)
  if ($idx -lt 0) { throw "Unknown stage: $Stage" }
  return $idx
}

if (-not [string]::IsNullOrWhiteSpace($Only)) {
  $StartAt = $Only
  $StopAfter = $Only
}
$StartIndex = Get-StageIndex $StartAt
$StopIndex = Get-StageIndex $StopAfter
if ($StartIndex -gt $StopIndex) { throw "Invalid range: -StartAt $StartAt is after -StopAfter $StopAfter" }

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
  $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Virtualenv Python missing: $PythonExe" }
$Py = $PythonExe
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $Root

$Final = Join-Path $Root "data\final_freeze"
$LogDir = Join-Path $Final "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ("run_oracle_stage0_stage1_universal_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Section([string]$Name) {
  $msg = "`n============================================================`n[$Name]`n============================================================"
  Write-Host $msg
  Add-Content -Path $Log -Value $msg
}

function Invoke-Step {
  param([string]$Name, [string[]]$Cmd)
  Write-Section $Name
  $cmdLine = "python " + ($Cmd -join " ")
  Write-Host $cmdLine
  Add-Content -Path $Log -Value $cmdLine
  & $Py @Cmd
  $exitCode = $LASTEXITCODE
  Add-Content -Path $Log -Value ("[exit_code] {0}" -f $exitCode)
  if ($exitCode -ne 0) { throw "FAILED: $Name exit=$exitCode" }
}

function Test-FileExists([string]$Path) { return (Test-Path -LiteralPath $Path -PathType Leaf) }

function Test-JsonPass([string]$Path) {
  if (-not (Test-FileExists $Path)) { return $false }
  try {
    $j = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $j.status) { return $false }
    return ($j.status -like "PASS*")
  } catch { return $false }
}

function Test-Stage0Ready {
  $S0 = Join-Path $Final "stage0_oracle_foundation"
  $Required = @(
    (Join-Path $S0 "canonical_panel\stage0_canonical_panel.parquet"),
    (Join-Path $S0 "canonical_panel\statement_items_panel.parquet"),
    (Join-Path $S0 "stage0_manifest.json")
  )
  foreach ($p in $Required) { if (-not (Test-FileExists $p)) { return $false } }
  return (Test-JsonPass (Join-Path $S0 "stage0_manifest.json"))
}

function Test-Stage1Ready {
  $Required = @(
    (Join-Path $Final "configs\oracle_backend_registry.yaml"),
    (Join-Path $Final "stage1_oracle_inputs\alpha_vanilla_input_candidate.parquet"),
    (Join-Path $Final "stage1_oracle_backends\alpha\oracle_alpha_params.json"),
    (Join-Path $Final "stage1_oracle_backends\alpha\oracle_firm_year_output_alpha.parquet"),
    (Join-Path $Final "stage1_oracle_backends\beta\benchmark_beta_params.json"),
    (Join-Path $Final "stage1_oracle_backends\beta\benchmark_firm_year_output_beta.parquet"),
    (Join-Path $Final "stage1_oracle_backends\gamma\benchmark_gamma_params.json"),
    (Join-Path $Final "stage1_oracle_backends\gamma\benchmark_gamma_model.joblib"),
    (Join-Path $Final "stage1_oracle_backends\gamma\benchmark_firm_year_output_gamma.parquet"),
    (Join-Path $Final "ledgers\stage1_oracle_backends_full_development.json"),
    (Join-Path $Final "ledgers\oracle_backend_diagnostic_report.json"),
    (Join-Path $Final "ledgers\stage1_substrate_validation_loopB1.json")
  )
  foreach ($p in $Required) { if (-not (Test-FileExists $p)) { return $false } }
  if (-not (Test-JsonPass (Join-Path $Final "ledgers\stage1_oracle_backends_full_development.json"))) { return $false }
  if (-not (Test-JsonPass (Join-Path $Final "ledgers\oracle_backend_diagnostic_report.json"))) { return $false }
  if (-not (Test-JsonPass (Join-Path $Final "ledgers\stage1_substrate_validation_loopB1.json"))) { return $false }
  return $true
}

function Get-StageOutputPath([string]$Stage) {
  switch ($Stage) {
    "stage0" { return (Join-Path $Final "stage0_oracle_foundation") }
    "stage00_01" { return (Join-Path $Final "stage1_oracle_inputs\stage00_01_rating_statement_integration") }
    "stage00_02" { return (Join-Path $Final "stage1_oracle_inputs\stage00_02_financial_ratio_engineering") }
    "stage00_03" { return (Join-Path $Final "stage1_oracle_inputs\stage00_03_nonfinancial_metadata") }
    "stage00_04" { return (Join-Path $Final "stage1_oracle_inputs\stage00_04_variable_selection") }
    "backend_alpha" { return (Join-Path $Final "stage1_oracle_backends\alpha") }
    "backend_beta" { return (Join-Path $Final "stage1_oracle_backends\beta") }
    "backend_gamma" { return (Join-Path $Final "stage1_oracle_backends\gamma") }
    "bridge" { return (Join-Path $Final "stage1_oracle_inputs\alpha_vanilla_input_candidate.parquet") }
    "diagnostics" { return (Join-Path $Final "ledgers\oracle_backend_diagnostic_report.json") }
    "substrate_validation" { return (Join-Path $Final "ledgers\stage1_substrate_validation_loopB1.json") }
    default { throw "Unknown stage: $Stage" }
  }
}

function Remove-PathIfExists([string]$Path) {
  if (Test-Path -LiteralPath $Path) {
    Write-Host "[REMOVE] $Path"
    Add-Content -Path $Log -Value "[REMOVE] $Path"
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

function Clear-SelectedRangeOutputs {
  Write-Section "Force selected Oracle range"
  for ($i = $StartIndex; $i -le $StopIndex; $i++) {
    $stage = $StageOrder[$i]
    if ($stage -eq "stage0") {
      if (-not $CleanStage0) { Write-Host "[KEEP] stage0 requires explicit -CleanStage0 to delete protected foundation artifacts." }
      continue
    }
    $path = Get-StageOutputPath $stage
    Remove-PathIfExists $path
    if ($stage -eq "bridge") { Remove-PathIfExists (Join-Path $Final "stage1_oracle_inputs\alpha_vanilla_input_candidate_metadata.json") }
    if ($stage -eq "diagnostics") { Remove-PathIfExists (Join-Path $Final "ledgers\oracle_backend_gate_summary.csv") }
  }
}

Write-Section "Oracle preflight"
Write-Host "ProjectRoot=$Root"
Write-Host "Python=$Py"
Write-Host "PYTHONPATH=$env:PYTHONPATH"
Write-Host "StageRange=$StartAt -> $StopAfter"
Write-Host "Log=$Log"

if (-not $SkipCompile) {
  Invoke-Step "Compile source" @("-m", "compileall", "-q", (Join-Path $Root "src"))
}
Invoke-Step "Materialize final-freeze configs" @("-m", "credit_recourse.utils.materialize_final_freeze_configs", "--project-root", $Root, "--overwrite")
if (-not $SkipNoRegression) {
  Invoke-Step "No-regression contract" @("-m", "credit_recourse.verification.verify_final_no_regression_contract", "--project-root", $Root)
}

if ($ForceSelectedRange) { Clear-SelectedRangeOutputs }

$IncludesStage0 = ($StartIndex -le (Get-StageIndex "stage0") -and $StopIndex -ge (Get-StageIndex "stage0"))
$IncludesStage1 = ($StopIndex -ge (Get-StageIndex "stage00_01"))

if ($IncludesStage0) {
  if ((-not $CleanStage0) -and (Test-Stage0Ready)) {
    Write-Section "Stage0 Oracle foundation"
    Write-Host "[SKIP] Stage0 artifacts already exist and manifest is PASS. Use -CleanStage0 to rebuild."
  } else {
    $Args = @("-m", "credit_recourse.oracle.stage0.build_stage0_foundation_from_raw", "--project-root", $Root)
    if (-not [string]::IsNullOrWhiteSpace($RawAllDir)) { $Args += @("--raw-all-dir", $RawAllDir) }
    if (-not [string]::IsNullOrWhiteSpace($RawRatingDir)) { $Args += @("--raw-rating-dir", $RawRatingDir) }
    if ($CleanStage0) { $Args += "--clean" }
    Invoke-Step "Stage0 Oracle foundation" $Args
  }
  if ($RunVerifiers) {
    Invoke-Step "Verify Stage0" @("-m", "credit_recourse.verification.stage_boundary_contracts", "--project-root", $Root, "--stage", "stage0")
  }
}

if ($StopAfter -eq "stage0") {
  Write-Section "DONE"
  Write-Host "Oracle run log: $Log"
  return
}

if ($IncludesStage1) {
  $Stage1Start = $StartAt
  if ($Stage1Start -eq "stage0") { $Stage1Start = "stage00_01" }
  $Stage1End = $StopAfter
  if ($Stage1End -eq "stage0") { throw "Internal range error: Stage1 requested but StopAfter is stage0" }

  if (($Stage1Start -eq "stage00_01") -and ($Stage1End -eq "substrate_validation") -and (-not $CleanStage1All) -and (-not $CleanBackendsOnly) -and (-not $ForceSelectedRange) -and (Test-Stage1Ready)) {
    Write-Section "Stage1 Oracle development"
    Write-Host "[SKIP] Full Stage1 artifacts already exist and ledgers are PASS. Use -ForceSelectedRange or -CleanStage1All to rebuild."
  } else {
    $Args = @(
      "-m", "credit_recourse.oracle.stage1.run_stage1_oracle_development",
      "--project-root", $Root,
      "--score-end-year", "$ScoreEndYear",
      "--dev-start-year", "$DevStartYear",
      "--dev-end-year", "$DevEndYear",
      "--oot-start-year", "$OotStartYear",
      "--start-step", $Stage1Start,
      "--end-step", $Stage1End,
      "--resume"
    )
    if ($CleanStage1All) { $Args += "--clean" }
    if ($ReuseStage1Inputs) { $Args += "--reuse-stage1-inputs" }
    if ($CleanBackendsOnly) { $Args += "--clean-backends-only" }
    if (-not [string]::IsNullOrWhiteSpace($RawRatingDir)) { $Args += @("--raw-rating-dir", $RawRatingDir) }
    Invoke-Step "Stage1 Oracle development [$Stage1Start -> $Stage1End]" $Args
  }

  if ($RunVerifiers) {
    if ($Stage1End -eq "substrate_validation") {
      Invoke-Step "Verify Stage1 substrate" @("-m", "credit_recourse.oracle.verification.verify_stage1_substrate_validation", "--project-root", $Root)
      Invoke-Step "Verify Stage1 bridge" @("-m", "credit_recourse.verification.stage_boundary_contracts", "--project-root", $Root, "--stage", "stage1_bridge")
      Invoke-Step "Verify Stage1 backends" @("-m", "credit_recourse.verification.stage_boundary_contracts", "--project-root", $Root, "--stage", "stage1")
    } else {
      Write-Section "Partial verifier note"
      Write-Host "[SKIP] Full Stage1 boundary verifiers require StopAfter=substrate_validation. Current StopAfter=$Stage1End"
    }
  }
}

Write-Section "DONE"
Write-Host "Oracle run log: $Log"
