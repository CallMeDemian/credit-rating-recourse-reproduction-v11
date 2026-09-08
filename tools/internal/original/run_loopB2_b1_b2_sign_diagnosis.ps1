param(
  [string]$ProjectRoot = "",
  [string]$B2Csv = "",
  [string]$OutputDir = "",
  [switch]$FailOnInversionSuspect,
  [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot([string]$ProvidedRoot) {
  if (-not [string]::IsNullOrWhiteSpace($ProvidedRoot)) {
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
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "MISSING: $Label => $Path"
  }
}

function Require-Dir([string]$Path, [string]$Label) {
  if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    throw "MISSING: $Label => $Path"
  }
}

$Root = Resolve-ProjectRoot $ProjectRoot
$SrcDir = Join-Path $Root "src"
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$DefaultOutputDir = Join-Path $Root "data\final_freeze\stage2_substrate_loopA_loopB2"
$ResolvedOutputDir = if ([string]::IsNullOrWhiteSpace($OutputDir)) { $DefaultOutputDir } else { [System.IO.Path]::GetFullPath($OutputDir) }
$ResolvedB2Csv = if ([string]::IsNullOrWhiteSpace($B2Csv)) {
  Join-Path $DefaultOutputDir "loopB2_alpha_predicted_score_vs_real_rating_change.csv"
} else {
  [System.IO.Path]::GetFullPath($B2Csv)
}
$ReportPath = Join-Path $ResolvedOutputDir "loopB2_b1_b2_sign_alignment_report.json"
$ModulePath = Join-Path $SrcDir "credit_recourse\oracle\verification\diagnose_loopb2_b1_alignment.py"

Require-Dir $SrcDir "source directory"
Require-File $ModulePath "Loop B2/B1 alignment diagnostic module"
Require-File $PythonExe "virtualenv Python"

$Args = @(
  "-m", "credit_recourse.oracle.verification.diagnose_loopb2_b1_alignment",
  "--project-root", $Root,
  "--b2-csv", $ResolvedB2Csv,
  "--output-dir", $ResolvedOutputDir
)
if ($FailOnInversionSuspect) {
  $Args += "--fail-on-inversion-suspect"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[LoopB2 vs Stage1 B1 sign/scale alignment diagnosis]"
Write-Host "============================================================"
Write-Host ("CMD: " + $PythonExe + " " + ($Args -join " "))

if ($PlanOnly) {
  Write-Host "PlanOnly: no diagnostic input was read and no output was written."
  exit 0
}

Require-File $ResolvedB2Csv "Loop B2 score/rating transition CSV"
Require-Dir $ResolvedOutputDir "Loop B2 output directory"

$env:PYTHONPATH = $SrcDir
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $PythonExe @Args
$Code = $LASTEXITCODE
if ($Code -ne 0) {
  throw "Loop B2/B1 sign alignment diagnostic failed with exit code $Code."
}

Require-File $ReportPath "loopB2_b1_b2_sign_alignment_report.json"
$Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
if ([string]$Report.status -ne "PASS") {
  throw "Loop B2/B1 sign alignment report status is not PASS: $($Report.status)"
}
if ($Report.does_not_change_gate_verdict -ne $true) {
  throw "Loop B2/B1 sign alignment report must preserve diagnostic-only semantics (does_not_change_gate_verdict=true)."
}

Write-Host "PASS: diagnostic-only Loop B2/B1 sign alignment report verified."
Write-Host "Report: $ReportPath"
