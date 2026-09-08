<# Canonical N5 generation-time L1-budget matrix runner. #>
[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$Backend = "openai:gpt-5.4-mini",
  [string[]]$InformationConditions = @("IC-a", "IC-b", "IC-c"),
  [string]$RunRole = "paper_n5_l1_1p27",
  [int]$Seed = 1,
  [int]$ReferenceDrawSeed = 1,
  [double]$L1Budget = 1.27,
  [int]$CandidateLibraryQuantile = 50,
  [int]$MaxConcurrency = 6,
  [int]$MaxRetries = 3,
  [double]$RetrySleepSeconds = 20.0,
  [string]$DateTag = (Get-Date -Format "yyyyMMdd_HHmmss")
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Resolve-Root([string]$Given) {
  if ($Given) { return (Resolve-Path -LiteralPath $Given).Path }
  return (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
}
$Root = Resolve-Root $ProjectRoot
$Py = if ($env:REPRO_PYTHON_EXE) { $env:REPRO_PYTHON_EXE } else { Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Virtualenv Python missing: $Py" }
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "OPENAI_API_KEY is not set." }
Set-Location -LiteralPath $Root
$env:PYTHONPATH = (Join-Path $Root "src") + ";" + $env:PYTHONPATH
$help = (& $Py -m credit_recourse.utils.run_llm_stages --help 2>&1 | Out-String)
foreach ($flag in @("--freeform-l1-budget", "--budgeted-conditions", "--run-role")) {
  if ($help -notmatch [regex]::Escape($flag)) { throw "N5 contract flag missing: $flag" }
}
& $Py -m credit_recourse.verification.smoke_stage7_n5_budget_contract --json
if ($LASTEXITCODE -ne 0) { throw "N5 synthetic budget-contract verifier failed." }

foreach ($ic in $InformationConditions) {
  if ($ic -notin @("IC-a", "IC-b", "IC-c")) { throw "Unsupported IC: $ic" }
  foreach ($name in @("stage7_llm_action_generation", "stage8_llm_multi_oracle_eval", "stage9_llm_rl_comparison", "stage9_statistical_inference")) {
    $active = Join-Path $Root "data\final_freeze\$name"
    if (Test-Path -LiteralPath $active) { Remove-Item -LiteralPath $active -Recurse -Force }
  }
  $icToken = $ic.Replace("-", "")
  $budgetToken = ("{0:0.00}" -f $L1Budget).Replace(".", "p")
  $runLabel = "N5_C6_L1_${budgetToken}_${icToken}_gpt54mini_p${CandidateLibraryQuantile}_main_seed${Seed}_${DateTag}"
  $archiveDir = Join-Path $Root "data\final_freeze\llm_runs\$runLabel"
  if (Test-Path -LiteralPath $archiveDir) { throw "Immutable N5 archive exists: $archiveDir" }
  $checkpoint = Join-Path $Root "data\final_freeze\stage7_llm_action_generation\checkpoint_${runLabel}.jsonl"
  $argsList = @(
    "-m", "credit_recourse.utils.run_llm_stages",
    "--project-root", $Root,
    "--backend", $Backend,
    "--information-condition", $ic,
    "--conditions", "C4,C6",
    "--modes", "free_form_10d",
    "--seed", "$Seed",
    "--reference-draw-seed", "$ReferenceDrawSeed",
    "--candidate-library-quantile", "$CandidateLibraryQuantile",
    "--openai-api-mode", "responses",
    "--openai-reasoning-effort", "low",
    "--openai-max-output-tokens", "1200",
    "--freeform-l1-budget", "$L1Budget",
    "--budgeted-conditions", "C6",
    "--budget-contract-label", "N5_C6_L1_${budgetToken}_${icToken}",
    "--checkpoint-path", $checkpoint,
    "--run-label", $runLabel,
    "--run-role", $RunRole,
    "--max-concurrency", "$MaxConcurrency",
    "--max-retries", "$MaxRetries",
    "--retry-sleep-seconds", "$RetrySleepSeconds"
  )
  Write-Host "`n==== N5 Stage7-9: $runLabel ====" -ForegroundColor Cyan
  Write-Host "CMD> $Py $($argsList -join ' ')" -ForegroundColor DarkGray
  & $Py @argsList
  if ($LASTEXITCODE -ne 0) { throw "N5 failed: $runLabel" }
  if (-not (Test-Path -LiteralPath (Join-Path $archiveDir "archive_manifest.json"))) { throw "N5 archive missing: $archiveDir" }
}
Write-Host "`nPASS: N5 IC-a/b/c archived." -ForegroundColor Green
