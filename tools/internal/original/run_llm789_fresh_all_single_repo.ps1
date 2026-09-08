<#
Runs one homogeneous Stage7-9 LLM matrix and archives each cell through the
canonical Python archive writer. This script does not perform paper analysis.
#>
[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [Parameter(Mandatory=$true)][string]$RunRole,
  [string[]]$Backends = @("openai:gpt-5.4-mini"),
  [string[]]$InformationConditions = @("IC-a", "IC-b", "IC-c"),
  [string]$Conditions = "C4,C5,C6,C6X,C7,C8",
  [string]$Modes = "candidate_selection,free_form_10d",
  [int]$Seed = 1,
  [int]$ReferenceDrawSeed = 1,
  [int]$CandidateLibraryQuantile = 50,
  [int]$MaxConcurrency = 6,
  [int]$MaxRetries = 3,
  [double]$RetrySleepSeconds = 20.0,
  [string]$RunLabelPrefix = "paper",
  [string]$DateTag = (Get-Date -Format "yyyyMMdd_HHmmss"),
  [string]$OpenAIApiMode = "",
  [string]$OpenAIReasoningEffort = "",
  [int]$OpenAIMaxOutputTokens = 0,
  [switch]$Fresh
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-Root([string]$Given) {
  if ($Given) { return (Resolve-Path -LiteralPath $Given).Path }
  $candidate = Split-Path -Parent $PSScriptRoot
  if (-not (Test-Path -LiteralPath (Join-Path $candidate "src\credit_recourse") -PathType Container)) {
    throw "Cannot infer ProjectRoot. Pass -ProjectRoot."
  }
  return (Resolve-Path -LiteralPath $candidate).Path
}
function Safe-Token([string]$Value) {
  return (($Value -replace '^openai:','' -replace '^anthropic:','' -replace '[^A-Za-z0-9._-]+','_').Trim('_'))
}
function Remove-ActiveStages([string]$Root) {
  foreach ($name in @("stage7_llm_action_generation", "stage8_llm_multi_oracle_eval", "stage9_llm_rl_comparison", "stage9_statistical_inference")) {
    $path = Join-Path $Root "data\final_freeze\$name"
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
  }
}

$Root = Resolve-Root $ProjectRoot
$Py = if ($env:REPRO_PYTHON_EXE) { $env:REPRO_PYTHON_EXE } else { Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Virtualenv Python missing: $Py" }
Set-Location -LiteralPath $Root
$env:PYTHONPATH = (Join-Path $Root "src") + ";" + $env:PYTHONPATH
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$help = (& $Py -m credit_recourse.utils.run_llm_stages --help 2>&1 | Out-String)
foreach ($flag in @("--run-role", "--run-label", "--checkpoint-path")) {
  if ($help -notmatch [regex]::Escape($flag)) { throw "Current run_llm_stages is missing $flag" }
}

foreach ($backend in $Backends) {
  if ($backend -like "openai:*" -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "OPENAI_API_KEY is not set." }
  if ($backend -like "anthropic:*" -and [string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) { throw "ANTHROPIC_API_KEY is not set." }
  foreach ($ic in $InformationConditions) {
    if ($ic -notin @("IC-a", "IC-b", "IC-c")) { throw "Unsupported information condition: $ic" }
    if ($Fresh) { Remove-ActiveStages $Root }
    $backendToken = Safe-Token $backend
    $icToken = $ic.Replace("-", "")
    $runLabel = "${RunLabelPrefix}_${RunRole}_${icToken}_${backendToken}_seed${Seed}_${DateTag}"
    $archiveDir = Join-Path $Root "data\final_freeze\llm_runs\$runLabel"
    if (Test-Path -LiteralPath $archiveDir) { throw "Immutable archive already exists: $archiveDir" }
    $checkpoint = Join-Path $Root "data\final_freeze\stage7_llm_action_generation\checkpoint_${runLabel}.jsonl"

    $argsList = @(
      "-m", "credit_recourse.utils.run_llm_stages",
      "--project-root", $Root,
      "--backend", $backend,
      "--information-condition", $ic,
      "--conditions", $Conditions,
      "--modes", $Modes,
      "--seed", "$Seed",
      "--reference-draw-seed", "$ReferenceDrawSeed",
      "--candidate-library-quantile", "$CandidateLibraryQuantile",
      "--checkpoint-path", $checkpoint,
      "--run-label", $runLabel,
      "--run-role", $RunRole,
      "--max-concurrency", "$MaxConcurrency",
      "--max-retries", "$MaxRetries",
      "--retry-sleep-seconds", "$RetrySleepSeconds"
    )
    if ($backend -like "openai:*") {
      if ($OpenAIApiMode) { $argsList += @("--openai-api-mode", $OpenAIApiMode) }
      if ($OpenAIReasoningEffort) { $argsList += @("--openai-reasoning-effort", $OpenAIReasoningEffort) }
      if ($OpenAIMaxOutputTokens -gt 0) { $argsList += @("--openai-max-output-tokens", "$OpenAIMaxOutputTokens") }
    }

    Write-Host "`n==== LLM Stage7-9: $runLabel ====" -ForegroundColor Cyan
    Write-Host "CMD> $Py $($argsList -join ' ')" -ForegroundColor DarkGray
    & $Py @argsList
    if ($LASTEXITCODE -ne 0) { throw "LLM run failed: $runLabel" }
    if (-not (Test-Path -LiteralPath (Join-Path $archiveDir "archive_manifest.json"))) {
      throw "Canonical archive manifest missing after run: $archiveDir"
    }
  }
}
Write-Host "`nPASS: homogeneous LLM matrix archived." -ForegroundColor Green
