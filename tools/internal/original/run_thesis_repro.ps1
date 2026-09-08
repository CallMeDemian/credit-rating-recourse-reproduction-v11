<#
Canonical fresh-workspace reproduction entry point.

Workspace before first run:
  src\
  tools\
  data\raw\raw_all\
  data\raw\raw_nonfinancial\...
  data\raw\rating_sample\

Run in order:
  -Task Oracle
  -Task RL
  -Task LLM -ConfirmLiveApiSpend
  -Task Analysis -ReplaceAnalysisOutput
  -Task RemainingNoApi
  -Task BudgetFrontier -ConfirmLiveApiSpend
  -Task RemainingAll -ConfirmLiveApiSpend
#>
[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [ValidateSet("Oracle", "RL", "LLM", "Analysis", "AnalysisResume", "RemainingNoApi", "BudgetFrontier", "RemainingAll")]
  [string]$Task = "Oracle",
  [switch]$ConfirmLiveApiSpend,
  [switch]$PromptApiKeys,
  [switch]$ReplaceAnalysisOutput,
  [string]$FailedAnalysisPath = "",
  [string]$BudgetFrontierDateTag = "",
  [switch]$MatchedBudgetC4,
  [switch]$C4RMatched,
  [string]$ReplicationGroupId = "",
  [string]$ExperimentBackend = "openai:gpt-5.4-mini",
  [string]$ExperimentBackendLabel = "",
  [int]$ExperimentSeed = 1,
  [int]$ExperimentReferenceDrawSeed = 1,
  [switch]$RunN5MAxisSwap,
  [string]$AxisSwapArm0p75 = "",
  [string]$AxisSwapArm1p27 = "",
  [string]$AxisSwapArm2p00 = "",
  [string]$AxisSwapArmUnbounded = "",
  [int]$AxisSwapShapleyPermutations = 64,
  [int]$AxisSwapSeed = 20260714,
  [switch]$AxisSwapOatOnly,
  [switch]$PlanOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-Root([string]$Given) {
  if ($Given) { return (Resolve-Path -LiteralPath $Given).Path }
  $candidate = Split-Path -Parent $PSScriptRoot
  if (-not (Test-Path -LiteralPath (Join-Path $candidate "src\credit_recourse") -PathType Container)) { throw "Cannot infer ProjectRoot." }
  return (Resolve-Path -LiteralPath $candidate).Path
}
function Invoke-Checked([string]$Label, [string[]]$Command) {
  Write-Host "`n============================================================" -ForegroundColor Cyan
  Write-Host "[$Label]" -ForegroundColor Cyan
  Write-Host "CMD> $($Command -join ' ')" -ForegroundColor DarkGray
  & $Command[0] @($Command[1..($Command.Count - 1)])
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $Label exit=$LASTEXITCODE" }
}
function Read-Secret([string]$Prompt) {
  $secure = Read-Host $Prompt -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
  finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

$Root = Resolve-Root $ProjectRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Run tools\setup_env.ps1 first. Missing: $Py" }
Set-Location -LiteralPath $Root
$env:PYTHONPATH = (Join-Path $Root "src") + ";" + $env:PYTHONPATH
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$Tools = Join-Path $Root "tools"

function Invoke-FrontierPaperAssetRefresh {
  $analysisDir = Join-Path $Root "data\analysis\paper_repro"
  $analysisManifest = Join-Path $analysisDir "00_manifest\paper_repro_analysis_manifest.json"
  if (-not (Test-Path -LiteralPath $analysisManifest -PathType Leaf)) {
    throw "Budget frontier inference completed, but canonical paper analysis is missing: $analysisManifest"
  }
  $analysisStatus = [string](Get-Content -LiteralPath $analysisManifest -Raw -Encoding UTF8 | ConvertFrom-Json).status
  if ($analysisStatus -ne "PASS") {
    throw "Budget frontier paper-asset refresh requires canonical analysis status PASS; found $analysisStatus"
  }
  Invoke-Checked "refresh paper assets after generation-time frontier" @(
    $Py, "-m", "credit_recourse.analysis.paper_repro_assets",
    "--analysis-dir", $analysisDir,
    "--strict"
  )
  Invoke-Checked "refresh final paper-analysis output contract" @(
    $Py, "-m", "credit_recourse.verification.verify_paper_repro_output_contract",
    "--project-root", $Root,
    "--analysis-dir", $analysisDir,
    "--out-json", (Join-Path $analysisDir "99_verification\verify_paper_repro_output_contract.json")
  )
}

# Oracle/RL/LLM reproduction starts from raw data. Post-freeze Analysis does not:
# it checks only the files directly consumed by the analysis and starts immediately.
if ($Task -in @("RemainingNoApi", "BudgetFrontier", "RemainingAll")) {
  Invoke-Checked "remaining-analysis implementation contract" @(
    $Py, "-m", "credit_recourse.verification.verify_remaining_thesis_analyses_contract",
    "--project-root", $Root
  )
}

if ($Task -notin @("Analysis", "AnalysisResume", "RemainingNoApi", "BudgetFrontier", "RemainingAll")) {
  Invoke-Checked "workspace/raw contract" @(
    $Py, "-m", "credit_recourse.verification.verify_reproduction_workspace",
    "--project-root", $Root
  )
}

switch ($Task) {
  "Oracle" {
    Invoke-Checked "Oracle Stage0-1 and verifiers" @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_oracle_stage0_stage1.ps1"),
      "-ProjectRoot", $Root,
      "-PythonExe", $Py,
      "-RunVerifiers"
    )
  }
  "RL" {
    # The substrate LoopA/B2 extension is a required paper artifact and is frozen before RL.
    Invoke-Checked "LoopA/B2 substrate extension" @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_loopA_loopB2_stage2_extension.ps1"),
      "-ProjectRoot", $Root,
      "-SimBusinessPlanMode", "calibrated",
      "-MagnitudeQuantile", "50"
    )
    $loopReport = Join-Path $Root "data\final_freeze\stage2_substrate_loopA_loopB2\substrate_loopA_loopB2_report.json"
    if (-not (Test-Path -LiteralPath $loopReport)) { throw "LoopA/B2 report missing: $loopReport" }
    $loopStatus = (Get-Content -LiteralPath $loopReport -Raw -Encoding UTF8 | ConvertFrom-Json).status
    if ($loopStatus -ne "PASS") { throw "LoopA/B2 report status=$loopStatus. Canonical RL reproduction requires PASS." }
    $cashFlowPanel = Join-Path $Root "data\final_freeze\stage1_oracle_inputs\stage00_01_rating_statement_integration\cleaned_statement_panels\현금흐름표_clean.parquet"
    $cmd = @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_rl_unified_stage3456.ps1"),
      "-ProjectRoot", $Root,
      "-RunLabel", "sector0_m0p05_f0p45_liq0p45_s2_s3train_g0p60_tau0p85_b15_cql0_distill1p5_qpareto",
      "-RunStage2", "-Stage2Rho", "0.0",
      "-Stage2MertonLambda", "0.05", "-Stage2FcffLambda", "0.45", "-Stage2LiquidityLambda", "0.45",
      "-JoinCashFlowSubstrate", "-CashFlowEncoderMode", "reward_only", "-CashFlowPanel", $cashFlowPanel,
      "-CounterfactualTransitions", "-CounterfactualRewardMode", "phi_merton_fcff_liquidity",
      "-CounterfactualDoneMode", "terminal", "-CounterfactualFidelityGate", "warn",
      "-CounterfactualMaxRelErrAssets", "0.05", "-SimBusinessPlanMode", "calibrated",
      "-PreserveCurrentNonCurrentResidual",
      "-Stage3Mode", "train", "-Stage3BatchSize", "512", "-Stage3Epochs", "30", "-Stage3Seed", "2",
      "-Stage3MaskingRatio", "0.15", "-Stage3LearningRate", "3e-4", "-Stage3WeightDecay", "1e-5",
      "-MagnitudeQuantile", "50", "-Stage4Epochs", "80", "-Stage4BatchSize", "512", "-Stage4Seed", "2",
      "-FamilyBalanced", "-FamilyBalancePower", "0.5", "-FamilyWeightCap", "3.0", "-CombinedWeightCap", "5.0",
      "-Stage5Epochs", "80", "-Stage5BatchSize", "128", "-Stage5Seed", "2",
      "-Stage5LearningRate", "1e-4", "-Stage5WeightDecay", "2e-3",
      "-Gamma", "0.60", "-ExpectileTau", "0.85", "-Beta", "15.0", "-CqlAlpha", "0.0",
      "-DistillLambda", "1.5", "-DistillMarginMin", "0.01", "-DistillTemperature", "1.0",
      "-Stage5CriticHeadArch", "cross_attention", "-CrossAttnBlocks", "2", "-CrossAttnHeads", "4", "-CrossAttnDropout", "0.25",
      "-ActorExtractionMode", "awr", "-ActorHeadArch", "linear", "-Stage5SelectionMetric", "q_pareto_knee",
      "-VerifierMode", "warn", "-ZipArchive"
    )
    Invoke-Checked "RL Stage2-6 final-paper preset" $cmd
  }
  "LLM" {
    if (-not $ConfirmLiveApiSpend) { throw "LLM task spends live API calls. Re-run with -ConfirmLiveApiSpend." }
    if ($PromptApiKeys) {
      if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { $env:OPENAI_API_KEY = Read-Secret "OPENAI_API_KEY" }
      if ([string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) { $env:ANTHROPIC_API_KEY = Read-Secret "ANTHROPIC_API_KEY" }
    }
    if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "OPENAI_API_KEY is required." }
    if ([string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) { throw "ANTHROPIC_API_KEY is required for the supplementary paper profile." }
    $tag = Get-Date -Format "yyyyMMdd_HHmmss"
    $matrix = Join-Path $Tools "run_llm789_fresh_all_single_repo.ps1"
    foreach ($ic in @("IC-a", "IC-b", "IC-c")) {
      Invoke-Checked "Primary GPT-5.4-mini $ic" @(
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $matrix,
        "-ProjectRoot", $Root, "-RunRole", "paper_primary_gpt54",
        "-Backends", "openai:gpt-5.4-mini", "-InformationConditions", $ic,
        "-RunLabelPrefix", "paper", "-DateTag", $tag,
        "-OpenAIApiMode", "responses", "-OpenAIReasoningEffort", "low", "-OpenAIMaxOutputTokens", "1200", "-Fresh"
      )
    }
    Invoke-Checked "Supplementary GPT-4.1-mini IC-b" @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $matrix,
      "-ProjectRoot", $Root, "-RunRole", "paper_supplementary_gpt41",
      "-Backends", "openai:gpt-4.1-mini", "-InformationConditions", "IC-b",
      "-RunLabelPrefix", "paper", "-DateTag", $tag, "-Fresh"
    )
    Invoke-Checked "Supplementary Haiku 4.5 IC-b" @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $matrix,
      "-ProjectRoot", $Root, "-RunRole", "paper_supplementary_haiku45",
      "-Backends", "anthropic:claude-haiku-4-5-20251001", "-InformationConditions", "IC-b",
      "-RunLabelPrefix", "paper", "-DateTag", $tag, "-Fresh"
    )
    Invoke-Checked "N5 L1=1.27 IC-a/b/c" @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_n5_main_gpt54mini_icab.ps1"),
      "-ProjectRoot", $Root, "-DateTag", $tag
    )
    $frontierCmd = @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_n5_budget_frontier_icb.ps1"),
      "-ProjectRoot", $Root, "-DateTag", $tag
    )
    # Canonical paper frontier is the matched N5M C4/C6 design.
    $frontierCmd += "-BudgetC4"
    Invoke-Checked "N5M IC-b matched generation-time budget frontier" $frontierCmd
    $probeLabel = "paper_paper_icc_probe_ICc_gpt54mini_seed1_$tag"
    Invoke-Checked "IC-c probe-only archive" @(
      $Py, "-m", "credit_recourse.rl.pipelines.final_stage7_llm_action_generation.icc_probe_runner",
      "--project-root", $Root,
      "--run-label", $probeLabel,
      "--run-role", "paper_icc_probe",
      "--base-run-role", "paper_primary_gpt54",
      "--icc-probe-tolerance", "0.20",
      "--max-retries", "3", "--retry-sleep-seconds", "20"
    )
  }
  "Analysis" {
    $cmd = @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_postfreeze_analysis.ps1"),
      "-ProjectRoot", $Root
    )
    if ($ReplaceAnalysisOutput) { $cmd += "-ReplaceOutput" }
    if ($RunN5MAxisSwap) {
      $cmd += @("-RunN5MAxisSwap", "-AxisSwapShapleyPermutations", "$AxisSwapShapleyPermutations", "-AxisSwapSeed", "$AxisSwapSeed")
      if ($AxisSwapOatOnly) { $cmd += "-AxisSwapOatOnly" }
      if ($AxisSwapArm0p75) { $cmd += @("-AxisSwapArm0p75", $AxisSwapArm0p75) }
      if ($AxisSwapArm1p27) { $cmd += @("-AxisSwapArm1p27", $AxisSwapArm1p27) }
      if ($AxisSwapArm2p00) { $cmd += @("-AxisSwapArm2p00", $AxisSwapArm2p00) }
      if ($AxisSwapArmUnbounded) { $cmd += @("-AxisSwapArmUnbounded", $AxisSwapArmUnbounded) }
    }
    if ($PlanOnly) { $cmd += "-PlanOnly" }
    Invoke-Checked "all paper post-freeze analyses" $cmd
  }
  "AnalysisResume" {
    if ($PlanOnly) { throw "AnalysisResume does not support -PlanOnly; it validates and resumes an existing failed archive." }
    if ([string]::IsNullOrWhiteSpace($FailedAnalysisPath)) {
      $canonicalPartial = Join-Path $Root "data\analysis\paper_repro"
      $canonicalManifest = Join-Path $canonicalPartial "00_manifest\paper_repro_analysis_manifest.json"
      $canonicalFrontier = Join-Path $canonicalPartial "03_output_contract_diagnostics\budget_frontier\metadata.json"
      $canonicalFrontierStatus = Join-Path $canonicalPartial "03_output_contract_diagnostics\budget_frontier\frontier_grid_status.csv"
      $useCanonicalPartial = $false
      if (
        (Test-Path -LiteralPath $canonicalManifest -PathType Leaf) -and
        (Test-Path -LiteralPath $canonicalFrontier -PathType Leaf) -and
        (Test-Path -LiteralPath $canonicalFrontierStatus -PathType Leaf)
      ) {
        $canonicalStatus = [string](Get-Content -LiteralPath $canonicalManifest -Raw -Encoding UTF8 | ConvertFrom-Json).status
        $useCanonicalPartial = $canonicalStatus -in @("FAIL", "RUNNING")
      }
      if ($useCanonicalPartial) {
        $FailedAnalysisPath = $canonicalPartial
        Write-Host "Auto-selected canonical partial analysis for in-place resume:" -ForegroundColor Yellow
        Write-Host "  $FailedAnalysisPath"
      } else {
        $failedRoot = Join-Path $Root "data\archive\paper_repro_rebuilds"
        $candidates = @(
          Get-ChildItem -LiteralPath $failedRoot -Directory -Filter "failed_*" -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime -Descending
        )
        if ($candidates.Count -eq 0) {
          throw "No resumable canonical partial output or failed_* analysis archive was found."
        }
        $FailedAnalysisPath = $candidates[0].FullName
        Write-Host "Auto-selected latest failed analysis archive:" -ForegroundColor Yellow
        Write-Host "  $FailedAnalysisPath"
      }
    }
    Invoke-Checked "resume failed paper post-freeze analysis" @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_postfreeze_analysis.ps1"),
      "-ProjectRoot", $Root,
      "-ResumeFailedAnalysisPath", $FailedAnalysisPath
    )
  }
  "RemainingNoApi" {
    $cmd = @(
      $Py, "-m", "credit_recourse.analysis.remaining_thesis_analyses",
      "--project-root", $Root,
      "--mode", "noapi"
    )
    if ($PlanOnly) { $cmd += "--plan-only" }
    Invoke-Checked "remaining no-API thesis analyses" $cmd
  }
  "BudgetFrontier" {
    if (-not $PlanOnly -and -not $ConfirmLiveApiSpend) { throw "BudgetFrontier spends live OpenAI API calls. Re-run with -ConfirmLiveApiSpend." }
    if (-not $PlanOnly -and $PromptApiKeys -and $ExperimentBackend -match "^openai:" -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
      $env:OPENAI_API_KEY = Read-Secret "OPENAI_API_KEY"
    }
    if (-not $PlanOnly -and $PromptApiKeys -and $ExperimentBackend -match "^anthropic:" -and [string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) {
      $env:ANTHROPIC_API_KEY = Read-Secret "ANTHROPIC_API_KEY"
    }
    if (-not $PlanOnly -and $ExperimentBackend -match "^openai:" -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "OPENAI_API_KEY is required." }
    if (-not $PlanOnly -and $ExperimentBackend -match "^anthropic:" -and [string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) { throw "ANTHROPIC_API_KEY is required." }
    $tag = if ([string]::IsNullOrWhiteSpace($BudgetFrontierDateTag)) { Get-Date -Format "yyyyMMdd_HHmmss" } else { $BudgetFrontierDateTag }
    if ($tag -notmatch "^[0-9]{8}_[0-9]{6}$") { throw "BudgetFrontierDateTag must match yyyyMMdd_HHmmss; got $tag" }
    if ($C4RMatched -and $MatchedBudgetC4) { throw "-C4RMatched and -MatchedBudgetC4 are mutually exclusive." }
    if ($C4RMatched -and -not [string]::IsNullOrWhiteSpace($ReplicationGroupId)) { throw "C4R matched run cannot be a replication group." }
    if (-not [string]::IsNullOrWhiteSpace($ReplicationGroupId) -and -not $MatchedBudgetC4) { throw "N5M replication requires -MatchedBudgetC4." }
    $frontierCmd = @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_n5_budget_frontier_icb.ps1"),
      "-ProjectRoot", $Root,
      "-DateTag", $tag,
      "-Backend", $ExperimentBackend,
      "-Seed", "$ExperimentSeed",
      "-ReferenceDrawSeed", "$ExperimentReferenceDrawSeed"
    )
    if ($ExperimentBackendLabel) { $frontierCmd += @("-BackendLabel", $ExperimentBackendLabel) }
    if ($C4RMatched) {
      $frontierCmd += "-C4RMatched"
    } elseif (-not [string]::IsNullOrWhiteSpace($ReplicationGroupId)) {
      $frontierCmd += @("-BudgetC4", "-ExperimentClass", "replication", "-ReplicationGroupId", $ReplicationGroupId)
    } elseif ($MatchedBudgetC4) {
      $frontierCmd += "-BudgetC4"
    }
    if ($PlanOnly) { $frontierCmd += "-PlanOnly" }
    $label = if ($C4RMatched) { "IC-b C4/C4R/C6 matched experiment" } elseif ($ReplicationGroupId) { "IC-b N5M replication group $ReplicationGroupId" } else { "IC-b C4/C6 generation-time budget frontier" }
    Invoke-Checked $label $frontierCmd
    if (-not $C4RMatched -and [string]::IsNullOrWhiteSpace($ReplicationGroupId)) {
      $analysisCmd = @(
        $Py, "-m", "credit_recourse.analysis.remaining_thesis_analyses",
        "--project-root", $Root,
        "--mode", "frontier",
        "--frontier-design", $(if ($MatchedBudgetC4) { "matched_c4_c6" } else { "legacy_c6_only" })
      )
      if ($PlanOnly) { $analysisCmd += "--plan-only" }
      Invoke-Checked "budget-frontier inference" $analysisCmd
      if (-not $PlanOnly) { Invoke-FrontierPaperAssetRefresh }
    }
  }
  "RemainingAll" {
    if ($C4RMatched -or -not [string]::IsNullOrWhiteSpace($ReplicationGroupId)) {
      throw "Use -Task BudgetFrontier for C4R or replication; RemainingAll is reserved for the canonical frontier."
    }
    if (-not $PlanOnly -and -not $ConfirmLiveApiSpend) { throw "RemainingAll includes the live budget frontier. Re-run with -ConfirmLiveApiSpend." }
    $noApiCmd = @(
      $Py, "-m", "credit_recourse.analysis.remaining_thesis_analyses",
      "--project-root", $Root,
      "--mode", "noapi"
    )
    if ($PlanOnly) { $noApiCmd += "--plan-only" }
    Invoke-Checked "remaining no-API thesis analyses" $noApiCmd
    if (-not $PlanOnly -and $PromptApiKeys -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
      $env:OPENAI_API_KEY = Read-Secret "OPENAI_API_KEY"
    }
    if (-not $PlanOnly -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "OPENAI_API_KEY is required." }
    $tag = if ([string]::IsNullOrWhiteSpace($BudgetFrontierDateTag)) { Get-Date -Format "yyyyMMdd_HHmmss" } else { $BudgetFrontierDateTag }
    if ($tag -notmatch "^[0-9]{8}_[0-9]{6}$") { throw "BudgetFrontierDateTag must match yyyyMMdd_HHmmss; got $tag" }
    $frontierCmd = @(
      "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
      "-File", (Join-Path $Tools "run_n5_budget_frontier_icb.ps1"),
      "-ProjectRoot", $Root,
      "-DateTag", $tag
    )
    if ($MatchedBudgetC4) { $frontierCmd += "-BudgetC4" }
    if ($PlanOnly) { $frontierCmd += "-PlanOnly" }
    Invoke-Checked "IC-b C4/C6 generation-time budget frontier" $frontierCmd
    $frontierAnalysisCmd = @(
      $Py, "-m", "credit_recourse.analysis.remaining_thesis_analyses",
      "--project-root", $Root,
      "--mode", "frontier",
      "--frontier-design", $(if ($MatchedBudgetC4) { "matched_c4_c6" } else { "legacy_c6_only" })
    )
    if ($PlanOnly) { $frontierAnalysisCmd += "--plan-only" }
    Invoke-Checked "budget-frontier inference" $frontierAnalysisCmd
    if (-not $PlanOnly) { Invoke-FrontierPaperAssetRefresh }
  }
}
Write-Host "`nPASS: $Task" -ForegroundColor Green
