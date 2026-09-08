<#
Freeze-before-API journal extension runner for C4/C4R/C6 x four L1 arms x two backend cohorts.

Live execution is blocked unless -PreregEvidence identifies a public git commit,
OSF registration, or equivalent immutable timestamp. -PlanOnly and
-ScriptedRehearsal do not spend API calls.
#>
[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$PreregPath = "",
  [string]$PreregContractPath = "",
  [string]$PreregEvidence = "",
  [string[]]$ReuseGridManifest = @(),
  [string[]]$CohortIds = @(),
  [string[]]$BudgetLabels = @(),
  [switch]$AllowIncompleteGrid,
  [string]$DateTag = (Get-Date -Format "yyyyMMdd_HHmmss"),
  [int]$Seed = 1,
  [int]$ReferenceDrawSeed = 1,
  [int]$CandidateLibraryQuantile = 50,
  [int]$MaxConcurrency = 6,
  [int]$MaxRetries = 3,
  [double]$RetrySleepSeconds = 20.0,
  [switch]$RunAxisAttribution,
  [int]$AxisShapleyPermutations = 64,
  [switch]$PlanOnly,
  [switch]$ScriptedRehearsal
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Root([string]$Given) {
  if (-not [string]::IsNullOrWhiteSpace($Given)) { return (Resolve-Path -LiteralPath $Given).Path }
  return (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
}
function Sha256([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file missing for SHA-256: $Path" }
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Budget-Label([object]$Budget) {
  if ($null -eq $Budget) { return "unbounded" }
  if ([math]::Abs([double]$Budget - 0.75) -lt 1e-9) { return "0p75" }
  if ([math]::Abs([double]$Budget - 1.27) -lt 1e-9) { return "1p27" }
  if ([math]::Abs([double]$Budget - 2.00) -lt 1e-9) { return "2p00" }
  throw "Unexpected preregistered budget: $Budget"
}
function Expected-BackendId([string]$Backend, [object]$Cohort) {
  if ($Backend -eq "scripted") { return "scripted_reproducibility_v1" }
  if ($Backend -match "^openai:(.+)$") {
    return "openai_$($Matches[1])_responses_reasoning-low_maxout-$([int]$Cohort.max_output_tokens)"
  }
  if ($Backend -match "^anthropic:(.+)$") { return "anthropic_$($Matches[1])" }
  if ($Backend -match "^gemini:(.+)$") {
    $id = "google_$($Matches[1])_generate-content_thinking-$([string]$Cohort.thinking_level)"
    if ($null -ne $Cohort.max_output_tokens) { $id += "_maxout-$([int]$Cohort.max_output_tokens)" }
    if ([string]$Cohort.response_mime_type -eq "application/json") { $id += "_json" }
    return $id
  }
  throw "Unsupported journal backend: $Backend"
}
function Normalize-StringSet([string[]]$Values) {
  return @($Values | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
}
function Get-OptionalProperty([object]$Object, [string]$Name, [object]$Default = $null) {
  if ($null -eq $Object) { return $Default }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $Default
}
function Load-ReuseArmIndex([string[]]$ManifestPaths) {
  $index = @{}
  foreach ($manifestInput in (Normalize-StringSet $ManifestPaths)) {
    $manifestPath = (Resolve-Path -LiteralPath $manifestInput).Path
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($arm in @($manifest.arms)) {
      if ([string]$arm.status -notin @("PASS", "PASS_REUSED")) { continue }
      $key = "$([string]$arm.cohort_id)|$([string]$arm.budget_label)"
      if ($index.ContainsKey($key)) {
        $existing = $index[$key]
        if ((Resolve-Path -LiteralPath ([string]$existing.archive_dir)).Path -ne (Resolve-Path -LiteralPath ([string]$arm.archive_dir)).Path) {
          throw "Conflicting reusable archives for $key across prior grid manifests."
        }
        continue
      }
      if (-not (Test-Path -LiteralPath ([string]$arm.archive_dir) -PathType Container)) {
        throw "Reusable arm archive is missing for ${key}: $([string]$arm.archive_dir)"
      }
      $index[$key] = [ordered]@{
        cohort_id = [string]$arm.cohort_id
        budget_label = [string]$arm.budget_label
        archive_dir = (Resolve-Path -LiteralPath ([string]$arm.archive_dir)).Path
        source_manifest = $manifestPath
      }
    }
  }
  return $index
}
function Write-Json([string]$Path, [object]$Value) {
  $parent = Split-Path -Parent $Path
  if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
  [System.IO.File]::WriteAllText(
    $Path,
    ($Value | ConvertTo-Json -Depth 12),
    (New-Object System.Text.UTF8Encoding -ArgumentList $false)
  )
}
function Assert-PreregEvidence([string]$Value) {
  if ([string]::IsNullOrWhiteSpace($Value)) {
    throw "Live execution requires -PreregEvidence with a real immutable public commit, OSF registration, DOI, or HTTPS record."
  }
  $candidate = $Value.Trim()
  $placeholderPatterns = @(
    '^<.*>$',
    '(?i)public commit or osf identifier',
    '(?i)placeholder',
    '(?i)pending',
    '(?i)todo',
    '(?i)tbd'
  )
  foreach ($pattern in $placeholderPatterns) {
    if ($candidate -match $pattern) {
      throw "-PreregEvidence is still a placeholder and cannot authorize live execution: $candidate"
    }
  }
  $isCommitHash = $candidate -match '(?i)^(git:)?[0-9a-f]{7,64}$'
  $isOsfId = $candidate -match '(?i)^osf:[a-z0-9_-]{4,}$'
  $isDoi = $candidate -match '(?i)^doi:10\.\d{4,9}/\S+$'
  $uri = $null
  $isHttps = [System.Uri]::TryCreate($candidate, [System.UriKind]::Absolute, [ref]$uri) -and $uri.Scheme -eq 'https'
  if (-not ($isCommitHash -or $isOsfId -or $isDoi -or $isHttps)) {
    throw "-PreregEvidence must be an HTTPS immutable record, git commit hash, osf:<id>, or doi:<doi>. Received: $candidate"
  }
}
function Invoke-Checked([string]$Label, [string[]]$Command) {
  Write-Host "`n==== $Label ====" -ForegroundColor Cyan
  Write-Host "CMD> $($Command -join ' ')" -ForegroundColor DarkGray
  & $Command[0] @($Command[1..($Command.Count - 1)])
  if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

$Root = Resolve-Root $ProjectRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Virtualenv Python missing: $Py" }
if ([string]::IsNullOrWhiteSpace($PreregContractPath)) {
  $PreregContractPath = Join-Path $Root "src\credit_recourse\configs\c4r_journal_extension_prereg_v3.json"
}
if (-not (Test-Path -LiteralPath $PreregContractPath -PathType Leaf)) {
  throw "Machine-readable journal extension contract missing: $PreregContractPath"
}
# The Markdown preregistration note is documentation only. It is optional and is
# never required for planning, rehearsal, analysis, or live execution.
if (-not [string]::IsNullOrWhiteSpace($PreregPath) -and -not (Test-Path -LiteralPath $PreregPath -PathType Leaf)) {
  throw "Optional -PreregPath was supplied but the file does not exist: $PreregPath"
}
if (-not $PlanOnly -and -not $ScriptedRehearsal) {
  Assert-PreregEvidence $PreregEvidence
}

$Contract = Get-Content -LiteralPath $PreregContractPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$Contract.design_status -notin @("LOCKED_FOR_FREEZE_BEFORE_LIVE_API", "LOCKED_AMENDMENT_BEFORE_REPLACEMENT_LIVE_API", "LOCKED_SECOND_AMENDMENT_AFTER_GEMINI35_PROVIDER_FEASIBILITY_FAILURE_BEFORE_FLASHLITE_LIVE_API")) {
  throw "Preregistration design_status is not locked: $($Contract.design_status)"
}
if ([string]$Contract.information_condition -ne "IC-b") { throw "Journal extension contract must be IC-b." }
if ((@($Contract.conditions) -join ",") -ne "C4,C4R,C6") { throw "Preregistered conditions must be C4,C4R,C6." }
if ([string]$Contract.mode -ne "free_form_10d") { throw "Preregistered mode must be free_form_10d." }
if ([int]$Contract.firm_count_per_arm -ne 575) { throw "Preregistered firm_count_per_arm must be 575." }
if ($Seed -ne [int]$Contract.seed_protocol.run_contract_seed -or $ReferenceDrawSeed -ne [int]$Contract.seed_protocol.reference_draw_seed) {
  throw "Runner seed/reference seed must match preregistration."
}
if ($CandidateLibraryQuantile -ne [int]$Contract.candidate_library_quantile) {
  throw "Candidate-library quantile must match preregistration."
}
$Budgets = @($Contract.budgets | ForEach-Object { if ($null -eq $_.l1_budget) { $null } else { [double]$_.l1_budget } })
if ($Budgets.Count -ne 4) { throw "Preregistration must contain four budget arms." }

Set-Location -LiteralPath $Root
$env:PYTHONPATH = (Join-Path $Root "src") + ";" + $env:PYTHONPATH

# Static/synthetic acceptance before plan or API execution.
Invoke-Checked "C4R journal v3 contract verifier" @(
  $Py, "-m", "credit_recourse.verification.verify_c4r_matched_v3_contract",
  "--project-root", $Root
)
Invoke-Checked "Axis-swap policy-pair contract verifier" @(
  $Py, "-m", "credit_recourse.verification.verify_n5m_axis_swap_contract",
  "--project-root", $Root
)
Invoke-Checked "Gemini Stage7 backend contract verifier" @(
  $Py, "-m", "credit_recourse.verification.verify_gemini_stage7_backend_contract",
  "--project-root", $Root
)

# PowerShell functions enumerate their output.  A single normalized filter value
# therefore collapses to a scalar unless the caller explicitly re-wraps it.
# Keep all selection variables as arrays so StrictMode-safe .Count checks work
# for one-cohort pilots such as (-CohortIds gemini31flashlite -BudgetLabels 0p75).
$RequestedCohortIds = @(Normalize-StringSet $CohortIds)
$RequestedBudgetLabels = @(Normalize-StringSet $BudgetLabels)
$ContractBudgetLabels = @($Contract.budgets | ForEach-Object { [string]$_.label })
$ContractCohortIds = @($Contract.cohorts | ForEach-Object { [string]$_.cohort_id })
foreach ($requested in $RequestedCohortIds) {
  if ($requested -notin $ContractCohortIds) { throw "Unknown -CohortIds value not in preregistration: $requested" }
}
foreach ($requested in $RequestedBudgetLabels) {
  if ($requested -notin $ContractBudgetLabels) { throw "Unknown -BudgetLabels value not in preregistration: $requested" }
}
$SelectedCohortIds = @(
  if ($RequestedCohortIds.Count -gt 0) { $RequestedCohortIds } else { $ContractCohortIds }
)
$SelectedBudgetLabels = @(
  if ($RequestedBudgetLabels.Count -gt 0) { $RequestedBudgetLabels } else { $ContractBudgetLabels }
)
$IsCompleteContractGrid = (
  $SelectedCohortIds.Count -eq $ContractCohortIds.Count -and
  $SelectedBudgetLabels.Count -eq $ContractBudgetLabels.Count
)
if (-not $IsCompleteContractGrid -and -not $AllowIncompleteGrid) {
  throw "Cohort/budget filters create an incomplete grid. Re-run with -AllowIncompleteGrid for an explicit QC-only pilot."
}
if (-not $IsCompleteContractGrid -and $RunAxisAttribution) {
  throw "-RunAxisAttribution is not allowed for an incomplete pilot grid."
}

$ReuseArmIndex = if ($ScriptedRehearsal) { @{} } else { Load-ReuseArmIndex $ReuseGridManifest }
$Cohorts = @()
foreach ($cohort in @($Contract.cohorts)) {
  if ([string]$cohort.cohort_id -notin $SelectedCohortIds) { continue }
  if ($ScriptedRehearsal) {
    $Cohorts += [ordered]@{
      cohort_id = "$($cohort.cohort_id)_rehearsal"
      backend = "scripted"
      run_role = "rehearsal_c4r_journal_ext_$($cohort.cohort_id)"
      execution_policy = "generate"
      max_output_tokens = [int](Get-OptionalProperty $cohort "max_output_tokens" 1200)
      temperature = 0.0
      thinking_level = $null
      response_mime_type = $null
      timeout_seconds = $null
      source_cohort_id = [string]$cohort.cohort_id
    }
  } else {
    $temperatureValue = Get-OptionalProperty $cohort "temperature" $null
    $Cohorts += [ordered]@{
      cohort_id = [string]$cohort.cohort_id
      backend = [string]$cohort.backend
      run_role = [string]$cohort.run_role
      execution_policy = [string](Get-OptionalProperty $cohort "execution_policy" "generate")
      max_output_tokens = [int](Get-OptionalProperty $cohort "max_output_tokens" 1200)
      temperature = if ($null -eq $temperatureValue) { $null } else { [double]$temperatureValue }
      thinking_level = Get-OptionalProperty $cohort "thinking_level" $null
      response_mime_type = Get-OptionalProperty $cohort "response_mime_type" $null
      timeout_seconds = Get-OptionalProperty $cohort "timeout_seconds" $null
      source_cohort_id = [string]$cohort.cohort_id
    }
  }
}
if ($Cohorts.Count -eq 0) { throw "No cohorts remain after applying -CohortIds." }

$GridKind = if ($ScriptedRehearsal) { "scripted_rehearsal" } elseif ($PlanOnly) { "plan" } else { "live" }
$GridDir = Join-Path $Root "data\analysis\c4r_journal_extension\${GridKind}_${DateTag}"
$ManifestPath = Join-Path $GridDir "c4r_journal_grid_manifest.json"
$QcDir = Join-Path $GridDir "arm_qc"
New-Item -ItemType Directory -Path $QcDir -Force | Out-Null

$SourceFiles = @(
  (Join-Path $Root "src\credit_recourse\analysis\c4r_matched_inference.py"),
  (Join-Path $Root "src\credit_recourse\analysis\c4r_matched_inference_v3.py"),
  (Join-Path $Root "src\credit_recourse\analysis\n5m_axis_swap_intervention.py"),
  (Join-Path $Root "src\credit_recourse\rl\pipelines\final_stage7_llm_action_generation\llm_backends.py"),
  (Join-Path $Root "src\credit_recourse\utils\run_llm_stages.py"),
  (Join-Path $Root "src\credit_recourse\verification\verify_c4r_matched_v3_contract.py"),
  (Join-Path $Root "src\credit_recourse\verification\verify_c4r_journal_arm_contract.py"),
  (Join-Path $Root "src\credit_recourse\verification\verify_gemini_stage7_backend_contract.py"),
  $PreregContractPath,
  (Join-Path $Root "tools\run_c4r_journal_grid.ps1")
)
$SourceHashes = [ordered]@{}
foreach ($file in $SourceFiles) { $SourceHashes[(Resolve-Path -LiteralPath $file).Path] = Sha256 $file }

$Arms = @()
foreach ($cohort in $Cohorts) {
  foreach ($budgetEntry in @($Contract.budgets)) {
    $budgetLabel = [string]$budgetEntry.label
    if ($budgetLabel -notin $SelectedBudgetLabels) { continue }
    $budget = if ($null -eq $budgetEntry.l1_budget) { $null } else { [double]$budgetEntry.l1_budget }
    $backendSlug = [string]$cohort.cohort_id
    $runLabel = "C4RXJ_C4C4RC6_L1_${budgetLabel}_ICb_${backendSlug}_p${CandidateLibraryQuantile}_seed${Seed}_${DateTag}"
    $generatedArchiveDir = Join-Path $Root "data\final_freeze\llm_runs\$runLabel"
    $checkpointDir = Join-Path $Root "data\diagnostics\c4r_journal_checkpoints"
    New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null
    $checkpoint = Join-Path $checkpointDir "checkpoint_${runLabel}.jsonl"
    $expectedBackendId = Expected-BackendId -Backend ([string]$cohort.backend) -Cohort $cohort
    $reuseKey = "$([string]$cohort.source_cohort_id)|${budgetLabel}"
    $reuseArm = if ($ReuseArmIndex.ContainsKey($reuseKey)) { $ReuseArmIndex[$reuseKey] } else { $null }
    if ($ScriptedRehearsal) { $reuseArm = $null }
    if ($null -eq $reuseArm -and [string]$cohort.execution_policy -eq "reuse_required") {
      throw "Cohort=$($cohort.cohort_id) budget=$budgetLabel is reuse_required but no PASS arm was found in -ReuseGridManifest."
    }

    $executionMode = if ($null -ne $reuseArm) { "reuse" } else { "generate" }
    $archiveDir = if ($null -ne $reuseArm) { [string]$reuseArm.archive_dir } else { $generatedArchiveDir }
    $cmd = @()
    if ($executionMode -eq "generate") {
      $cmd = @(
        $Py, "-m", "credit_recourse.utils.run_llm_stages",
        "--project-root", $Root,
        "--backend", [string]$cohort.backend,
        "--information-condition", "IC-b",
        "--conditions", "C4,C4R,C6",
        "--modes", "free_form_10d",
        "--seed", "$Seed",
        "--reference-draw-seed", "$ReferenceDrawSeed",
        "--candidate-library-quantile", "$CandidateLibraryQuantile",
        "--checkpoint-path", $checkpoint,
        "--run-label", $runLabel,
        "--run-role", [string]$cohort.run_role,
        "--max-concurrency", "$MaxConcurrency",
        "--max-retries", "$MaxRetries",
        "--retry-sleep-seconds", "$RetrySleepSeconds"
      )
      if ([string]$cohort.backend -match "^openai:") {
        $cmd += @(
          "--openai-api-mode", "responses",
          "--openai-reasoning-effort", "low",
          "--openai-max-output-tokens", "$($cohort.max_output_tokens)"
        )
      } elseif ([string]$cohort.backend -match "^anthropic:") {
        $cmd += @("--anthropic-max-tokens", "$($cohort.max_output_tokens)")
        if ($null -ne $cohort.temperature) { $cmd += @("--anthropic-temperature", "$($cohort.temperature)") }
      } elseif ([string]$cohort.backend -match "^gemini:") {
        $cmd += @(
          "--gemini-thinking-level", "$($cohort.thinking_level)",
          "--gemini-max-output-tokens", "$($cohort.max_output_tokens)",
          "--gemini-response-mime-type", "$($cohort.response_mime_type)",
          "--gemini-timeout-seconds", "$($cohort.timeout_seconds)"
        )
      }
      if ($null -ne $budget) {
        $budgetInvariant = ([double]$budget).ToString([Globalization.CultureInfo]::InvariantCulture)
        $cmd += @(
          "--freeform-l1-budget", $budgetInvariant,
          "--budgeted-conditions", "C4,C4R,C6",
          "--budget-contract-label", "C4RXJ_C4C4RC6_L1_${budgetLabel}_ICb"
        )
      }
    }
    $Arms += [ordered]@{
      cohort_id = [string]$cohort.cohort_id
      source_cohort_id = [string]$cohort.source_cohort_id
      backend = [string]$cohort.backend
      expected_backend_id = $expectedBackendId
      run_role = [string]$cohort.run_role
      execution_policy = [string]$cohort.execution_policy
      execution_mode = $executionMode
      reused_from_manifest = if ($null -eq $reuseArm) { $null } else { [string]$reuseArm.source_manifest }
      budget_label = $budgetLabel
      l1_budget = $budget
      run_label = if ($executionMode -eq "reuse") { Split-Path -Leaf $archiveDir } else { $runLabel }
      archive_dir = $archiveDir
      checkpoint = if ($executionMode -eq "reuse") { $null } else { $checkpoint }
      command = $cmd
      qc_json = (Join-Path $QcDir "${backendSlug}_${budgetLabel}.json")
      status = if ($PlanOnly) { if ($executionMode -eq "reuse") { "PLANNED_REUSE" } else { "PLANNED_GENERATION" } } else { "PENDING" }
    }
  }
}
if ($Arms.Count -eq 0) { throw "No arms remain after applying cohort/budget filters." }

$GeneratedArms = @($Arms | Where-Object { $_.execution_mode -eq "generate" })
if (-not $PlanOnly -and -not $ScriptedRehearsal) {
  if (@($GeneratedArms | Where-Object { $_.backend -match "^openai:" }).Count -gt 0 -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "OPENAI_API_KEY is not set for a newly generated OpenAI arm."
  }
  if (@($GeneratedArms | Where-Object { $_.backend -match "^anthropic:" }).Count -gt 0 -and [string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) {
    throw "ANTHROPIC_API_KEY is not set for a newly generated Anthropic arm."
  }
  if (@($GeneratedArms | Where-Object { $_.backend -match "^gemini:" }).Count -gt 0 -and [string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
    throw "GEMINI_API_KEY is not set for a newly generated Gemini arm."
  }
}

$Manifest = [ordered]@{
  schema_version = "c4r_journal_grid_runner_v2"
  status = if ($PlanOnly) { "PLANNED" } else { "RUNNING" }
  created_utc = [DateTime]::UtcNow.ToString("o")
  project_root = $Root
  grid_kind = $GridKind
  date_tag = $DateTag
  complete_contract_grid = [bool]$IsCompleteContractGrid
  selected_cohort_ids = $SelectedCohortIds
  selected_budget_labels = $SelectedBudgetLabels
  reuse_grid_manifests = @((Normalize-StringSet $ReuseGridManifest) | ForEach-Object { (Resolve-Path -LiteralPath $_).Path })
  preregistration = [ordered]@{
    contract_path = (Resolve-Path -LiteralPath $PreregContractPath).Path
    contract_sha256 = Sha256 $PreregContractPath
    documentation_path = if ([string]::IsNullOrWhiteSpace($PreregPath)) { $null } else { (Resolve-Path -LiteralPath $PreregPath).Path }
    documentation_sha256 = if ([string]::IsNullOrWhiteSpace($PreregPath)) { $null } else { Sha256 $PreregPath }
    immutable_timestamp_evidence = if ([string]::IsNullOrWhiteSpace($PreregEvidence)) { $null } else { $PreregEvidence }
  }
  provider_generation_seed_passed = $false
  run_contract_seed = $Seed
  reference_draw_seed = $ReferenceDrawSeed
  source_hashes = $SourceHashes
  arms = $Arms
  analysis_dir = (Join-Path $GridDir "inference_v3")
  analysis_status = if ($PlanOnly) { "PLANNED" } else { "PENDING" }
  axis_attribution_requested = [bool]$RunAxisAttribution
  axis_attribution_status = if ($RunAxisAttribution) { if ($PlanOnly) { "PLANNED" } else { "PENDING" } } else { "NOT_REQUESTED" }
}
Write-Json -Path $ManifestPath -Value $Manifest

foreach ($arm in $Arms) {
  Write-Host "`n==== C4R journal arm: $($arm.cohort_id) / $($arm.budget_label) ====" -ForegroundColor Cyan
  if ($arm.execution_mode -eq "reuse") {
    Write-Host "REUSE> $($arm.archive_dir)" -ForegroundColor DarkGray
  } else {
    Write-Host "CMD> $($arm.command -join ' ')" -ForegroundColor DarkGray
  }
  if ($PlanOnly) { continue }
  if ($arm.execution_mode -eq "generate") {
    if (Test-Path -LiteralPath $arm.archive_dir) {
      throw "Immutable archive path already exists; refusing overwrite: $($arm.archive_dir)"
    }
    foreach ($name in @(
      "stage7_llm_action_generation",
      "stage8_llm_multi_oracle_eval",
      "stage9_llm_rl_comparison",
      "stage9_statistical_inference"
    )) {
      $active = Join-Path $Root "data\final_freeze\$name"
      if (Test-Path -LiteralPath $active) { Remove-Item -LiteralPath $active -Recurse -Force }
    }
    $armExecutable = [string]$arm.command[0]
    $armArguments = @($arm.command[1..($arm.command.Count - 1)])
    & $armExecutable @armArguments
    if ($LASTEXITCODE -ne 0) {
      $arm.status = "FAIL_GENERATION"
      Write-Json -Path $ManifestPath -Value $Manifest
      throw "Journal arm failed: $($arm.run_label)"
    }
  } elseif (-not (Test-Path -LiteralPath $arm.archive_dir -PathType Container)) {
    throw "Reusable archive disappeared before QC: $($arm.archive_dir)"
  }
  $qcCmd = @(
    $Py, "-m", "credit_recourse.verification.verify_c4r_journal_arm_contract",
    "--archive-dir", $arm.archive_dir,
    "--cohort-id", $arm.cohort_id,
    "--budget-label", $arm.budget_label,
    "--l1-budget", $(if ($null -eq $arm.l1_budget) { "unbounded" } else { ([double]$arm.l1_budget).ToString([Globalization.CultureInfo]::InvariantCulture) }),
    "--expected-run-role", $arm.run_role,
    "--expected-backend-id", $arm.expected_backend_id,
    "--aggregate-raw-compliance-min", "$($Contract.quality_gates.finite_arm_raw_budget_compliance_aggregate_min)",
    "--condition-raw-compliance-min", "$($Contract.quality_gates.finite_arm_raw_budget_compliance_each_condition_min)",
    "--out-json", $arm.qc_json
  )
  if ($ScriptedRehearsal) { $qcCmd += "--allow-nonlive" }
  & $qcCmd[0] @($qcCmd[1..($qcCmd.Count - 1)])
  if ($LASTEXITCODE -ne 0) {
    $arm.status = "FAIL_QC"
    Write-Json -Path $ManifestPath -Value $Manifest
    throw "Journal arm QC failed; stopping before next arm: $($arm.run_label)"
  }
  $qc = Get-Content -LiteralPath $arm.qc_json -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$qc.status -ne "PASS") { throw "Journal arm QC status is not PASS: $($arm.qc_json)" }
  if ($null -ne $arm.l1_budget -and [double]$qc.aggregate_raw_budget_compliance -lt [double]$Contract.quality_gates.finite_arm_raw_budget_compliance_aggregate_min) {
    throw "Finite arm aggregate compliance gate failed after verifier: $($arm.run_label)"
  }
  if (-not [bool]$qc.prompt_payload_hash_match) { throw "FULL_PAYLOAD_ARCHIVED hash gate failed: $($arm.run_label)" }
  $arm.status = if ($arm.execution_mode -eq "reuse") { "PASS_REUSED" } else { "PASS" }
  $arm.candidate_library_hash = [string]$qc.candidate_library_hash
  $arm.aggregate_raw_budget_compliance = $qc.aggregate_raw_budget_compliance
  $arm.condition_raw_budget_compliance = $qc.condition_raw_budget_compliance
  Write-Json -Path $ManifestPath -Value $Manifest
}

if (-not $PlanOnly) {
  $hashes = @($Arms | ForEach-Object { [string]$_.candidate_library_hash } | Sort-Object -Unique)
  if ($hashes.Count -ne 1) { throw "Candidate-library hash mismatch across completed arms: $($hashes -join ',')" }
  if (-not $IsCompleteContractGrid) {
    $Manifest.analysis_status = "NOT_RUN_INCOMPLETE_GRID"
    $Manifest.axis_attribution_status = "NOT_RUN_INCOMPLETE_GRID"
    $Manifest.status = "PASS_PARTIAL_QC"
    $Manifest.completed_utc = [DateTime]::UtcNow.ToString("o")
    Write-Json -Path $ManifestPath -Value $Manifest
    Write-Host "`nPASS_PARTIAL_QC: C4R journal pilot manifest -> $ManifestPath" -ForegroundColor Green
    exit 0
  }
  $analysisDir = [string]$Manifest.analysis_dir
  New-Item -ItemType Directory -Path $analysisDir -Force | Out-Null
  $analysisCmd = @($Py, "-m", "credit_recourse.analysis.c4r_matched_inference_v3")
  foreach ($arm in $Arms) {
    $analysisCmd += @("--arm", "$($arm.cohort_id)|$($arm.budget_label)=$($arm.archive_dir)")
  }
  foreach ($budgetEntry in @($Contract.budgets)) {
    $budgetValue = if ($null -eq $budgetEntry.l1_budget) { "none" } else { ([double]$budgetEntry.l1_budget).ToString([Globalization.CultureInfo]::InvariantCulture) }
    $analysisCmd += @("--budget-spec", "$($budgetEntry.label)=$budgetValue")
  }
  foreach ($cohort in $Cohorts) {
    $analysisCmd += @("--expected-run-role", "$($cohort.cohort_id)=$($cohort.run_role)")
  }
  $analysisCmd += @(
    "--expected-firm-count", "575",
    "--information-condition", "IC-b",
    "--out", $analysisDir
  )
  if (-not [string]::IsNullOrWhiteSpace($PreregPath)) {
    $analysisCmd += @("--prereg-path", $PreregPath)
  }
  if ($ScriptedRehearsal) { $analysisCmd += "--allow-nonlive" }
  Invoke-Checked "C4R journal multi-cohort inference v3" $analysisCmd
  $analysisManifest = Join-Path $analysisDir "c4r_matched_inference_v3_manifest.json"
  if (-not (Test-Path -LiteralPath $analysisManifest -PathType Leaf)) { throw "v3 analysis manifest missing: $analysisManifest" }
  $analysisMeta = Get-Content -LiteralPath $analysisManifest -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$analysisMeta.status -ne "PASS") { throw "v3 analysis did not PASS: $analysisManifest" }
  $Manifest.analysis_status = "PASS"
  $Manifest.analysis_manifest = $analysisManifest
  if ($RunAxisAttribution) {
    $AxisOutputs = @()
    foreach ($cohort in $Cohorts) {
      $axisArms = @($Arms | Where-Object { $_.cohort_id -eq $cohort.cohort_id -and $_.budget_label -in @("0p75", "unbounded") })
      if ($axisArms.Count -ne 2) { throw "Axis attribution requires 0p75 and unbounded arms for cohort=$($cohort.cohort_id)." }
      foreach ($pair in @(
        [ordered]@{ name = "self_revision"; base = "C4"; target = "C4R" },
        [ordered]@{ name = "reference_content"; base = "C4R"; target = "C6" }
      )) {
        $axisOut = Join-Path $GridDir "axis_attribution\$($cohort.cohort_id)\$($pair.name)"
        $axisCmd = @(
          $Py, "-m", "credit_recourse.analysis.n5m_axis_swap_intervention",
          "--project-root", $Root,
          "--base-policy", $pair.base,
          "--target-policy", $pair.target,
          "--shapley-permutations", "$AxisShapleyPermutations",
          "--shapley-arms", "0p75,unbounded",
          "--out", $axisOut
        )
        foreach ($axisArm in $axisArms) { $axisCmd += @("--arm", "$($axisArm.budget_label)=$($axisArm.archive_dir)") }
        Invoke-Checked "Axis attribution $($cohort.cohort_id) $($pair.base)->$($pair.target)" $axisCmd
        $axisManifest = Join-Path $axisOut "intervention_manifest.json"
        if (-not (Test-Path -LiteralPath $axisManifest -PathType Leaf)) { throw "Axis-attribution manifest missing: $axisManifest" }
        $axisMeta = Get-Content -LiteralPath $axisManifest -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([string]$axisMeta.base_policy -ne [string]$pair.base -or [string]$axisMeta.target_policy -ne [string]$pair.target) {
          throw "Axis-attribution policy-pair manifest mismatch: $axisManifest"
        }
        $AxisOutputs += [ordered]@{ cohort_id = $cohort.cohort_id; component = $pair.name; manifest = $axisManifest }
      }
    }
    $Manifest.axis_attribution_status = "PASS"
    $Manifest.axis_attribution_outputs = $AxisOutputs
  }
  $Manifest.status = "PASS"
  $Manifest.completed_utc = [DateTime]::UtcNow.ToString("o")
  Write-Json -Path $ManifestPath -Value $Manifest
}

Write-Host "`n$($Manifest.status): C4R journal grid manifest -> $ManifestPath" -ForegroundColor Green
