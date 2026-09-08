<# Canonical same-date N5 generation-time budget frontier for IC-b.
Default: finite-arm budget applies to C6 only (legacy N5F).
-BudgetC4: matched 2x4 factorial where the same finite-arm budget applies to both C4 and C6.
-C4RMatched: two-arm (0.75/unbounded) fresh same-batch C4/C4R/C6 experiment. #>
[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$Backend = "openai:gpt-5.4-mini",
  [string]$BackendLabel = "",
  [string]$InformationCondition = "IC-b",
  [ValidateSet("canonical", "replication")][string]$ExperimentClass = "canonical",
  [string]$ReplicationGroupId = "",
  [string]$RunRole = "",
  [int]$Seed = 1,
  [int]$ReferenceDrawSeed = 1,
  [object[]]$L1Budgets = @(0.75, 1.27, 2.00, $null),
  [int]$CandidateLibraryQuantile = 50,
  [int]$MaxConcurrency = 6,
  [int]$MaxRetries = 3,
  [double]$RetrySleepSeconds = 20.0,
  [string]$DateTag = (Get-Date -Format "yyyyMMdd_HHmmss"),
  [switch]$BudgetC4,
  [switch]$C4RMatched,
  [switch]$PlanOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Root([string]$Given) {
  if ($Given) { return (Resolve-Path -LiteralPath $Given).Path }
  return (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
}
function Budget-Label([object]$Budget) {
  if ($null -eq $Budget) { return "unbounded" }
  return (("{0:0.00}" -f [double]$Budget).Replace(".", "p"))
}
function Backend-Slug([string]$BackendSpec, [string]$ExplicitLabel) {
  if (-not [string]::IsNullOrWhiteSpace($ExplicitLabel)) {
    $slug = $ExplicitLabel.Trim().ToLowerInvariant()
  } elseif ($BackendSpec -eq "openai:gpt-5.4-mini") {
    $slug = "gpt54mini"
  } elseif ($BackendSpec -eq "openai:gpt-4.1-mini") {
    $slug = "gpt41mini"
  } elseif ($BackendSpec -match "claude-haiku-4-5") {
    $slug = "haiku45"
  } else {
    $slug = ($BackendSpec.ToLowerInvariant() -replace "^[^:]+:", "" -replace "[^a-z0-9]+", "")
  }
  if ([string]::IsNullOrWhiteSpace($slug) -or $slug -notmatch "^[a-z0-9]+$") {
    throw "BackendLabel must resolve to a non-empty lowercase alphanumeric token. backend=$BackendSpec label=$ExplicitLabel"
  }
  return $slug
}
function Expected-BackendId([string]$BackendSpec) {
  if ($BackendSpec -match "^openai:(.+)$") {
    return "openai_$($Matches[1])_responses_reasoning-low_maxout-1200"
  }
  if ($BackendSpec -match "^anthropic:(.+)$") {
    return "anthropic_$($Matches[1])"
  }
  throw "Generation-time frontier requires an explicit live openai:<model> or anthropic:<model> backend; got $BackendSpec"
}
function Assert-ArchivedBackend(
  [string]$ArchiveDir,
  [string]$ExpectedBackendId,
  [string]$ExpectedRunLabel,
  [string]$ExpectedRunRole
) {
  $archiveManifest = Join-Path $ArchiveDir "archive_manifest.json"
  $stage7MetaPath = Join-Path $ArchiveDir "stage7_llm_action_generation\metadata.json"
  if (-not (Test-Path -LiteralPath $archiveManifest -PathType Leaf)) { throw "Archive manifest missing: $archiveManifest" }
  if (-not (Test-Path -LiteralPath $stage7MetaPath -PathType Leaf)) { throw "Archived Stage7 metadata missing: $stage7MetaPath" }
  $archiveMeta = Get-Content -LiteralPath $archiveManifest -Raw -Encoding UTF8 | ConvertFrom-Json
  $stage7Meta = Get-Content -LiteralPath $stage7MetaPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$archiveMeta.run_label -ne $ExpectedRunLabel) { throw "Existing archive run_label mismatch: expected=$ExpectedRunLabel observed=$($archiveMeta.run_label)" }
  if ([string]$archiveMeta.run_role -ne $ExpectedRunRole) { throw "Existing archive run_role mismatch: expected=$ExpectedRunRole observed=$($archiveMeta.run_role)" }
  if ([int]$archiveMeta.file_count -le 0) { throw "Existing archive has no files: $archiveManifest" }
  if ([string]$stage7Meta.backend_id -ne $ExpectedBackendId) { throw "Existing archive backend_id mismatch: expected=$ExpectedBackendId observed=$($stage7Meta.backend_id)" }
  if (-not [bool]$stage7Meta.backend_is_live) { throw "Frontier archive must use a live backend: $stage7MetaPath" }
}
function Assert-C4R-ArchivedContract([string]$ArchiveDir, [object]$ExpectedBudget) {
  $stage7MetaPath = Join-Path $ArchiveDir "stage7_llm_action_generation\metadata.json"
  $stage7Meta = Get-Content -LiteralPath $stage7MetaPath -Raw -Encoding UTF8 | ConvertFrom-Json
  $conditions = @($stage7Meta.conditions | ForEach-Object { [string]$_ } | Sort-Object)
  if (($conditions -join ",") -ne "C4,C4R,C6") { throw "C4R archive conditions mismatch: $($conditions -join ',')" }
  $modes = @($stage7Meta.modes | ForEach-Object { [string]$_ } | Sort-Object)
  if (($modes -join ",") -ne "free_form_10d") { throw "C4R archive modes mismatch: $($modes -join ',')" }
  $contract = $stage7Meta.action_budget_contract
  if ($null -ne $ExpectedBudget) {
    if (-not [bool]$contract.enabled) { throw "Finite C4R arm must enable the action-budget contract." }
    if ([math]::Abs([double]$contract.l1_budget - [double]$ExpectedBudget) -gt 1e-9) { throw "Finite C4R arm L1 mismatch." }
    $budgeted = @($contract.budgeted_conditions | ForEach-Object { [string]$_ } | Sort-Object)
    if (($budgeted -join ",") -ne "C4,C4R,C6") { throw "Finite C4R budgeted-condition mismatch: $($budgeted -join ',')" }
  } elseif ([bool]$contract.enabled) {
    throw "Unbounded C4R arm must not enable the action-budget contract."
  }
}
function Assert-Budget-Grid([object[]]$Budgets, [bool]$IsC4RMatched) {
  if ($IsC4RMatched) {
    if ($Budgets.Count -ne 2) { throw "C4R matched experiment requires exactly two arms: 0.75 and unbounded." }
    $finite = @($Budgets | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ })
    if ($finite.Count -ne 1 -or [math]::Abs($finite[0] - 0.75) -gt 1e-9) {
      throw "C4R matched finite arm must be exactly 0.75."
    }
    if (@($Budgets | Where-Object { $null -eq $_ }).Count -ne 1) { throw "C4R matched experiment requires exactly one unbounded arm." }
    return
  }
  if ($Budgets.Count -ne 4) { throw "Budget frontier requires exactly four arms." }
  $finite = @($Budgets | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ } | Sort-Object)
  if ($finite.Count -ne 3 -or [math]::Abs($finite[0] - 0.75) -gt 1e-9 -or [math]::Abs($finite[1] - 1.27) -gt 1e-9 -or [math]::Abs($finite[2] - 2.00) -gt 1e-9) {
    throw "Finite budget arms must be exactly 0.75, 1.27, and 2.00."
  }
  if (@($Budgets | Where-Object { $null -eq $_ }).Count -ne 1) { throw "Budget frontier requires exactly one unbounded arm." }
}
function Preserve-CheckpointAcrossCanonicalCleanup([string]$Checkpoint, [string]$RecoveryCopy) {
  if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { return $false }
  $recoveryParent = Split-Path -Parent $RecoveryCopy
  New-Item -ItemType Directory -Path $recoveryParent -Force | Out-Null
  Copy-Item -LiteralPath $Checkpoint -Destination $RecoveryCopy -Force
  return $true
}
function Restore-CheckpointAfterCanonicalCleanup([string]$Checkpoint, [string]$RecoveryCopy) {
  if (-not (Test-Path -LiteralPath $RecoveryCopy -PathType Leaf)) { return $false }
  $checkpointParent = Split-Path -Parent $Checkpoint
  New-Item -ItemType Directory -Path $checkpointParent -Force | Out-Null
  Copy-Item -LiteralPath $RecoveryCopy -Destination $Checkpoint -Force
  return $true
}

$Root = Resolve-Root $ProjectRoot
if ($BudgetC4 -and $C4RMatched) { throw "-BudgetC4 and -C4RMatched are mutually exclusive." }
if ($C4RMatched -and $ExperimentClass -ne "canonical") { throw "C4R matched experiment is canonical-only; replication uses the matched C4/C6 design." }
if ($C4RMatched -and -not $PSBoundParameters.ContainsKey("L1Budgets")) { $L1Budgets = @(0.75, $null) }
$Design = if ($C4RMatched) { "c4r_matched" } elseif ($BudgetC4) { "matched_c4_c6" } else { "legacy_c6_only" }
$ProfileKey = if ($C4RMatched) { "c4r_matched" } elseif ($BudgetC4) { "n5_matched_budget_frontier" } else { "n5_budget_frontier" }
$DefaultRunRole = if ($C4RMatched) { "paper_c4r_matched_icb" } elseif ($BudgetC4) { "paper_n5_matched_budget_frontier_icb" } else { "paper_n5_budget_frontier_icb" }
$BackendSlug = Backend-Slug $Backend $BackendLabel
$ExpectedBackendId = Expected-BackendId $Backend
if ($ExperimentClass -eq "canonical") {
  if ([string]::IsNullOrWhiteSpace($RunRole)) { $RunRole = $DefaultRunRole }
  if ($RunRole -ne $DefaultRunRole) {
    throw "Canonical RunRole must match design=$Design. Expected=$DefaultRunRole observed=$RunRole"
  }
  if (-not [string]::IsNullOrWhiteSpace($ReplicationGroupId)) {
    throw "ReplicationGroupId is allowed only when -ExperimentClass replication."
  }
} else {
  if (-not $BudgetC4) { throw "Replication frontier requires -BudgetC4 matched design." }
  if ([string]::IsNullOrWhiteSpace($ReplicationGroupId) -or $ReplicationGroupId -notmatch "^[A-Za-z0-9_-]+$") {
    throw "ReplicationGroupId is required and must be directory-safe for replication runs."
  }
  if ([string]::IsNullOrWhiteSpace($RunRole)) { $RunRole = "replication_n5m_icb_$ReplicationGroupId" }
  if ($RunRole -eq $DefaultRunRole) { throw "Replication runs must not use the canonical paper run role." }
}
$Conditions = if ($C4RMatched) { @("C4", "C4R", "C6") } else { @("C4", "C6") }
$ConditionCsv = $Conditions -join ","
$BudgetedConditionCsv = if ($C4RMatched) { "C4,C4R,C6" } elseif ($BudgetC4) { "C4,C6" } else { "C6" }
$BudgetedConditions = if ($C4RMatched) { @("C4", "C4R", "C6") } elseif ($BudgetC4) { @("C4", "C6") } else { @("C6") }
$RunPrefix = if ($C4RMatched) { "C4R" } elseif ($ExperimentClass -eq "replication") { "N5MR" } elseif ($BudgetC4) { "N5M" } else { "N5F" }
$PolicyToken = if ($C4RMatched) { "C4C4RC6" } else { "C4C6" }
$ManifestStem = if ($C4RMatched) { "c4r_matched_icb" } elseif ($ExperimentClass -eq "replication") { "n5m_replication_${ReplicationGroupId}" } elseif ($BudgetC4) { "n5_matched_budget_frontier_icb" } else { "n5_budget_frontier_icb" }
$Py = if ($env:REPRO_PYTHON_EXE) { $env:REPRO_PYTHON_EXE } else { Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Virtualenv Python missing: $Py" }
if ($InformationCondition -ne "IC-b") { throw "The canonical generation-time frontier is IC-b only." }
Assert-Budget-Grid $L1Budgets $C4RMatched.IsPresent
if (-not $PlanOnly) {
  if ($Backend -match "^openai:" -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) { throw "OPENAI_API_KEY is not set." }
  if ($Backend -match "^anthropic:" -and [string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) { throw "ANTHROPIC_API_KEY is not set." }
}

Set-Location -LiteralPath $Root
$env:PYTHONPATH = (Join-Path $Root "src") + ";" + $env:PYTHONPATH
if ($C4RMatched) {
  & $Py -m credit_recourse.verification.verify_c4r_matched_contract --project-root $Root
  if ($LASTEXITCODE -ne 0) { throw "Integrated C4R matched contract verifier failed." }
} else {
  & $Py -m credit_recourse.verification.verify_n5_budget_frontier_runner --project-root $Root --design $Design
  if ($LASTEXITCODE -ne 0) { throw "N5 budget-frontier runner contract verifier failed." }
}
$help = (& $Py -m credit_recourse.utils.run_llm_stages --help 2>&1 | Out-String)
foreach ($flag in @("--freeform-l1-budget", "--budgeted-conditions", "--run-role")) {
  if ($help -notmatch [regex]::Escape($flag)) { throw "Budget frontier contract flag missing: $flag" }
}
if (-not $PlanOnly) {
  if ($C4RMatched) {
    & $Py -m credit_recourse.verification.smoke_stage7_c4r_contract
    if ($LASTEXITCODE -ne 0) { throw "C4R prompt/revision synthetic smoke failed." }
  } else {
    & $Py -m credit_recourse.verification.smoke_stage7_n5_budget_contract --json
    if ($LASTEXITCODE -ne 0) { throw "N5 synthetic budget-contract verifier failed." }
  }
}

$manifestDir = Join-Path $Root "data\final_freeze\llm_runs\_manifests"
New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
$manifestPath = Join-Path $manifestDir "${ManifestStem}_${DateTag}.json"
$arms = @()
foreach ($budget in $L1Budgets) {
  $budgetToken = Budget-Label $budget
  $GroupToken = if ($ExperimentClass -eq "replication") { "_grp${ReplicationGroupId}" } else { "" }
  $runLabel = "${RunPrefix}_${PolicyToken}_L1_${budgetToken}_ICb_${BackendSlug}_p${CandidateLibraryQuantile}_main_seed${Seed}${GroupToken}_${DateTag}"
  $archiveDir = Join-Path $Root "data\final_freeze\llm_runs\$runLabel"
  $checkpoint = Join-Path $Root "data\final_freeze\stage7_llm_action_generation\checkpoint_${runLabel}.jsonl"
  $checkpointRecovery = Join-Path $Root "data\diagnostics\n5_budget_frontier_checkpoints\checkpoint_${runLabel}.jsonl"
  $argsList = @(
    "-m", "credit_recourse.utils.run_llm_stages",
    "--project-root", $Root,
    "--backend", $Backend,
    "--information-condition", $InformationCondition,
    "--conditions", $ConditionCsv,
    "--modes", "free_form_10d",
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
  if ($Backend -match "^openai:") {
    $argsList += @(
      "--openai-api-mode", "responses",
      "--openai-reasoning-effort", "low",
      "--openai-max-output-tokens", "1200"
    )
  }
  if ($null -ne $budget) {
    $argsList += @(
      "--freeform-l1-budget", ([double]$budget).ToString([Globalization.CultureInfo]::InvariantCulture),
      "--budgeted-conditions", $BudgetedConditionCsv,
      "--budget-contract-label", "${RunPrefix}_${BudgetedConditionCsv}_L1_${budgetToken}_ICb"
    )
  }
  $arms += [ordered]@{
    run_label = $runLabel
    run_role = $RunRole
    experiment_class = $ExperimentClass
    replication_group_id = if ($ExperimentClass -eq "replication") { $ReplicationGroupId } else { $null }
    backend_spec = $Backend
    backend_label = $BackendSlug
    expected_backend_id = $ExpectedBackendId
    information_condition = $InformationCondition
    l1_budget = $budget
    budget_label = $budgetToken
    conditions = $Conditions
    modes = @("free_form_10d")
    budgeted_conditions = if ($null -eq $budget) { @() } else { $BudgetedConditions }
    archive_dir = $archiveDir
    checkpoint = $checkpoint
    checkpoint_recovery = $checkpointRecovery
    command = @($Py) + $argsList
    status = if ($PlanOnly) { "PLANNED" } else { "PENDING" }
  }
}

$c4rAnalysisDir = if ($C4RMatched) {
  Join-Path $Root "data\analysis\c4r_matched\${BackendSlug}_seed${Seed}_${DateTag}"
} else { $null }
$c4rAnalysisCommand = if ($C4RMatched) {
  @($Py, "-m", "credit_recourse.analysis.c4r_matched_inference") +
  @($arms | ForEach-Object { @("--arm-dir", $_.archive_dir) } | ForEach-Object { $_ }) +
  @("--out", $c4rAnalysisDir)
} else { @() }
$replicationAnalysisDir = if ($ExperimentClass -eq "replication") {
  Join-Path $Root "data\analysis\n5m_replications\${ReplicationGroupId}_${DateTag}"
} else { $null }
$replicationAnalysisCommand = if ($ExperimentClass -eq "replication") {
  @(
    $Py, "-m", "credit_recourse.analysis.n5_budget_frontier_holm_inference",
    "--run-dirs"
  ) + @($arms | ForEach-Object { $_.archive_dir }) + @(
    "--out-dir", $replicationAnalysisDir,
    "--run-role", $RunRole,
    "--information-condition", $InformationCondition,
    "--expected-budgets", "0.75", "1.27", "2.00", "unbounded",
    "--design", "matched_c4_c6"
  )
} else { @() }

$manifest = [ordered]@{
  schema_version = "n5_generation_budget_frontier_runner_v3"
  status = if ($PlanOnly) { "PLANNED" } else { "RUNNING" }
  created_utc = [DateTime]::UtcNow.ToString("o")
  project_root = $Root
  date_tag = $DateTag
  run_role = $RunRole
  experiment_class = $ExperimentClass
  replication_group_id = if ($ExperimentClass -eq "replication") { $ReplicationGroupId } else { $null }
  backend_spec = $Backend
  backend_label = $BackendSlug
  expected_backend_id = $ExpectedBackendId
  profile_key = $ProfileKey
  design = $Design
  same_date_batch = $true
  within_arm_control = if ($C4RMatched) { "budget_matched_C4_C4R_C6" } elseif ($BudgetC4) { "budget_matched_C4" } else { "unbudgeted_C4" }
  information_condition = $InformationCondition
  control_condition = "C4"
  revision_conditions = if ($C4RMatched) { @("C4R", "C6") } else { @("C6") }
  budgeted_condition = "C6"
  budgeted_conditions = $BudgetedConditions
  c4r_analysis_dir = $c4rAnalysisDir
  c4r_analysis_command = $c4rAnalysisCommand
  c4r_analysis_status = if ($C4RMatched) { if ($PlanOnly) { "PLANNED" } else { "PENDING" } } else { $null }
  same_batch_required = if ($C4RMatched) { $true } else { $null }
  c4_reuse_from_prior_snapshot_forbidden = if ($C4RMatched) { $true } else { $null }
  replication_analysis_dir = $replicationAnalysisDir
  replication_analysis_command = $replicationAnalysisCommand
  replication_analysis_status = if ($ExperimentClass -eq "replication") { if ($PlanOnly) { "PLANNED" } else { "PENDING" } } else { $null }
  arms = $arms
}
[System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))

foreach ($arm in $arms) {
  Write-Host "`n==== N5 generation-time frontier ($Design): $($arm.run_label) ====" -ForegroundColor Cyan
  Write-Host "CMD> $($arm.command -join ' ')" -ForegroundColor DarkGray
  if ($PlanOnly) { continue }
  $archiveManifest = Join-Path $arm.archive_dir "archive_manifest.json"
  if (Test-Path -LiteralPath $archiveManifest -PathType Leaf) {
    Assert-ArchivedBackend -ArchiveDir $arm.archive_dir -ExpectedBackendId $ExpectedBackendId -ExpectedRunLabel $arm.run_label -ExpectedRunRole $RunRole
    if ($C4RMatched) { Assert-C4R-ArchivedContract -ArchiveDir $arm.archive_dir -ExpectedBudget $arm.l1_budget }
    Write-Host "PASS (existing immutable archive): $($arm.archive_dir)" -ForegroundColor Green
    $arm.status = "PASS"
    [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))
    continue
  }
  if (Test-Path -LiteralPath $arm.archive_dir) {
    throw "Frontier archive directory exists without archive_manifest.json: $($arm.archive_dir)"
  }
  $checkpointPreserved = Preserve-CheckpointAcrossCanonicalCleanup `
    -Checkpoint $arm.checkpoint `
    -RecoveryCopy $arm.checkpoint_recovery
  foreach ($name in @("stage7_llm_action_generation", "stage8_llm_multi_oracle_eval", "stage9_llm_rl_comparison", "stage9_statistical_inference")) {
    $active = Join-Path $Root "data\final_freeze\$name"
    if (Test-Path -LiteralPath $active) { Remove-Item -LiteralPath $active -Recurse -Force }
  }
  if ($checkpointPreserved) {
    $restored = Restore-CheckpointAfterCanonicalCleanup `
      -Checkpoint $arm.checkpoint `
      -RecoveryCopy $arm.checkpoint_recovery
    if (-not $restored) { throw "Failed to restore preserved Stage7 checkpoint: $($arm.checkpoint_recovery)" }
    Write-Host "RESUME checkpoint restored: $($arm.checkpoint)" -ForegroundColor Yellow
  }
  & $Py @($arm.command[1..($arm.command.Count - 1)])
  if ($LASTEXITCODE -ne 0) { throw "N5 budget frontier arm failed: $($arm.run_label)" }
  if (-not (Test-Path -LiteralPath (Join-Path $arm.archive_dir "archive_manifest.json"))) {
    throw "N5 budget frontier archive missing: $($arm.archive_dir)"
  }
  Assert-ArchivedBackend -ArchiveDir $arm.archive_dir -ExpectedBackendId $ExpectedBackendId -ExpectedRunLabel $arm.run_label -ExpectedRunRole $RunRole
  if ($C4RMatched) { Assert-C4R-ArchivedContract -ArchiveDir $arm.archive_dir -ExpectedBudget $arm.l1_budget }
  $arm.status = "PASS"
  [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))
}
if (-not $PlanOnly -and $C4RMatched) {
  Write-Host "`n==== C4R matched inference ====" -ForegroundColor Cyan
  Write-Host "CMD> $($c4rAnalysisCommand -join ' ')" -ForegroundColor DarkGray
  New-Item -ItemType Directory -Path $c4rAnalysisDir -Force | Out-Null
  & $Py @($c4rAnalysisCommand[1..($c4rAnalysisCommand.Count - 1)])
  if ($LASTEXITCODE -ne 0) {
    $manifest.c4r_analysis_status = "FAIL"
    [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))
    throw "C4R matched inference failed."
  }
  $c4rManifest = Join-Path $c4rAnalysisDir "c4r_matched_inference_manifest.json"
  if (-not (Test-Path -LiteralPath $c4rManifest -PathType Leaf)) { throw "C4R inference manifest missing: $c4rManifest" }
  $c4rMeta = Get-Content -LiteralPath $c4rManifest -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$c4rMeta.status -ne "PASS" -or [string]$c4rMeta.run_role -ne "paper_c4r_matched_icb") {
    throw "C4R inference contract mismatch: $c4rManifest"
  }
  $manifest.c4r_analysis_status = "PASS"
  $manifest.c4r_analysis_manifest = $c4rManifest
}
if (-not $PlanOnly -and $ExperimentClass -eq "replication") {
  Write-Host "`n==== N5M replication-group inference: $ReplicationGroupId ====" -ForegroundColor Cyan
  Write-Host "CMD> $($replicationAnalysisCommand -join ' ')" -ForegroundColor DarkGray
  New-Item -ItemType Directory -Path $replicationAnalysisDir -Force | Out-Null
  & $Py @($replicationAnalysisCommand[1..($replicationAnalysisCommand.Count - 1)])
  if ($LASTEXITCODE -ne 0) {
    $manifest.replication_analysis_status = "FAIL"
    [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))
    throw "N5M replication-group inference failed: group=$ReplicationGroupId"
  }
  $analysisManifest = Join-Path $replicationAnalysisDir "n5_budget_frontier_holm_manifest.json"
  if (-not (Test-Path -LiteralPath $analysisManifest -PathType Leaf)) {
    $manifest.replication_analysis_status = "FAIL"
    [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))
    throw "N5M replication-group inference manifest missing: $analysisManifest"
  }
  $analysisMeta = Get-Content -LiteralPath $analysisManifest -Raw -Encoding UTF8 | ConvertFrom-Json
  if ([string]$analysisMeta.status -ne "PASS" -or [string]$analysisMeta.run_role -ne $RunRole -or [string]$analysisMeta.design -ne "matched_c4_c6") {
    $manifest.replication_analysis_status = "FAIL"
    [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))
    throw "N5M replication-group inference contract mismatch: $analysisManifest"
  }
  $manifest.replication_analysis_status = "PASS"
  $manifest.replication_analysis_manifest = $analysisManifest
}
if (-not $PlanOnly) {
  $manifest.status = "PASS"
  $manifest.completed_utc = [DateTime]::UtcNow.ToString("o")
  [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding -ArgumentList $false))
}
Write-Host "`n$($manifest.status): IC-b generation-time budget frontier manifest: $manifestPath" -ForegroundColor Green
