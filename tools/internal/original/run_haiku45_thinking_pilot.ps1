param(
  [Parameter(Mandatory=$true)][string]$ProjectRoot,
  [string]$PreregContractPath = "",
  [Parameter(Mandatory=$true)][string]$PreregEvidence,
  [switch]$PlanOnly,
  [int]$MaxConcurrency = 4,
  [int]$MaxRetries = 3,
  [double]$RetrySleepSeconds = 20.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Python not found: $Py" }
if ([string]::IsNullOrWhiteSpace($PreregContractPath)) {
  $PreregContractPath = Join-Path $Root "src\credit_recourse\configs\c4r_journal_extension_prereg_v4_haiku_thinking_pilot.json"
}
$PreregContractPath = (Resolve-Path -LiteralPath $PreregContractPath).Path
$Contract = Get-Content -LiteralPath $PreregContractPath -Raw -Encoding UTF8 | ConvertFrom-Json
$ExpectedStatus = "LOCKED_THIRD_AMENDMENT_AFTER_HAIKU45_NONTHINKING_FAILURE_BEFORE_HAIKU45_THINKING_PILOT"
if ([string]$Contract.design_status -ne $ExpectedStatus) {
  throw "Unexpected preregistration design_status: $($Contract.design_status)"
}
if (-not $PlanOnly -and [string]::IsNullOrWhiteSpace($PreregEvidence)) {
  throw "-PreregEvidence is required before a live pilot."
}

$Scope = $Contract.pilot_scope
$Provider = $Contract.provider_contract
$Gates = $Contract.quality_gates
$DateTag = [DateTime]::Now.ToString("yyyyMMdd_HHmmss")
$RunLabel = "C4RHTP_C4C4RC6_L1_0p75_ICb_haiku45thinking_b2048_max4096_p50_n50_seed1_${DateTag}"
$RunRole = [string]$Scope.run_role
$ArchiveDir = Join-Path $Root "data\final_freeze\llm_runs\$RunLabel"
$CheckpointDir = Join-Path $Root "data\diagnostics\c4r_journal_checkpoints"
$Checkpoint = Join-Path $CheckpointDir "checkpoint_${RunLabel}.jsonl"
$PilotDir = Join-Path $Root "data\analysis\c4r_journal_extension\haiku_thinking_pilot_${DateTag}"
$BackendQc = Join-Path $PilotDir "haiku45_thinking_backend_contract.json"
$ArmQc = Join-Path $PilotDir "haiku45_thinking_arm_qc.json"
$Manifest = Join-Path $PilotDir "haiku45_thinking_pilot_manifest.json"
New-Item -ItemType Directory -Path $CheckpointDir,$PilotDir -Force | Out-Null

Write-Host "`n==== Haiku 4.5 thinking backend contract verifier ====" -ForegroundColor Cyan
& $Py -m credit_recourse.verification.verify_haiku45_thinking_backend_contract `
  --project-root $Root --out-json $BackendQc
if ($LASTEXITCODE -ne 0) { throw "Haiku thinking backend contract verifier failed." }

$Cmd = @(
  $Py, "-m", "credit_recourse.utils.run_llm_stages",
  "--project-root", $Root,
  "--backend", [string]$Scope.backend,
  "--information-condition", [string]$Scope.information_condition,
  "--conditions", ([string]::Join(",", @($Scope.conditions))),
  "--modes", [string]$Scope.mode,
  "--seed", "1",
  "--reference-draw-seed", "1",
  "--candidate-library-quantile", "$($Scope.candidate_library_quantile)",
  "--checkpoint-path", $Checkpoint,
  "--run-label", $RunLabel,
  "--run-role", $RunRole,
  "--max-concurrency", "$MaxConcurrency",
  "--max-retries", "$MaxRetries",
  "--retry-sleep-seconds", "$RetrySleepSeconds",
  "--anthropic-thinking-budget-tokens", "$($Provider.thinking_budget_tokens)",
  "--anthropic-max-tokens", "$($Provider.max_tokens)",
  "--sample-size", "$($Scope.sample_size)",
  "--sample-seed", "$($Scope.sample_seed)",
  "--sample-strata", ([string]::Join(",", @($Scope.sample_strata))),
  "--freeform-l1-budget", "0.75",
  "--budgeted-conditions", "C4,C4R,C6",
  "--budget-contract-label", "C4RHTP_C4C4RC6_L1_0p75_ICb"
)

$PilotManifest = [ordered]@{
  schema_version = "haiku45_thinking_pilot_runner_v1"
  status = if ($PlanOnly) { "PLANNED" } else { "RUNNING" }
  created_utc = [DateTime]::UtcNow.ToString("o")
  project_root = $Root
  preregistration = [ordered]@{
    path = $PreregContractPath
    sha256 = (Get-FileHash -LiteralPath $PreregContractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    immutable_timestamp_evidence = $PreregEvidence
  }
  run_label = $RunLabel
  run_role = $RunRole
  backend = [string]$Scope.backend
  expected_backend_id = [string]$Scope.expected_backend_id
  thinking_budget_tokens = [int]$Provider.thinking_budget_tokens
  max_tokens = [int]$Provider.max_tokens
  temperature = $null
  sample_size = [int]$Scope.sample_size
  sample_seed = [int]$Scope.sample_seed
  sample_strata = @($Scope.sample_strata)
  planned_request_count = [int]$Scope.planned_live_request_count
  checkpoint = $Checkpoint
  archive_dir = $ArchiveDir
  command = $Cmd
  backend_qc = $BackendQc
  arm_qc = $ArmQc
}
$PilotManifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Manifest -Encoding UTF8

Write-Host "`n==== Haiku 4.5 thinking pilot ====" -ForegroundColor Cyan
Write-Host "CMD> $($Cmd -join ' ')" -ForegroundColor DarkGray
if ($PlanOnly) {
  Write-Host "PLANNED: $Manifest" -ForegroundColor Green
  exit 0
}
if ([string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) {
  throw "ANTHROPIC_API_KEY is not set."
}
if (Test-Path -LiteralPath $ArchiveDir) {
  throw "Immutable archive path already exists: $ArchiveDir"
}

$Exe = [string]$Cmd[0]
$ExeArgs = @($Cmd[1..($Cmd.Count-1)])
& $Exe @ExeArgs
if ($LASTEXITCODE -ne 0) {
  $PilotManifest.status = "FAIL_RUNNER"
  $PilotManifest.completed_utc = [DateTime]::UtcNow.ToString("o")
  $PilotManifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Manifest -Encoding UTF8
  throw "Haiku thinking pilot runner failed."
}

Write-Host "`n==== Haiku 4.5 thinking pilot arm QC ====" -ForegroundColor Cyan
& $Py -m credit_recourse.verification.verify_c4r_journal_arm_contract `
  --archive-dir $ArchiveDir `
  --cohort-id "haiku45thinking" `
  --budget-label "0p75" `
  --l1-budget "0.75" `
  --expected-run-role $RunRole `
  --expected-backend-id ([string]$Scope.expected_backend_id) `
  --expected-firm-count ([int]$Gates.expected_firm_count) `
  --aggregate-raw-compliance-min ([double]$Gates.finite_arm_raw_budget_compliance_aggregate_min) `
  --condition-raw-compliance-min ([double]$Gates.finite_arm_raw_budget_compliance_each_condition_min) `
  --out-json $ArmQc
if ($LASTEXITCODE -ne 0) {
  $PilotManifest.status = "FAIL_QC"
  $PilotManifest.completed_utc = [DateTime]::UtcNow.ToString("o")
  $PilotManifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Manifest -Encoding UTF8
  throw "Haiku thinking pilot QC failed."
}

$Qc = Get-Content -LiteralPath $ArmQc -Raw -Encoding UTF8 | ConvertFrom-Json
if ([int]$Qc.translational_failure_rows -ne 0) {
  throw "Pilot QC unexpectedly retained translational failures."
}
$PilotManifest.status = "PASS_PILOT_QC"
$PilotManifest.completed_utc = [DateTime]::UtcNow.ToString("o")
$PilotManifest.qc_summary = [ordered]@{
  stage7_action_rows = $Qc.stage7_action_rows
  stage8_score_rows = $Qc.stage8_score_rows
  stage9_revision_rows = $Qc.stage9_revision_rows
  aggregate_raw_budget_compliance = $Qc.aggregate_raw_budget_compliance
  condition_raw_budget_compliance = $Qc.condition_raw_budget_compliance
  translational_failure_rows = $Qc.translational_failure_rows
}
$PilotManifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Manifest -Encoding UTF8
Write-Host "PASS_PILOT_QC: $Manifest" -ForegroundColor Green
