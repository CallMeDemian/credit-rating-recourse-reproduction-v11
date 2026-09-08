[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectRoot,

  [string]$OutputRoot = "",

  [ValidateRange(1, 10000)]
  [int]$ShapleyPermutations = 64,

  [double]$FidelityTolerance = 1e-9,

  [double]$EfficiencyTolerance = 1e-8,

  [switch]$PlanOnly,

  [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = if ($env:REPRO_PYTHON_EXE) { $env:REPRO_PYTHON_EXE } else { Join-Path $Root ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Thesis venv Python not found: $Python"
}

$RunRoot = Join-Path $Root "data\final_freeze\llm_runs"
if (-not (Test-Path -LiteralPath $RunRoot -PathType Container)) {
  throw "Frozen LLM run root not found: $RunRoot"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
  $FinalRoot = Join-Path $Root "data\analysis\c4r_journal_extension\haiku_axis_compare_$Stamp"
} else {
  $FinalRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}

if (Test-Path -LiteralPath $FinalRoot) {
  if (-not $Overwrite) {
    throw "OutputRoot already exists. Use -Overwrite or choose another path: $FinalRoot"
  }
  Remove-Item -LiteralPath $FinalRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $FinalRoot -Force | Out-Null

$OldPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($OldPythonPath)) {
  $env:PYTHONPATH = (Join-Path $Root "src")
} else {
  $env:PYTHONPATH = (Join-Path $Root "src") + ";" + $OldPythonPath
}

function Invoke-PythonModule {
  param(
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )
  Write-Host ""
  Write-Host "==== $Module ====" -ForegroundColor Cyan
  Write-Host ("CMD> `"{0}`" -m {1} {2}" -f $Python, $Module, ($Arguments -join " "))
  if ($PlanOnly) {
    return
  }
  & $Python -m $Module @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python module failed ($LASTEXITCODE): $Module"
  }
}

function Get-HaikuFrozenRun {
  param(
    [Parameter(Mandatory = $true)][string]$Pattern,
    [Parameter(Mandatory = $true)][string]$ExpectedRole,
    [Parameter(Mandatory = $true)][string]$ExpectedBackendId
  )

  $Candidates = @(
    Get-ChildItem -LiteralPath $RunRoot -Directory |
      Where-Object {
        $_.Name -like $Pattern -and
        $_.Name -notlike "*rehearsal*" -and
        $_.Name -notlike "*n50*"
      } |
      Sort-Object LastWriteTimeUtc -Descending
  )

  $Valid = @()
  foreach ($Dir in $Candidates) {
    $ArchiveManifest = Join-Path $Dir.FullName "archive_manifest.json"
    $Stage7Meta = Join-Path $Dir.FullName "stage7_llm_action_generation\metadata.json"
    $Stage8Meta = Join-Path $Dir.FullName "stage8_llm_multi_oracle_eval\metadata.json"
    $Stage9Meta = Join-Path $Dir.FullName "stage9_llm_rl_comparison\metadata.json"
    if (-not (
      (Test-Path -LiteralPath $ArchiveManifest -PathType Leaf) -and
      (Test-Path -LiteralPath $Stage7Meta -PathType Leaf) -and
      (Test-Path -LiteralPath $Stage8Meta -PathType Leaf) -and
      (Test-Path -LiteralPath $Stage9Meta -PathType Leaf)
    )) {
      continue
    }
    try {
      $Archive = Get-Content -LiteralPath $ArchiveManifest -Raw -Encoding UTF8 | ConvertFrom-Json
      $S7 = Get-Content -LiteralPath $Stage7Meta -Raw -Encoding UTF8 | ConvertFrom-Json
      $S8 = Get-Content -LiteralPath $Stage8Meta -Raw -Encoding UTF8 | ConvertFrom-Json
      $S9 = Get-Content -LiteralPath $Stage9Meta -Raw -Encoding UTF8 | ConvertFrom-Json
      $Conditions = @($S7.conditions | ForEach-Object { [string]$_ })
      $Modes = @($S7.modes | ForEach-Object { [string]$_ })
      if (
        [string]$Archive.run_role -eq $ExpectedRole -and
        [string]$S7.status -eq "PASS" -and
        [string]$S8.status -eq "PASS" -and
        [string]$S9.status -eq "PASS" -and
        [bool]$S7.backend_is_live -eq $true -and
        [string]$S7.backend_id -eq $ExpectedBackendId -and
        [string]$S7.information_condition -eq "IC-b" -and
        [int]$S7.request_count -eq 1725 -and
        [int]$S7.row_count -eq 575 -and
        (($Conditions -join ",") -eq "C4,C4R,C6") -and
        (($Modes -join ",") -eq "free_form_10d")
      ) {
        $Valid += $Dir
      }
    } catch {
      continue
    }
  }

  if ($Valid.Count -ne 1) {
    $Names = @($Valid | ForEach-Object { $_.FullName })
    throw "Expected exactly one canonical Haiku run for pattern=$Pattern; found $($Valid.Count): $($Names -join '; ')"
  }
  return $Valid[0].FullName
}

$NonThinking = Get-HaikuFrozenRun `
  -Pattern "C4RXJ_C4C4RC6_L1_0p75_ICb_haiku45_p50_seed1_*" `
  -ExpectedRole "paper_c4r_journal_ext_haiku45" `
  -ExpectedBackendId "anthropic_claude-haiku-4-5-20251001"

$Thinking = Get-HaikuFrozenRun `
  -Pattern "C4RHTF_C4C4RC6_L1_0p75_ICb_haiku45thinking_b2048_max4096_p50_n575_seed1_*" `
  -ExpectedRole "paper_c4r_haiku45_thinking_full_characterization" `
  -ExpectedBackendId "anthropic_claude-haiku-4-5-20251001_messages_thinking-2048_maxout-4096"

Write-Host ""
Write-Host "Canonical Haiku runs:" -ForegroundColor Green
Write-Host " - nonthinking -> $NonThinking"
Write-Host " - thinking    -> $Thinking"

$CohortOut = Join-Path $FinalRoot "common_cohort"
$CohortArgs = @(
  "--run", "nonthinking=$NonThinking",
  "--run", "thinking=$Thinking",
  "--expected-n", "571",
  "--policies", "C4,C4R,C6",
  "--mode", "free_form_10d",
  "--out", $CohortOut
)
Invoke-PythonModule -Module "credit_recourse.analysis.c4r_haiku_axis_common_cohort" -Arguments $CohortArgs

$CohortCsv = Join-Path $CohortOut "haiku_common_complete_case_row_ids.csv"
if (-not $PlanOnly -and -not (Test-Path -LiteralPath $CohortCsv -PathType Leaf)) {
  throw "Common-cohort CSV missing: $CohortCsv"
}

$Jobs = @(
  @{ Protocol="nonthinking"; Pair="C4_to_C4R"; Base="C4"; Target="C4R"; Run=$NonThinking },
  @{ Protocol="thinking"; Pair="C4_to_C4R"; Base="C4"; Target="C4R"; Run=$Thinking },
  @{ Protocol="nonthinking"; Pair="C4R_to_C6"; Base="C4R"; Target="C6"; Run=$NonThinking },
  @{ Protocol="thinking"; Pair="C4R_to_C6"; Base="C4R"; Target="C6"; Run=$Thinking }
)

$AxisOutputs = @()
foreach ($Job in $Jobs) {
  $AxisOut = Join-Path $FinalRoot ("axis_{0}\{1}" -f $Job.Pair.ToLowerInvariant(), $Job.Protocol)
  $AxisArgs = @(
    "--project-root", $Root,
    "--arm", "0p75=$($Job.Run)",
    "--base-policy", $Job.Base,
    "--target-policy", $Job.Target,
    "--expected-n", "571",
    "--row-id-file", $CohortCsv,
    "--shapley-permutations", [string]$ShapleyPermutations,
    "--shapley-arms", "0p75",
    "--seed", "20260714",
    "--fidelity-tol", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:R}", $FidelityTolerance)),
    "--efficiency-tol", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:R}", $EfficiencyTolerance)),
    "--out", $AxisOut
  )
  Invoke-PythonModule -Module "credit_recourse.analysis.n5m_axis_swap_intervention" -Arguments $AxisArgs
  if (-not $PlanOnly) {
    $Manifest = Join-Path $AxisOut "intervention_manifest.json"
    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
      throw "Axis manifest missing: $Manifest"
    }
  }
  $AxisOutputs += [pscustomobject]@{
    Protocol = $Job.Protocol
    Pair = $Job.Pair
    Directory = $AxisOut
  }
}

$SummaryOut = Join-Path $FinalRoot "summary"
$SummaryArgs = @()
foreach ($Axis in $AxisOutputs) {
  $SummaryArgs += @("--input", "$($Axis.Protocol)|$($Axis.Pair)=$($Axis.Directory)")
}
$SummaryArgs += @("--expected-n", "571", "--out", $SummaryOut)
Invoke-PythonModule -Module "credit_recourse.analysis.c4r_haiku_axis_summary" -Arguments $SummaryArgs

if ($PlanOnly) {
  $Plan = [ordered]@{
    schema_version = "c4r_haiku_axis_compare_plan_v1"
    status = "PLANNED"
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    output_root = $FinalRoot
    common_complete_case_n = 571
    shapley_permutations = $ShapleyPermutations
    nonthinking_run = $NonThinking
    thinking_run = $Thinking
    jobs = $Jobs
    api_calls = 0
  }
  $PlanPath = Join-Path $FinalRoot "haiku_axis_compare_plan.json"
  $Plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $PlanPath -Encoding UTF8
  Write-Host ""
  Write-Host "PLANNED: $PlanPath" -ForegroundColor Yellow
  exit 0
}

$SummaryMeta = Get-Content -LiteralPath (Join-Path $SummaryOut "metadata.json") -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$SummaryMeta.status -ne "PASS") {
  throw "Haiku axis summary failed."
}

$HashCsv = Join-Path $FinalRoot "source_hashes.csv"
$HashRows = @(
  Get-ChildItem -LiteralPath $FinalRoot -File -Recurse |
    Where-Object {
      $_.FullName -ne $HashCsv -and
      $_.Name -ne "haiku_axis_compare_manifest.json"
    } |
    Sort-Object FullName |
    ForEach-Object {
      [pscustomobject]@{
        relative_path = $_.FullName.Substring($FinalRoot.Length).TrimStart('\')
        size_bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
      }
    }
)
$HashRows | Export-Csv -LiteralPath $HashCsv -NoTypeInformation -Encoding UTF8

$GitCommit = $null
try {
  $GitCommit = (& git -C $Root rev-parse HEAD 2>$null).Trim()
} catch {
  $GitCommit = $null
}

$FinalManifest = [ordered]@{
  schema_version = "c4r_haiku_axis_compare_manifest_v1"
  status = "PASS"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  project_root = $Root
  git_commit = $GitCommit
  nonthinking_run = $NonThinking
  thinking_run = $Thinking
  common_complete_case_n = 571
  common_cohort_csv = $CohortCsv
  shapley_permutations = $ShapleyPermutations
  fidelity_tolerance = $FidelityTolerance
  efficiency_tolerance = $EfficiencyTolerance
  axis_outputs = $AxisOutputs
  summary_metadata = (Join-Path $SummaryOut "metadata.json")
  source_hashes = $HashCsv
  api_calls = 0
  evidence_tier = "EXPLORATORY_COMPLETE_CASE_EVALUATOR_ONLY"
  interpretation_boundary = "Both Haiku protocols are evaluated on the same 571-firm complete-case cohort. Neither protocol passed the full raw-budget feasibility contract, so this is mechanism characterization rather than confirmatory contract-compliant model ranking."
}
$FinalManifestPath = Join-Path $FinalRoot "haiku_axis_compare_manifest.json"
$FinalManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $FinalManifestPath -Encoding UTF8

Write-Host ""
Write-Host "HAIKU AXIS COMPARE STATUS: PASS" -ForegroundColor Green
Write-Host "Output: $FinalRoot"
Write-Host "No LLM API calls were made."
