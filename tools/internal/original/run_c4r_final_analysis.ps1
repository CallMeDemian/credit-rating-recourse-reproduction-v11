[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectRoot,

  [string]$OutputRoot = "",

  [ValidateRange(1, 10000)]
  [int]$ShapleyPermutations = 64,

  [double]$FidelityTolerance = 1e-9,

  [double]$EfficiencyTolerance = 1e-8,

  [switch]$SkipAxis,

  [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Thesis venv Python not found: $Python"
}

$Prereg = Join-Path $Root "src\credit_recourse\configs\c4r_journal_extension_prereg_v3.json"
if (-not (Test-Path -LiteralPath $Prereg -PathType Leaf)) {
  throw "C4R preregistration v3 not found: $Prereg"
}

$RunRoot = Join-Path $Root "data\final_freeze\llm_runs"
if (-not (Test-Path -LiteralPath $RunRoot -PathType Container)) {
  throw "Frozen LLM run root not found: $RunRoot"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
  $FinalRoot = Join-Path $Root "data\analysis\c4r_journal_extension\final_$Stamp"
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
  & $Python -m $Module @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python module failed ($LASTEXITCODE): $Module"
  }
}

function Get-FrozenC4RRun {
  param(
    [Parameter(Mandatory = $true)][string]$Cohort,
    [Parameter(Mandatory = $true)][string]$Budget,
    [Parameter(Mandatory = $true)][string]$ExpectedRole
  )

  $Pattern = "C4RXJ_C4C4RC6_L1_${Budget}_ICb_${Cohort}_p50_seed1_*"
  $Candidates = @(
    Get-ChildItem -LiteralPath $RunRoot -Directory |
      Where-Object {
        $_.Name -like $Pattern -and
        $_.Name -notlike "*rehearsal*"
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
      if (
        [string]$Archive.run_role -eq $ExpectedRole -and
        [string]$S7.status -eq "PASS" -and
        [string]$S8.status -eq "PASS" -and
        [string]$S9.status -eq "PASS" -and
        [bool]$S7.backend_is_live -eq $true
      ) {
        $Valid += $Dir
      }
    } catch {
      continue
    }
  }

  if ($Valid.Count -ne 1) {
    $Names = @($Valid | ForEach-Object { $_.FullName })
    throw "Expected exactly one frozen PASS run for cohort=$Cohort budget=$Budget; found $($Valid.Count): $($Names -join '; ')"
  }
  return $Valid[0].FullName
}

$BudgetLabels = @("0p75", "1p27", "2p00", "unbounded")
$RoleByCohort = @{
  "gpt54mini" = "paper_c4r_journal_ext_gpt54mini"
  "gemini31flashlite" = "paper_c4r_journal_ext_gemini31flashlite"
}

$Runs = @{}
foreach ($Cohort in @("gpt54mini", "gemini31flashlite")) {
  $Runs[$Cohort] = @{}
  foreach ($Budget in $BudgetLabels) {
    $Runs[$Cohort][$Budget] = Get-FrozenC4RRun `
      -Cohort $Cohort `
      -Budget $Budget `
      -ExpectedRole $RoleByCohort[$Cohort]
  }
}

Write-Host ""
Write-Host "Frozen 2x4 grid:" -ForegroundColor Green
foreach ($Cohort in @("gpt54mini", "gemini31flashlite")) {
  foreach ($Budget in $BudgetLabels) {
    Write-Host " - $Cohort / $Budget -> $($Runs[$Cohort][$Budget])"
  }
}

# R1: exact eight-arm matched inference.
$V3Out = Join-Path $FinalRoot "v3_matched"
$V3Args = @()
foreach ($Cohort in @("gpt54mini", "gemini31flashlite")) {
  foreach ($Budget in $BudgetLabels) {
    $V3Args += @("--arm", "${Cohort}|${Budget}=$($Runs[$Cohort][$Budget])")
  }
}
$V3Args += @(
  "--expected-run-role", "gpt54mini=$($RoleByCohort['gpt54mini'])",
  "--expected-run-role", "gemini31flashlite=$($RoleByCohort['gemini31flashlite'])",
  "--information-condition", "IC-b",
  "--expected-firm-count", "575",
  "--prereg-path", $Prereg,
  "--out", $V3Out
)
Invoke-PythonModule -Module "credit_recourse.analysis.c4r_matched_inference_v3" -Arguments $V3Args

$V3MetaPath = Join-Path $V3Out "c4r_matched_inference_v3_manifest.json"
if (-not (Test-Path -LiteralPath $V3MetaPath -PathType Leaf)) {
  throw "R1 v3 manifest missing: $V3MetaPath"
}
$V3Meta = Get-Content -LiteralPath $V3MetaPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (
  [string]$V3Meta.status -ne "PASS" -or
  [int]$V3Meta.cohort_count -ne 2 -or
  [int]$V3Meta.firm_frame_row_count -ne 4600 -or
  [int]$V3Meta.contrast_row_count -ne 72 -or
  [int]$V3Meta.interaction_row_count -ne 54
) {
  throw "R1 v3 output contract failed."
}

$InteractionsPath = Join-Path $V3Out "c4r_matched_v3_interactions.csv"
$ContrastsPath = Join-Path $V3Out "c4r_matched_v3_contrasts.csv"
# Byte hashes written by a Linux reference run are retained as provenance only.
# CSV byte serialization and Wilcoxon p-value formatting can differ across OS/library builds,
# so the hard gate is the locally generated manifest SHA plus the semantic ledger contract.
$ReferenceInteractionsSha = "a9b0983a587a698681d153cac1a9c3ce9031b63fb50cc6b52611d1520b69309a"
$ReferenceContrastsSha = "84264b7ff7816fe74cab2555f4d0a4214d3de1a3a733436b3c122e036df1638c"
$ObservedInteractionsSha = (Get-FileHash -LiteralPath $InteractionsPath -Algorithm SHA256).Hash.ToLowerInvariant()
$ObservedContrastsSha = (Get-FileHash -LiteralPath $ContrastsPath -Algorithm SHA256).Hash.ToLowerInvariant()
$ManifestInteractionsSha = ([string]$V3Meta.outputs.interactions.sha256).ToLowerInvariant()
$ManifestContrastsSha = ([string]$V3Meta.outputs.contrasts.sha256).ToLowerInvariant()

if ($ObservedInteractionsSha -ne $ManifestInteractionsSha) {
  throw "R1 interactions file differs from its local v3 manifest: manifest=$ManifestInteractionsSha observed=$ObservedInteractionsSha"
}
if ($ObservedContrastsSha -ne $ManifestContrastsSha) {
  throw "R1 contrasts file differs from its local v3 manifest: manifest=$ManifestContrastsSha observed=$ObservedContrastsSha"
}

$InteractionRows = @(Import-Csv -LiteralPath $InteractionsPath -Encoding UTF8)
$ContrastRows = @(Import-Csv -LiteralPath $ContrastsPath -Encoding UTF8)
if ($InteractionRows.Count -ne 54) {
  throw "R1 interaction ledger must contain 54 rows; got $($InteractionRows.Count)."
}
if ($ContrastRows.Count -ne 72) {
  throw "R1 contrast ledger must contain 72 rows; got $($ContrastRows.Count)."
}

$InteractionKeys = @(
  $InteractionRows |
    ForEach-Object {
      "{0}|{1}|{2}|{3}" -f $_.cohort_id, $_.finite_budget_label, $_.contrast, $_.oracle_backend
    } |
    Sort-Object -Unique
)
$ContrastKeys = @(
  $ContrastRows |
    ForEach-Object {
      "{0}|{1}|{2}|{3}" -f $_.cohort_id, $_.budget_label, $_.contrast, $_.oracle_backend
    } |
    Sort-Object -Unique
)
if ($InteractionKeys.Count -ne 54) {
  throw "R1 interaction ledger has duplicate or missing cohort/budget/contrast/oracle keys."
}
if ($ContrastKeys.Count -ne 72) {
  throw "R1 contrast ledger has duplicate or missing cohort/budget/contrast/oracle keys."
}
if (@($InteractionRows | Where-Object { [int]$_.n_pairs -ne 575 }).Count -gt 0) {
  throw "R1 interaction ledger contains n_pairs other than 575."
}
if (@($ContrastRows | Where-Object { [int]$_.n_pairs -ne 575 }).Count -gt 0) {
  throw "R1 contrast ledger contains n_pairs other than 575."
}

if ($ObservedInteractionsSha -ne $ReferenceInteractionsSha) {
  Write-Warning "R1 interactions byte SHA differs from the Linux reference run. Local manifest SHA and semantic contract passed. reference=$ReferenceInteractionsSha local=$ObservedInteractionsSha"
}
if ($ObservedContrastsSha -ne $ReferenceContrastsSha) {
  Write-Warning "R1 contrasts byte SHA differs from the Linux reference run. Local manifest SHA and semantic contract passed. reference=$ReferenceContrastsSha local=$ObservedContrastsSha"
}
Write-Host "R1 PASS: 4600/72/54, local manifest SHA integrity, and semantic ledger contract." -ForegroundColor Green

# R2: exact simulator/Oracle axis attribution on 0.75 and unbounded.
$AxisManifests = @()
if (-not $SkipAxis) {
  $AxisJobs = @(
    @{ Cohort="gpt54mini"; Pair="C4_to_C4R"; Base="C4"; Target="C4R" },
    @{ Cohort="gemini31flashlite"; Pair="C4_to_C4R"; Base="C4"; Target="C4R" },
    @{ Cohort="gpt54mini"; Pair="C4R_to_C6"; Base="C4R"; Target="C6" },
    @{ Cohort="gemini31flashlite"; Pair="C4R_to_C6"; Base="C4R"; Target="C6" }
  )

  foreach ($Job in $AxisJobs) {
    $AxisOut = Join-Path $FinalRoot ("axis_{0}\{1}" -f $Job.Pair.ToLowerInvariant(), $Job.Cohort)
    $AxisArgs = @(
      "--project-root", $Root,
      "--arm", "0p75=$($Runs[$Job.Cohort]['0p75'])",
      "--arm", "unbounded=$($Runs[$Job.Cohort]['unbounded'])",
      "--base-policy", $Job.Base,
      "--target-policy", $Job.Target,
      "--shapley-permutations", [string]$ShapleyPermutations,
      "--shapley-arms", "0p75,unbounded",
      "--seed", "20260714",
      "--fidelity-tol", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:R}", $FidelityTolerance)),
      "--efficiency-tol", ([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:R}", $EfficiencyTolerance)),
      "--out", $AxisOut
    )
    Invoke-PythonModule -Module "credit_recourse.analysis.n5m_axis_swap_intervention" -Arguments $AxisArgs
    $AxisManifest = Join-Path $AxisOut "intervention_manifest.json"
    if (-not (Test-Path -LiteralPath $AxisManifest -PathType Leaf)) {
      throw "R2 axis manifest missing: $AxisManifest"
    }
    $AxisManifests += [pscustomobject]@{
      Cohort = $Job.Cohort
      Pair = $Job.Pair
      Directory = $AxisOut
      Manifest = $AxisManifest
    }
  }

  $AxisSummaryOut = Join-Path $FinalRoot "axis_summary"
  $AxisSummaryArgs = @()
  foreach ($Axis in $AxisManifests) {
    $AxisSummaryArgs += @("--input", "$($Axis.Cohort)|$($Axis.Pair)=$($Axis.Directory)")
  }
  $AxisSummaryArgs += @("--out", $AxisSummaryOut)
  Invoke-PythonModule -Module "credit_recourse.analysis.c4r_axis_swap_summary" -Arguments $AxisSummaryArgs
  Write-Host "R2 PASS: four exact axis-swap jobs and combined attribution ledger." -ForegroundColor Green
} else {
  Write-Warning "R2 axis-swap skipped by explicit -SkipAxis. Final manifest will be PARTIAL."
}

# R3: preregistered alpha-only TOST, margin read from v3 preregistration.
$TostOut = Join-Path $FinalRoot "tost_alpha"
$V3FirmFramePath = Join-Path $V3Out ([string]$V3Meta.outputs.firm_frame.path)
if (-not (Test-Path -LiteralPath $V3FirmFramePath -PathType Leaf)) {
  throw "R1 firm frame missing: $V3FirmFramePath"
}
$TostArgs = @(
  "--firm-frame", $V3FirmFramePath,
  "--prereg-path", $Prereg,
  "--out", $TostOut,
  "--alpha", "0.05"
)
Invoke-PythonModule -Module "credit_recourse.analysis.c4r_tost_equivalence" -Arguments $TostArgs
$TostMeta = Get-Content -LiteralPath (Join-Path $TostOut "metadata.json") -Raw -Encoding UTF8 | ConvertFrom-Json
if (
  [string]$TostMeta.status -ne "PASS" -or
  [int]$TostMeta.within_arm_row_count -ne 24 -or
  [int]$TostMeta.interaction_row_count -ne 18 -or
  [double]$TostMeta.equivalence_margin -ne 0.109
) {
  throw "R3 TOST output contract failed."
}
Write-Host "R3 PASS: alpha-only TOST, margin ±0.109, 24 within-arm + 18 interaction rows." -ForegroundColor Green

# Freeze commands and file hashes. The hash ledger excludes itself and the final manifest.
$CommandLog = Join-Path $FinalRoot "commands.txt"
@(
  "ProjectRoot=$Root",
  "Preregistration=$Prereg",
  "ShapleyPermutations=$ShapleyPermutations",
  "FidelityTolerance=$FidelityTolerance",
  "EfficiencyTolerance=$EfficiencyTolerance",
  "SkipAxis=$([bool]$SkipAxis)",
  "R1=credit_recourse.analysis.c4r_matched_inference_v3",
  "R2=credit_recourse.analysis.n5m_axis_swap_intervention (C4->C4R and C4R->C6; each cohort; 0p75+unbounded)",
  "R2Summary=credit_recourse.analysis.c4r_axis_swap_summary",
  "R3=credit_recourse.analysis.c4r_tost_equivalence"
) | Set-Content -LiteralPath $CommandLog -Encoding UTF8

$HashCsv = Join-Path $FinalRoot "source_hashes.csv"
$HashRows = @(
  Get-ChildItem -LiteralPath $FinalRoot -File -Recurse |
    Where-Object {
      $_.FullName -ne $HashCsv -and
      $_.Name -ne "final_analysis_manifest.json"
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

$FinalStatus = if ($SkipAxis) { "PASS_PARTIAL_R1_R3_AXIS_PENDING" } else { "PASS" }
$FinalManifest = [ordered]@{
  schema_version = "c4r_final_analysis_manifest_v1"
  status = $FinalStatus
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  project_root = $Root
  git_commit = $GitCommit
  preregistration_path = $Prereg
  preregistration_sha256 = (Get-FileHash -LiteralPath $Prereg -Algorithm SHA256).Hash.ToLowerInvariant()
  frozen_runs = $Runs
  r1 = [ordered]@{
    status = "PASS"
    output = $V3Out
    firm_frame_rows = 4600
    contrast_rows = 72
    interaction_rows = 54
    contrasts_sha256 = $ObservedContrastsSha
    interactions_sha256 = $ObservedInteractionsSha
  }
  r2 = [ordered]@{
    status = $(if ($SkipAxis) { "PENDING" } else { "PASS" })
    shapley_permutations = $ShapleyPermutations
    fidelity_tolerance = $FidelityTolerance
    efficiency_tolerance = $EfficiencyTolerance
    manifests = @($AxisManifests | ForEach-Object { $_.Manifest })
  }
  r3 = [ordered]@{
    status = "PASS"
    output = $TostOut
    primary_oracle = "alpha"
    equivalence_margin = 0.109
    within_arm_rows = 24
    interaction_rows = 18
  }
  hash_ledger = $HashCsv
  no_live_api_calls = $true
}
$FinalManifestPath = Join-Path $FinalRoot "final_analysis_manifest.json"
$FinalManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $FinalManifestPath -Encoding UTF8

Write-Host ""
Write-Host "FINAL STATUS: $FinalStatus" -ForegroundColor Green
Write-Host "Final analysis root: $FinalRoot"
Write-Host "Manifest: $FinalManifestPath"
