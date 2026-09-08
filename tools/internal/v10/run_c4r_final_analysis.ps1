[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectRoot,

  [string]$PythonExe = "",

  [Parameter(Mandatory = $true)][string]$PythonSourceRoot,
  [Parameter(Mandatory = $true)][string]$PythonSitePackagesRoot,
  [Parameter(Mandatory = $true)][string]$PythonEnvironmentConfig,
  [Parameter(Mandatory = $true)][string]$PythonEnvironmentContractId,
  [Parameter(Mandatory = $true)][string]$ExecutionRunRoot,
  [Parameter(Mandatory = $true)][string]$TemporaryRoot,

  [string]$OutputRoot = "",

  [ValidateRange(1, 10000)]
  [int]$ShapleyPermutations = 64,

  [double]$FidelityTolerance = 1e-9,

  [double]$EfficiencyTolerance = 1e-8,

  [switch]$SkipAxis,

  [switch]$Overwrite
)
# Called by the V11 public runner against the canonical working copy.

. (Join-Path $PSScriptRoot '_repro_common.ps1')
. (Join-Path $PSScriptRoot '_checkpoint_common.ps1')


Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try {
  [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
} catch {
  Write-Warning "Unable to lower E3 worker priority: $($_.Exception.Message)"
}

$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = if ([string]::IsNullOrWhiteSpace($PythonExe)) {
  Join-Path $Root ".venv\Scripts\python.exe"
} else {
  [System.IO.Path]::GetFullPath($PythonExe)
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Python executable not found: $Python"
}
$Python = (Resolve-Path -LiteralPath $Python).Path
Initialize-ReproFrozenPythonEnvironment `
  -Root $Root `
  -PythonExe $Python `
  -PythonSourceRoot $PythonSourceRoot `
  -PythonSitePackagesRoot $PythonSitePackagesRoot `
  -PythonEnvironmentConfig $PythonEnvironmentConfig `
  -PythonEnvironmentContractId $PythonEnvironmentContractId `
  -ExecutionRunRoot $ExecutionRunRoot `
  -TemporaryRoot $TemporaryRoot
$CheckpointContext = Get-ReproCheckpointContext `
  -ProjectRoot $Root `
  -PythonExe $Python `
  -ExecutionRunRoot $ExecutionRunRoot

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

function Invoke-PythonModule {
  param(
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )
  Write-Host ""
  Write-Host "==== $Module ====" -ForegroundColor Cyan
  Write-Host ("CMD> `"{0}`" -B -m {1} {2}" -f $Python, $Module, ($Arguments -join " "))
  & $Python -B -m $Module @Arguments
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
Invoke-ReproCheckpointedTask `
  -Context $CheckpointContext `
  -TaskName 'extensions.e3.r1.matched_v3' `
  -OutputRoot $V3Out `
  -Arguments ([ordered]@{cohorts='gpt54mini,gemini31flashlite';budgets='0p75,1p27,2p00,unbounded';expected_firm_count=575;information_condition='IC-b'}) `
  -Inputs ([ordered]@{
    frozen_runs=@($Runs.GetEnumerator()|Sort-Object Key|ForEach-Object{"$($_.Key)=$((@($_.Value.GetEnumerator()|Sort-Object Key|ForEach-Object{"$($_.Key):$(Split-Path -Leaf $_.Value)"})) -join ',')"})
  }) `
  -Execute {
    param($TaskOutput)
    $taskArgs=@($V3Args)
    $taskArgs[$taskArgs.Count-1]=$TaskOutput
    Invoke-PythonModule -Module "credit_recourse.analysis.c4r_matched_inference_v3" -Arguments $taskArgs
  } `
  -Validate {
    param($TaskOutput)
    $manifestPath=Join-Path $TaskOutput 'c4r_matched_inference_v3_manifest.json'
    if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){throw "R1 checkpoint manifest missing: $manifestPath"}
    $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
    if([string]$manifest.status -ne 'PASS' -or [int]$manifest.firm_frame_row_count -ne 4600 -or [int]$manifest.contrast_row_count -ne 72 -or [int]$manifest.interaction_row_count -ne 54){throw 'R1 checkpoint contract failed.'}
  } | Out-Null

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
# CSV byte serialization and Wilcoxon p-value formatting can differ across OS/library builds,


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
    $axisTaskName="extensions.e3.r2.axis.$($Job.Pair.ToLowerInvariant()).$($Job.Cohort)"
    $axisCohort=[string]$Job.Cohort
    $axisPair=[string]$Job.Pair
    $axisBase=[string]$Job.Base
    $axisTarget=[string]$Job.Target
    Invoke-ReproCheckpointedTask `
      -Context $CheckpointContext `
      -TaskName $axisTaskName `
      -OutputRoot $AxisOut `
      -Arguments ([ordered]@{
        cohort=$axisCohort;pair=$axisPair;base_policy=$axisBase;target_policy=$axisTarget
        budgets='0p75,unbounded';shapley_permutations=$ShapleyPermutations;seed=20260714
        fidelity_tolerance=([string]::Format([Globalization.CultureInfo]::InvariantCulture,"{0:R}",$FidelityTolerance))
        efficiency_tolerance=([string]::Format([Globalization.CultureInfo]::InvariantCulture,"{0:R}",$EfficiencyTolerance))
      }) `
      -Inputs ([ordered]@{finite_run=(Split-Path -Leaf $Runs[$axisCohort]['0p75']);unbounded_run=(Split-Path -Leaf $Runs[$axisCohort]['unbounded'])}) `
      -Execute {
        param($TaskOutput)
        $taskArgs=@($AxisArgs)
        $taskArgs[$taskArgs.Count-1]=$TaskOutput
        Invoke-PythonModule -Module "credit_recourse.analysis.n5m_axis_swap_intervention" -Arguments $taskArgs
      } `
      -Validate {
        param($TaskOutput)
        $manifestPath=Join-Path $TaskOutput 'intervention_manifest.json'
        if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){throw "R2 axis checkpoint manifest missing: $manifestPath"}
        $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
        if([string]$manifest.schema -ne 'n5m_axis_swap_intervention_v3'){throw "R2 axis checkpoint contract failed: $axisTaskName"}
      } | Out-Null
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
Invoke-ReproCheckpointedTask `
  -Context $CheckpointContext `
  -TaskName 'extensions.e3.r3.tost_alpha' `
  -OutputRoot $TostOut `
  -Arguments ([ordered]@{primary_oracle='alpha';alpha='0.05';equivalence_margin='0.109'}) `
  -Inputs ([ordered]@{
  }) `
  -Execute {
    param($TaskOutput)
    $taskArgs=@($TostArgs)
    $taskArgs[$taskArgs.Count-3]=$TaskOutput
    Invoke-PythonModule -Module "credit_recourse.analysis.c4r_tost_equivalence" -Arguments $taskArgs
  } `
  -Validate {
    param($TaskOutput)
    $manifestPath=Join-Path $TaskOutput 'metadata.json'
    if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){throw "R3 checkpoint manifest missing: $manifestPath"}
    $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
    if([string]$manifest.status -ne 'PASS' -or [int]$manifest.within_arm_row_count -ne 24 -or [int]$manifest.interaction_row_count -ne 18 -or [double]$manifest.equivalence_margin -ne 0.109){throw 'R3 checkpoint contract failed.'}
  } | Out-Null
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


$FinalStatus = if ($SkipAxis) { "PASS_PARTIAL_R1_R3_AXIS_PENDING" } else { "PASS" }
$FinalManifest = [ordered]@{
  schema_version = "c4r_final_analysis_manifest_v1"
  status = $FinalStatus
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  project_root = $Root
  preregistration_path = $Prereg
  frozen_runs = $Runs
  r1 = [ordered]@{
    status = "PASS"
    output = $V3Out
    firm_frame_rows = 4600
    contrast_rows = 72
    interaction_rows = 54
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
  no_live_api_calls = $true
}
$FinalManifestPath = Join-Path $FinalRoot "final_analysis_manifest.json"
$FinalManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $FinalManifestPath -Encoding UTF8

Write-Host ""
Write-Host "FINAL STATUS: $FinalStatus" -ForegroundColor Green
Write-Host "Final analysis root: $FinalRoot"
Write-Host "Manifest: $FinalManifestPath"
