<#
Canonical no-API extension analysis for thesis Sections 9.2, 9.6, and 9.7.

Inputs are precomputed Stage7/8/9 LLM run directories already present under
  data\final_freeze\llm_runs
No provider call and no Stage7 replay is performed.
#>
[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$PythonExe = "",
  [Parameter(Mandatory=$true)][string]$PythonSourceRoot,
  [Parameter(Mandatory=$true)][string]$PythonSitePackagesRoot,
  [Parameter(Mandatory=$true)][string]$PythonEnvironmentConfig,
  [Parameter(Mandatory=$true)][string]$PythonEnvironmentContractId,
  [Parameter(Mandatory=$true)][string]$ExecutionRunRoot,
  [Parameter(Mandatory=$true)][string]$TemporaryRoot,
  [ValidateRange(1,2)][int]$MaxParallelAxisTasks=1,
  [string]$OutputRoot = "",
  [switch]$ReplaceOutput,
  [switch]$PlanOnly
)
# V11 calls this preserved V10 extension analysis directly against its
# canonical data/final_freeze working copy. No compatibility view is used.

. (Join-Path $PSScriptRoot '_repro_common.ps1')
. (Join-Path $PSScriptRoot '_checkpoint_common.ps1')

Set-StrictMode -Version Latest
$ErrorActionPreference="Stop"
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

function Resolve-Root([string]$Given) {
  if ($Given) { return (Resolve-Path -LiteralPath $Given).Path }
  $candidate=Split-Path -Parent $PSScriptRoot
  if (-not (Test-Path -LiteralPath (Join-Path $candidate "src\credit_recourse"))) { throw "Cannot infer ProjectRoot." }
  return (Resolve-Path -LiteralPath $candidate).Path
}
function Invoke-Checked([string]$Label,[string[]]$Command) {
  Write-Host "`n==== $Label ====" -ForegroundColor Cyan
  Write-Host "CMD> $($Command -join ' ')" -ForegroundColor DarkGray
  & $Command[0] @($Command[1..($Command.Count-1)])
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $Label exit=$LASTEXITCODE" }
}
function Find-E2Arm([string]$BudgetLabel) {
  $pattern="C4R_C4C4RC6_L1_${BudgetLabel}_ICb_gpt54mini_p50_main_seed1_*"
  $valid=@()
  foreach ($d in @(Get-ChildItem -LiteralPath $RunRoot -Directory -ErrorAction Stop | Where-Object { $_.Name -like $pattern -and $_.Name -notlike "*rehearsal*" })) {
    $am=Join-Path $d.FullName "archive_manifest.json"; $s7=Join-Path $d.FullName "stage7_llm_action_generation\metadata.json"; $s8=Join-Path $d.FullName "stage8_llm_multi_oracle_eval\metadata.json"; $s9=Join-Path $d.FullName "stage9_llm_rl_comparison\metadata.json"
    if (-not ((Test-Path $am) -and (Test-Path $s7) -and (Test-Path $s8) -and (Test-Path $s9))) { continue }
    try {
      $a=Get-Content $am -Raw -Encoding UTF8 | ConvertFrom-Json; $m7=Get-Content $s7 -Raw -Encoding UTF8 | ConvertFrom-Json; $m8=Get-Content $s8 -Raw -Encoding UTF8 | ConvertFrom-Json; $m9=Get-Content $s9 -Raw -Encoding UTF8 | ConvertFrom-Json
      if ([string]$a.run_role -eq "paper_c4r_matched_icb" -and [string]$m7.status -eq "PASS" -and [string]$m8.status -eq "PASS" -and [string]$m9.status -eq "PASS") { $valid += $d.FullName }
    } catch { continue }
  }
  if ($valid.Count -ne 1) { throw "Expected one E2 $BudgetLabel arm, found $($valid.Count): $($valid -join '; ')" }
  return $valid[0]
}

$Root=Resolve-Root $ProjectRoot
$PyCandidate=if([string]::IsNullOrWhiteSpace($PythonExe)){Join-Path $Root ".venv\Scripts\python.exe"}else{$PythonExe}
if (-not (Test-Path -LiteralPath $PyCandidate -PathType Leaf)) { throw "Python executable missing: $PyCandidate" }
$Py=(Resolve-Path -LiteralPath $PyCandidate).Path
Initialize-ReproFrozenPythonEnvironment `
  -Root $Root `
  -PythonExe $Py `
  -PythonSourceRoot $PythonSourceRoot `
  -PythonSitePackagesRoot $PythonSitePackagesRoot `
  -PythonEnvironmentConfig $PythonEnvironmentConfig `
  -PythonEnvironmentContractId $PythonEnvironmentContractId `
  -ExecutionRunRoot $ExecutionRunRoot `
  -TemporaryRoot $TemporaryRoot
$CheckpointContext = Get-ReproCheckpointContext `
  -ProjectRoot $Root `
  -PythonExe $Py `
  -ExecutionRunRoot $ExecutionRunRoot
$RunRoot=Join-Path $Root "data\final_freeze\llm_runs"; if (-not (Test-Path $RunRoot -PathType Container)) { throw "LLM run folder missing: $RunRoot" }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot=Join-Path $Root "data\analysis\paper_repro\05_extension_e3_e4" }
$FinalRoot=[System.IO.Path]::GetFullPath($OutputRoot); $PartialRoot="$FinalRoot.partial"
$E2Finite=Find-E2Arm "0p75"; $E2Unbounded=Find-E2Arm "unbounded"
if ($PlanOnly) {
  Write-Host "PLAN ONLY — no files will be changed." -ForegroundColor Yellow
  Write-Host "E2 finite:     $E2Finite"
  Write-Host "E2 unbounded:  $E2Unbounded"
  Write-Host "E3: tools\run_c4r_final_analysis.ps1 -> $PartialRoot\e3_c4r_journal"
  Write-Host "E4: tools\run_haiku45_axis_compare.ps1 -> $PartialRoot\e4_haiku_characterization"
  exit 0
}
if (Test-Path $PartialRoot) { Remove-Item $PartialRoot -Recurse -Force }
if (Test-Path $FinalRoot) {
  if (-not $ReplaceOutput) { throw "Extension output exists. Use -ReplaceOutput: $FinalRoot" }
}
New-Item -ItemType Directory -Path $PartialRoot -Force | Out-Null
try {
  $E2Out=Join-Path $PartialRoot "e2_c4r_matched"
  Invoke-ReproCheckpointedTask `
    -Context $CheckpointContext `
    -TaskName 'extensions.e2.c4r_matched' `
    -OutputRoot $E2Out `
    -Arguments ([ordered]@{
      module='credit_recourse.analysis.c4r_matched_inference'
      finite_budget='0p75'
      unbounded_budget='unbounded'
    }) `
    -Inputs ([ordered]@{
      finite_run=(Split-Path -Leaf $E2Finite)
      unbounded_run=(Split-Path -Leaf $E2Unbounded)
    }) `
    -Execute {
      param($TaskOutput)
      Invoke-Checked "E2 matched C4/C4R/C6 decomposition" @(
        $Py,"-B","-m","credit_recourse.analysis.c4r_matched_inference",
        "--arm-dir",$E2Finite,"--arm-dir",$E2Unbounded,"--out",$TaskOutput
      )
    } `
    -Validate {
      param($TaskOutput)
      $manifestPath=Join-Path $TaskOutput 'c4r_matched_inference_manifest.json'
      if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){throw "E2 manifest missing: $manifestPath"}
      $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
      if([string]$manifest.status -ne 'PASS' -or [int]$manifest.firm_frame_row_count -ne 1150 -or [int]$manifest.contrast_row_count -ne 18 -or [int]$manifest.interaction_row_count -ne 9){throw 'E2 checkpoint scientific contract failed.'}
    } | Out-Null
  $E3Out=Join-Path $PartialRoot "e3_c4r_journal"
  Invoke-ReproCheckpointedTask `
    -Context $CheckpointContext `
    -TaskName 'extensions.e3.c4r_journal' `
    -OutputRoot $E3Out `
    -Arguments ([ordered]@{shapley_permutations=64;fidelity_tolerance='1e-9';efficiency_tolerance='1e-8'}) `
    -Inputs ([ordered]@{grid='gpt54mini+gemini31flashlite;2x4;IC-b;n=575'}) `
    -Execute {
      param($TaskOutput)
      Invoke-Checked "E3 two-backend matched/TOST/Shapley" @(
        "powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $PSScriptRoot "run_c4r_final_analysis.ps1"),
        "-ProjectRoot",$Root,"-PythonExe",$Py,"-PythonSourceRoot",$PythonSourceRoot,
        "-PythonSitePackagesRoot",$PythonSitePackagesRoot,"-PythonEnvironmentConfig",$PythonEnvironmentConfig,
        "-PythonEnvironmentContractId",$PythonEnvironmentContractId,
        "-ExecutionRunRoot",$ExecutionRunRoot,"-TemporaryRoot",$TemporaryRoot,"-OutputRoot",$TaskOutput,"-ShapleyPermutations","64"
      )
    } `
    -Validate {
      param($TaskOutput)
      $manifestPath=Join-Path $TaskOutput 'final_analysis_manifest.json'
      if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){throw "E3 manifest missing: $manifestPath"}
      $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
      if([string]$manifest.status -ne 'PASS' -or [string]$manifest.r1.status -ne 'PASS' -or [string]$manifest.r2.status -ne 'PASS' -or [string]$manifest.r3.status -ne 'PASS'){throw 'E3 checkpoint scientific contract failed.'}
    } | Out-Null
  $E4Out=Join-Path $PartialRoot "e4_haiku_characterization"
  Invoke-ReproCheckpointedTask `
    -Context $CheckpointContext `
    -TaskName 'extensions.e4.haiku_characterization' `
    -OutputRoot $E4Out `
    -Arguments ([ordered]@{shapley_permutations=64;expected_n=571;fidelity_tolerance='1e-9';efficiency_tolerance='1e-8'}) `
    -Inputs ([ordered]@{protocols='haiku45_nonthinking+haiku45_thinking';policies='C4,C4R,C6'}) `
    -Execute {
      param($TaskOutput)
      Invoke-Checked "E4 Haiku common-cohort Shapley characterization" @(
        "powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $Root "tools\internal\original\run_haiku45_axis_compare.ps1"),
        "-ProjectRoot",$Root,"-OutputRoot",$TaskOutput,"-ShapleyPermutations","64"
      )
    } `
    -Validate {
      param($TaskOutput)
      $manifestPath=Join-Path $TaskOutput 'haiku_axis_compare_manifest.json'
      if(-not(Test-Path -LiteralPath $manifestPath -PathType Leaf)){throw "E4 manifest missing: $manifestPath"}
      $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8|ConvertFrom-Json
      if([string]$manifest.status -ne 'PASS' -or [int]$manifest.common_complete_case_n -ne 571){throw 'E4 checkpoint scientific contract failed.'}
    } | Out-Null
  Invoke-Checked "E4 budget-contract feasibility ladder" @($Py,"-B","-m","credit_recourse.analysis.e4_budget_contract_ladder","--project-root",$Root,"--out",(Join-Path $E4Out "budget_contract_ladder"))
  if (Test-Path $FinalRoot) {
    $backup="$FinalRoot.backup_$(Get-Date -Format yyyyMMdd_HHmmss)"; Move-Item $FinalRoot $backup
    try { Move-Item $PartialRoot $FinalRoot; Remove-Item $backup -Recurse -Force }
    catch { if (Test-Path $FinalRoot) { Remove-Item $FinalRoot -Recurse -Force }; Move-Item $backup $FinalRoot; throw }
  } else { Move-Item $PartialRoot $FinalRoot }
} catch {
  Write-Warning "Extension analysis failed; partial output preserved at $PartialRoot"
  throw
}
Write-Host "`nPASS: canonical no-API E2/E3/E4 extension analysis" -ForegroundColor Green
