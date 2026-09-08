param(
  [string]$ProjectRoot = "C:\Users\Demian\Desktop\thesis_repo",
  [string]$RawAllDir = "",

  # Stage2 reward rebalance. Default/off preserves previous Stage3/4/5/6 runner behavior.
  [double]$Stage2Rho = 0.0,
  [double]$Stage2MertonLambda = 0.05,
  [double]$Stage2FcffLambda = 0.45,
  [double]$Stage2LiquidityLambda = 0.45,
  [switch]$RunStage2,

  [ValidateSet("skip", "train", "restore", "check", "archive_only")]
  [string]$Stage3Mode = "skip",

  [switch]$SkipStage3,
  [switch]$RunStage3,
  [switch]$ArchiveOnly,

  # Stage3 single defaults.
  [int]$Stage3BatchSize = 512,
  [int]$Stage3Epochs = 30,
  [int]$Stage3Seed = 1,
  [double]$Stage3MaskingRatio = 0.15,
  [double]$Stage3LearningRate = 3e-4,
  [double]$Stage3WeightDecay = 1e-5,
  [string]$Stage3SourceDir = "",
  [switch]$AllowMissingStage3Metadata,
  [switch]$AllowStage3MetadataMismatch,

  # Stage3 grids. These are active mainly with -Stage3Mode train.
  [string]$Stage3BatchGrid = "",
  [string]$Stage3EpochGrid = "",
  [string]$Stage3SeedGrid = "",
  [string]$Stage3MaskingRatioGrid = "",
  [string]$Stage3LearningRateGrid = "",
  [string]$Stage3WeightDecayGrid = "",

  # Stage4 / projection defaults and grids.
  [Alias("P")]
  [int]$MagnitudeQuantile = 50,
  [string]$PGrid = "",
  [int]$Stage4Epochs = 80,
  [int]$Stage4BatchSize = 512,
  [int]$Stage4Seed = 0,
  [switch]$NoClassBalanced,
  [double]$ClassBalanceBeta = 0.999,
  [double]$ClassWeightCap = 5.0,
  [switch]$FamilyBalanced,
  [double]$FamilyBalancePower = 0.5,
  [double]$FamilyWeightCap = 3.0,
  [double]$CombinedWeightCap = 5.0,

  [string]$Stage4EpochGrid = "",
  [string]$Stage4BatchGrid = "",
  [string]$Stage4SeedGrid = "",
  [string]$ClassBalancedGrid = "",
  [string]$ClassBalanceBetaGrid = "",
  [string]$ClassWeightCapGrid = "",

  # Stage5 IQL defaults and grids.
  [int]$Stage5Epochs = 150,
  [int]$Stage5BatchSize = 256,
  [int]$Stage5Seed = 0,
  [double]$Gamma = 0.8,
  [Alias("Tau")]
  [double]$ExpectileTau = 0.9,
  [double]$Beta = 10.0,
  [double]$CqlAlpha = 0.0,
  [double]$Stage5LearningRate = 3e-4,
  [double]$Stage5WeightDecay = 1e-4,
  [double]$DistillLambda = 1.0,
  [double]$DistillMarginMin = 0.05,
  [double]$DistillTemperature = 1.0,
  [ValidateSet("linear", "cross_attention", "cross_attention_film")]
  [string]$Stage5CriticHeadArch = "linear",
  [int]$CrossAttnBlocks = 2,
  [int]$CrossAttnHeads = 4,
  [double]$CrossAttnDropout = 0.1,

  [string]$Stage5EpochGrid = "",
  [string]$Stage5BatchGrid = "",
  [string]$Stage5SeedGrid = "",
  [string]$GammaGrid = "",
  [string]$ExpectileTauGrid = "",
  [Alias("TauGrid")]
  [string]$TauGridAlias = "",
  [string]$BetaGrid = "",
  [string]$CqlAlphaGrid = "",
  [string]$Stage5LearningRateGrid = "",
  [string]$Stage5WeightDecayGrid = "",
  [string]$DistillGrid = "",
  [string]$DistillMarginMinGrid = "",
  [string]$DistillTemperatureGrid = "",
  [string]$Stage5CriticHeadArchGrid = "",
  [string]$CrossAttnBlocksGrid = "",
  [string]$CrossAttnHeadsGrid = "",
  [string]$CrossAttnDropoutGrid = "",

  # Stage6 flags and grids.
  [switch]$IncludeStage6Extras,
  [switch]$AllowStage6Unscored,
  [string]$Stage6IncludeExtrasGrid = "",
  [string]$Stage6AllowUnscoredGrid = "",

  # Simulator counterfactual / cash-flow / actor extraction patch toggles.
  # Defaults preserve previous behavior.
  [switch]$CounterfactualTransitions,
  # Reuse an already-created Stage2 counterfactual transition artifact for Stage5 only.
  # This passes --transition-source counterfactual to Stage5 without invoking the Stage2 counterfactual generator.
  [switch]$ReuseCounterfactualTransitionsForStage5,
  [ValidateSet("phi_merton", "phi_merton_fcff", "phi_merton_liquidity", "phi_merton_fcff_liquidity")]
  [string]$CounterfactualRewardMode = "phi_merton",
  [ValidateSet("terminal", "bootstrap")]
  [string]$CounterfactualDoneMode = "terminal",
  [ValidateSet("strict", "warn", "off")]
  [string]$CounterfactualFidelityGate = "strict",
  [double]$CounterfactualMaxRelErrAssets = 0.05,
  [switch]$JoinCashFlowSubstrate,
  [ValidateSet("reward_only", "full")]
  [string]$CashFlowEncoderMode = "reward_only",
  [string]$CashFlowPanel = "",
  [ValidateSet("awr", "distill_only_finetune")]
  [string]$ActorExtractionMode = "awr",
  [int]$ActorFinetuneSteps = 200,
  [ValidateSet("linear", "action_conditioned")]
  [string]$ActorHeadArch = "linear",
  [switch]$DeployQArgmaxAsPolicy,
  [ValidateSet("actor_policy_q", "critic_value_greedy", "q_pareto_knee")]
  [string]$Stage5SelectionMetric = "actor_policy_q",
  [switch]$PreserveCurrentNonCurrentResidual,
  [ValidateSet("default", "calibrated")]
  [string]$SimBusinessPlanMode = "default",

  # Run controls.
  [switch]$NoClean,
  [switch]$SkipStage4,
  [switch]$SkipStage5,
  [switch]$ContinueOnCellFailure,
  [switch]$SkipStage6Inference,
  [ValidateSet("warn", "strict", "skip")]
  [string]$VerifierMode = "warn",
  [switch]$ZipArchive,
  [switch]$NoSourceSnapshot,
  [string]$RunLabel = "",
  [string]$ExistingArchiveRoot = "",
  [int]$MaxCells = 0,

  # Phase0 v3 / seed-lineage controls. These options keep this single runner
  # usable for all-stage aligned multi-seed cells without exploding into the
  # Stage3 x Stage4 x Stage5 Cartesian product.
  [switch]$Phase0V3Preset,
  [switch]$AlignStageSeeds,
  [string]$AlignedSeedGrid = "",
  [switch]$IncludeExactAnchorCell,
  [int]$AnchorStage3Seed = 1,
  [int]$AnchorStage4Seed = 0,
  [int]$AnchorStage5Seed = 0,

  # Phase1 cross-attention preset. This reuses an already-restored Stage3
  # encoder and runs Stage4/5/6 with the cross-attention critic head.
  [switch]$Phase1CrossAttentionPreset,

  # H1 preset: Phase1 cross-attention with Stage5 LR lowered to 1e-4.
  # This preset is intentionally separate from Phase0V3Preset so that H1 does
  # not silently inherit Phase0's linear/head or Stage3-train semantics.
  [switch]$H1CrossAttentionLr1e4Preset
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Preserve the user-supplied parameter values before any preset mutates defaults.
# This prevents presets from silently clobbering experiment-defining overrides
# such as -Stage5LearningRate 1e-4.
$ExplicitParamValues = @{}
foreach ($k in $PSBoundParameters.Keys) { $ExplicitParamValues[$k] = $PSBoundParameters[$k] }

function Restore-ExplicitParam {
  param([string]$Name)
  if ($ExplicitParamValues.ContainsKey($Name)) {
    Set-Variable -Name $Name -Value $ExplicitParamValues[$Name] -Scope Script
  }
}

function Restore-ExplicitExperimentParams {
  $names = @(
    "RawAllDir", "Stage2Rho", "Stage2MertonLambda", "Stage2FcffLambda", "Stage2LiquidityLambda", "RunStage2",
    "Stage3Mode", "Stage3BatchSize", "Stage3Epochs", "Stage3Seed", "Stage3MaskingRatio", "Stage3LearningRate", "Stage3WeightDecay", "Stage3SourceDir",
    "Stage3BatchGrid", "Stage3EpochGrid", "Stage3SeedGrid", "Stage3MaskingRatioGrid", "Stage3LearningRateGrid", "Stage3WeightDecayGrid",
    "MagnitudeQuantile", "PGrid", "Stage4Epochs", "Stage4BatchSize", "Stage4Seed", "ClassBalanceBeta", "ClassWeightCap", "FamilyBalanced", "FamilyBalancePower", "FamilyWeightCap", "CombinedWeightCap",
    "Stage4EpochGrid", "Stage4BatchGrid", "Stage4SeedGrid", "ClassBalancedGrid", "ClassBalanceBetaGrid", "ClassWeightCapGrid",
    "Stage5Epochs", "Stage5BatchSize", "Stage5Seed", "Gamma", "ExpectileTau", "Beta", "CqlAlpha", "Stage5LearningRate", "Stage5WeightDecay",
    "DistillLambda", "DistillMarginMin", "DistillTemperature", "Stage5CriticHeadArch", "CrossAttnBlocks", "CrossAttnHeads", "CrossAttnDropout",
    "Stage5EpochGrid", "Stage5BatchGrid", "Stage5SeedGrid", "GammaGrid", "ExpectileTauGrid", "TauGridAlias", "BetaGrid", "CqlAlphaGrid",
    "Stage5LearningRateGrid", "Stage5WeightDecayGrid", "DistillGrid", "DistillMarginMinGrid", "DistillTemperatureGrid", "Stage5CriticHeadArchGrid",
    "CrossAttnBlocksGrid", "CrossAttnHeadsGrid", "CrossAttnDropoutGrid", "CounterfactualTransitions", "ReuseCounterfactualTransitionsForStage5", "CounterfactualRewardMode", "CounterfactualDoneMode", "CounterfactualFidelityGate", "CounterfactualMaxRelErrAssets", "JoinCashFlowSubstrate", "CashFlowEncoderMode", "CashFlowPanel", "ActorExtractionMode", "ActorFinetuneSteps", "ActorHeadArch", "DeployQArgmaxAsPolicy", "Stage5SelectionMetric", "PreserveCurrentNonCurrentResidual", "SimBusinessPlanMode", "NoClean", "SkipStage4", "SkipStage5", "VerifierMode", "RunLabel", "MaxCells"
  )
  foreach ($n in $names) { Restore-ExplicitParam $n }
}

$RUNNER_VERSION = "UNIFIED_RL_STAGE23456_V17_STAGE5_CF_REUSE_20260606"

if ($SkipStage3) { $Stage3Mode = "skip" }
if ($RunStage3)  { $Stage3Mode = "train" }
if ($ArchiveOnly) { $Stage3Mode = "archive_only" }
if ($RunStage2 -and $Stage3Mode -eq "archive_only") { throw "-RunStage2 conflicts with -ArchiveOnly because Stage2 refresh mutates active final_freeze artifacts." }
if ($TauGridAlias -and $TauGridAlias.Trim().Length -gt 0) { $ExpectileTauGrid = $TauGridAlias }
$Stage5UseCounterfactualTransitions = ($CounterfactualTransitions -or $ReuseCounterfactualTransitionsForStage5)
if ($ReuseCounterfactualTransitionsForStage5 -and $CounterfactualTransitions) {
  Write-Warning "-ReuseCounterfactualTransitionsForStage5 is redundant because -CounterfactualTransitions already regenerates and forwards counterfactual transitions."
}
if ($CounterfactualTransitions -and $CounterfactualRewardMode -in @("phi_merton_fcff", "phi_merton_fcff_liquidity") -and -not $JoinCashFlowSubstrate) { throw "-CounterfactualRewardMode $CounterfactualRewardMode requires -JoinCashFlowSubstrate when Stage2 counterfactual transitions are regenerated." }
if ($JoinCashFlowSubstrate -and -not $RunStage2) { throw "-JoinCashFlowSubstrate requires -RunStage2 so Stage2 input_splits can be regenerated." }
if ($CounterfactualTransitions -and [math]::Abs($Stage2LiquidityLambda) -gt 1e-12 -and ($CounterfactualRewardMode -notin @("phi_merton_liquidity", "phi_merton_fcff_liquidity"))) { throw "-Stage2LiquidityLambda requires -CounterfactualRewardMode phi_merton_liquidity or phi_merton_fcff_liquidity when -CounterfactualTransitions is used." }
if ($CashFlowEncoderMode -eq "full") { Write-Warning "CashFlowEncoderMode=full changes encoder feature distribution; rerun Stage3/4/5/6 and expect new hashes." }

$presetCount = @(($Phase0V3Preset, $Phase1CrossAttentionPreset, $H1CrossAttentionLr1e4Preset) | Where-Object { $_ }).Count
if ($presetCount -gt 1) {
  throw "Use only one preset among -Phase0V3Preset, -Phase1CrossAttentionPreset, and -H1CrossAttentionLr1e4Preset. H1 is Phase1 cross_attention with Stage5 LR=1e-4."
}

if ($Phase0V3Preset) {
  $phase0Conflicts = @()
  if ($SkipStage3) { $phase0Conflicts += "-SkipStage3 conflicts with -Phase0V3Preset because Phase0 owns Stage3 training." }
  if ($ArchiveOnly) { $phase0Conflicts += "-ArchiveOnly conflicts with -Phase0V3Preset." }
  if (($PSBoundParameters.ContainsKey("Stage3Mode")) -and ($Stage3Mode -ne "train")) {
    $phase0Conflicts += "-Stage3Mode $Stage3Mode conflicts with -Phase0V3Preset. Do not use Phase0 preset for Phase1 Stage3 reuse."
  }
  if (($PSBoundParameters.ContainsKey("Stage5CriticHeadArch")) -and ($Stage5CriticHeadArch -ne "linear")) {
    $phase0Conflicts += "-Stage5CriticHeadArch $Stage5CriticHeadArch conflicts with -Phase0V3Preset. Use -Phase1CrossAttentionPreset or explicit Phase1 args instead."
  }
  if ($phase0Conflicts.Count -gt 0) {
    throw ("Invalid Phase0 preset combination:`n - " + ($phase0Conflicts -join "`n - "))
  }
}

if ($Phase1CrossAttentionPreset -or $H1CrossAttentionLr1e4Preset) {
  if ($AlignStageSeeds -or ($AlignedSeedGrid -and $AlignedSeedGrid.Trim().Length -gt 0)) {
    throw "-Phase1CrossAttentionPreset reuses one manually-restored Stage3 encoder at a time. Do not use -AlignStageSeeds/-AlignedSeedGrid. Pass Stage3Seed/Stage4Seed/Stage5Seed explicitly for each seed."
  }
  $Stage3Mode = "skip"
  $Stage3BatchSize = 512
  $Stage3Epochs = 30
  $Stage3MaskingRatio = 0.15
  $Stage3LearningRate = 3e-4
  $Stage3WeightDecay = 1e-5
  $MagnitudeQuantile = 50
  $Stage4Epochs = 80
  $Stage4BatchSize = 512
  $NoClassBalanced = $false
  $ClassBalanceBeta = 0.999
  $ClassWeightCap = 5.0
  $Stage5Epochs = 150
  $Stage5BatchSize = 256
  $Gamma = 0.8
  $ExpectileTau = 0.9
  $Beta = 10.0
  $CqlAlpha = 0.0
  $Stage5LearningRate = 3e-4
  if ($H1CrossAttentionLr1e4Preset) { $Stage5LearningRate = 1e-4 }
  $Stage5WeightDecay = 1e-4
  $DistillLambda = 1.0
  $DistillMarginMin = 0.05
  $DistillTemperature = 1.0
  $Stage5CriticHeadArch = "cross_attention"
  $CrossAttnBlocks = 2
  $CrossAttnHeads = 4
  $CrossAttnDropout = 0.1
  if (-not $RunLabel -or $RunLabel.Trim().Length -eq 0) {
    if ($H1CrossAttentionLr1e4Preset) { $RunLabel = "h1_cross_attention_lr1e4" } else { $RunLabel = "phase1_cross_attention" }
  }
}

if ($Phase0V3Preset) {
  $Stage3Mode = "train"
  $Stage3BatchSize = 512
  $Stage3Epochs = 30
  $Stage3MaskingRatio = 0.15
  $Stage3LearningRate = 3e-4
  $Stage3WeightDecay = 1e-5
  $MagnitudeQuantile = 50
  $Stage4Epochs = 80
  $Stage4BatchSize = 512
  $NoClassBalanced = $false
  $ClassBalanceBeta = 0.999
  $ClassWeightCap = 5.0
  $Stage5Epochs = 150
  $Stage5BatchSize = 256
  $Gamma = 0.8
  $ExpectileTau = 0.9
  $Beta = 10.0
  $CqlAlpha = 0.0
  $Stage5LearningRate = 3e-4
  $Stage5WeightDecay = 1e-4
  $DistillLambda = 1.0
  $DistillMarginMin = 0.05
  $DistillTemperature = 1.0
  $Stage5CriticHeadArch = "linear"
  $AlignStageSeeds = $true
  if (-not $AlignedSeedGrid -or $AlignedSeedGrid.Trim().Length -eq 0) { $AlignedSeedGrid = "1,2026,2030" }
  if (-not $RunLabel -or $RunLabel.Trim().Length -eq 0) { $RunLabel = "phase0_v3" }
}

# Re-apply explicit experiment parameters after presets. Presets define defaults;
# explicit command-line values remain authoritative.
Restore-ExplicitExperimentParams
if ($SkipStage3) { $Stage3Mode = "skip" }
if ($RunStage3)  { $Stage3Mode = "train" }
if ($ArchiveOnly) { $Stage3Mode = "archive_only" }
if ($RunStage2 -and $Stage3Mode -eq "archive_only") { throw "-RunStage2 conflicts with -ArchiveOnly because Stage2 refresh mutates active final_freeze artifacts." }
if ($TauGridAlias -and $TauGridAlias.Trim().Length -gt 0) { $ExpectileTauGrid = $TauGridAlias }
$Stage5UseCounterfactualTransitions = ($CounterfactualTransitions -or $ReuseCounterfactualTransitionsForStage5)
if ($ReuseCounterfactualTransitionsForStage5 -and $CounterfactualTransitions) {
  Write-Warning "-ReuseCounterfactualTransitionsForStage5 is redundant because -CounterfactualTransitions already regenerates and forwards counterfactual transitions."
}
if ($CounterfactualTransitions -and $CounterfactualRewardMode -in @("phi_merton_fcff", "phi_merton_fcff_liquidity") -and -not $JoinCashFlowSubstrate) { throw "-CounterfactualRewardMode $CounterfactualRewardMode requires -JoinCashFlowSubstrate when Stage2 counterfactual transitions are regenerated." }
if ($JoinCashFlowSubstrate -and -not $RunStage2) { throw "-JoinCashFlowSubstrate requires -RunStage2 so Stage2 input_splits can be regenerated." }
if ($CounterfactualTransitions -and [math]::Abs($Stage2LiquidityLambda) -gt 1e-12 -and ($CounterfactualRewardMode -notin @("phi_merton_liquidity", "phi_merton_fcff_liquidity"))) { throw "-Stage2LiquidityLambda requires -CounterfactualRewardMode phi_merton_liquidity or phi_merton_fcff_liquidity when -CounterfactualTransitions is used." }
if ($CashFlowEncoderMode -eq "full") { Write-Warning "CashFlowEncoderMode=full changes encoder feature distribution; rerun Stage3/4/5/6 and expect new hashes." }

function Normalize-Token {
  param([object]$Value)
  $s = ([string]$Value)
  return (($s -replace "\.", "p") -replace "-", "m")
}

function Get-ShortHash {
  param([string]$Text, [int]$Length = 10)
  $sha = [System.Security.Cryptography.SHA256]::Create()
  try {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $hashBytes = $sha.ComputeHash($bytes)
    $hex = ([System.BitConverter]::ToString($hashBytes)).Replace("-", "").ToLowerInvariant()
    if ($Length -lt 1) { $Length = 10 }
    if ($Length -gt $hex.Length) { $Length = $hex.Length }
    return $hex.Substring(0, $Length)
  } finally {
    $sha.Dispose()
  }
}

function Parse-IntGrid {
  param([string]$GridText, [int]$DefaultValue)
  if ($null -eq $GridText -or $GridText.Trim().Length -eq 0) { return @($DefaultValue) }
  $items = @()
  foreach ($raw in ($GridText -split ",")) {
    $t = $raw.Trim()
    if ($t.Length -eq 0) { continue }
    $items += [int]$t
  }
  if ($items.Count -eq 0) { throw "Empty integer grid: $GridText" }
  return @($items)
}

function Parse-DoubleGrid {
  param([string]$GridText, [double]$DefaultValue)
  if ($null -eq $GridText -or $GridText.Trim().Length -eq 0) { return @($DefaultValue) }
  $items = @()
  foreach ($raw in ($GridText -split ",")) {
    $t = $raw.Trim()
    if ($t.Length -eq 0) { continue }
    $items += [double]$t
  }
  if ($items.Count -eq 0) { throw "Empty double grid: $GridText" }
  return @($items)
}

function Parse-BoolGrid {
  param([string]$GridText, [bool]$DefaultValue)
  if ($null -eq $GridText -or $GridText.Trim().Length -eq 0) { return @($DefaultValue) }
  $items = @()
  foreach ($raw in ($GridText -split ",")) {
    $t = $raw.Trim().ToLowerInvariant()
    if ($t.Length -eq 0) { continue }
    if ($t -in @("1","true","t","yes","y","on")) { $items += $true; continue }
    if ($t -in @("0","false","f","no","n","off")) { $items += $false; continue }
    throw "Invalid boolean grid item '$raw' in '$GridText'. Use true,false or 1,0."
  }
  if ($items.Count -eq 0) { throw "Empty boolean grid: $GridText" }
  return @($items)
}

function Parse-StringGrid {
  param([string]$GridText, [string]$DefaultValue, [string[]]$AllowedValues = @())
  if ($null -eq $GridText -or $GridText.Trim().Length -eq 0) { return @($DefaultValue) }
  $items = @()
  foreach ($raw in ($GridText -split ",")) {
    $t = $raw.Trim()
    if ($t.Length -eq 0) { continue }
    if ($AllowedValues.Count -gt 0 -and -not ($AllowedValues -contains $t)) {
      throw "Invalid string grid item '$raw' in '$GridText'. Allowed: $($AllowedValues -join ',')"
    }
    $items += $t
  }
  if ($items.Count -eq 0) { throw "Empty string grid: $GridText" }
  return @($items)
}

function Assert-Exists {
  param([string]$Path, [string]$Label)
  if (-not (Test-Path -LiteralPath $Path)) { throw "MISSING: $Label => $Path" }
}

function Assert-NonEmptyDir {
  param([string]$Path, [string]$Label)
  Assert-Exists $Path $Label
  $items = Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue
  if (-not $items -or $items.Count -eq 0) { throw "EMPTY_DIR: $Label => $Path" }
}

function Read-JsonFile {
  param([string]$Path)
  Assert-Exists $Path "json"
  return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
}

function Has-Prop {
  param([object]$Obj, [string]$Name)
  if ($null -eq $Obj) { return $false }
  return ($Obj.PSObject.Properties.Name -contains $Name)
}

function Copy-TreeIfExists {
  param([string]$Src, [string]$Dst)
  if (-not (Test-Path -LiteralPath $Src)) { return }
  $item = Get-Item -LiteralPath $Src
  if ($item.PSIsContainer) {
    New-Item -ItemType Directory -Force -Path $Dst | Out-Null
    & robocopy $Src $Dst /E /R:2 /W:2 /NFL /NDL /NJH /NJS /NP /XD __pycache__ .pytest_cache /XF *.pyc | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -gt 7) { throw "ROBOCOPY_FAILED: src=$Src dst=$Dst exit=$rc" }
    $global:LASTEXITCODE = 0
  } else {
    New-Item -ItemType Directory -Force -Path (Split-Path $Dst -Parent) | Out-Null
    Copy-Item -LiteralPath $Src -Destination $Dst -Force
  }
}

function Remove-DirIfExists {
  param([string]$Path)
  if (Test-Path -LiteralPath $Path) {
    Write-Host "[REMOVE] $Path"
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

function ConvertTo-ProcessArgumentLine {
  param([string[]]$CommandArgs)
  $parts = @()
  foreach ($a in $CommandArgs) {
    if ($null -eq $a) { continue }
    $s = [string]$a
    if ($s -match '[\s"`]') {
      $escaped = $s -replace '"', '"'
      $parts += '"' + $escaped + '"'
    } else {
      $parts += $s
    }
  }
  return ($parts -join " ")
}

function Invoke-LoggedProcess {
  param(
    [string]$Name,
    [string]$Exe,
    [string[]]$CommandArgs,
    [string]$LogDir
  )
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $safeFull = ($Name -replace '[^A-Za-z0-9_.-]+', '_').Trim('_')
  if ([string]::IsNullOrWhiteSpace($safeFull)) { $safeFull = "process" }
  $nameHash = Get-ShortHash -Text $Name -Length 10
  $safe = $safeFull
  if ($safe.Length -gt 24) { $safe = $safe.Substring(0, 24).Trim('_') }
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  # Keep file names short. Long command/cell names are recorded inside the log body.
  $stdout = Join-Path $LogDir "${stamp}_${nameHash}.out.log"
  $stderr = Join-Path $LogDir "${stamp}_${nameHash}.err.log"
  $merged = Join-Path $LogDir "${stamp}_${nameHash}.log"
  New-Item -ItemType Directory -Force -Path (Split-Path $merged -Parent) | Out-Null
  $argLine = ConvertTo-ProcessArgumentLine -CommandArgs $CommandArgs

  Write-Host ""
  Write-Host "============================================================"
  Write-Host "[$Name]"
  Write-Host "============================================================"
  Write-Host ("& " + $Exe + " " + $argLine)
  Write-Host "Log = $merged"

  if ($argLine.Trim().Length -gt 0) {
    $proc = Start-Process -FilePath $Exe -ArgumentList $argLine -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  } else {
    $proc = Start-Process -FilePath $Exe -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  }

  "# COMMAND" | Set-Content -Encoding UTF8 -Path $merged
  ("& " + $Exe + " " + $argLine) | Add-Content -Encoding UTF8 -Path $merged
  "`n# STDOUT" | Add-Content -Encoding UTF8 -Path $merged
  if (Test-Path -LiteralPath $stdout) { Get-Content -LiteralPath $stdout | Add-Content -Encoding UTF8 -Path $merged }
  "`n# STDERR" | Add-Content -Encoding UTF8 -Path $merged
  if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr | Add-Content -Encoding UTF8 -Path $merged }

  if ($proc.ExitCode -ne 0) {
    $marker = Join-Path $LogDir "LAST_FAILED_LOG.txt"
    Set-Content -Encoding UTF8 -Path $marker -Value $merged
    Write-Host ""
    Write-Host "[FAILED LOG TAIL: $Name]"
    Write-Host "LogPath = $merged"
    Write-Host "Marker  = $marker"
    Write-Host "------------------------------------------------------------"
    if (Test-Path -LiteralPath $merged) { Get-Content -LiteralPath $merged -Tail 220 }
    Write-Host "------------------------------------------------------------"
    throw "FAILED: $Name exit=$($proc.ExitCode). Inspect log: $merged"
  }

  if (Test-Path -LiteralPath $merged) {
    Write-Host "[OK] $Name"
    Get-Content -LiteralPath $merged -Tail 40 | ForEach-Object { Write-Host $_ }
  }
  return $merged
}

function Write-DirManifest {
  param([string]$Root, [string]$OutCsv)
  $rows = @()
  if (Test-Path -LiteralPath $Root) {
    $files = Get-ChildItem -LiteralPath $Root -Recurse -File -Force -ErrorAction SilentlyContinue
    foreach ($f in $files) {
      try {
        if ($f.FullName -eq $OutCsv) { continue }
        if ($f.FullName -match '(\\|/)__pycache__(\\|/)' -or $f.Name -like '*.pyc') { continue }
        if (($f.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
          Write-Warning "SKIP manifest reparse-point: $($f.FullName)"
          continue
        }
        if (-not (Test-Path -LiteralPath $f.FullName -PathType Leaf)) {
          Write-Warning "SKIP manifest missing leaf: $($f.FullName)"
          continue
        }
        $rel = $f.FullName.Substring($Root.Length).TrimStart([char]92, [char]47)
        $hash = Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256 -ErrorAction Stop
        $rows += [pscustomobject]@{
          relative_path = $rel
          bytes = $f.Length
          sha256 = $hash.Hash.ToLowerInvariant()
          modified_utc = $f.LastWriteTimeUtc.ToString("o")
        }
      } catch {
        Write-Warning "SKIP manifest hash failure: $($f.FullName) :: $($_.Exception.Message)"
        continue
      }
    }
  }
  $rows | Sort-Object relative_path | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutCsv
}

function Get-ModuleHelpText {
  param([string]$Module)
  $out = & $Py -m $Module --help 2>&1
  $txt = ($out | Out-String)
  return [pscustomobject]@{ ok = ($LASTEXITCODE -eq 0); exit_code = $LASTEXITCODE; text = $txt }
}

function Test-CliOption {
  param([string]$HelpText, [string]$Option)
  return ($HelpText -match [regex]::Escape($Option))
}

function Add-OptionalArg {
  param(
    [string[]]$ArgsIn,
    [string]$HelpText,
    [string]$Option,
    [object]$Value,
    [string]$Label
  )
  $argsOut = @($ArgsIn)
  if (Test-CliOption -HelpText $HelpText -Option $Option) {
    $argsOut += @($Option, "$Value")
  } else {
    Write-Warning "$Label CLI does not support $Option; requested value '$Value' will be recorded in manifest only."
  }
  return @($argsOut)
}

function Add-OptionalSwitch {
  param(
    [string[]]$ArgsIn,
    [string]$HelpText,
    [string]$Option,
    [bool]$Enabled,
    [string]$Label
  )
  $argsOut = @($ArgsIn)
  if (-not $Enabled) { return @($argsOut) }
  if (Test-CliOption -HelpText $HelpText -Option $Option) {
    $argsOut += @($Option)
  } else {
    Write-Warning "$Label CLI does not support $Option; switch ignored."
  }
  return @($argsOut)
}

function Check-NumericMetadata {
  param([object]$Meta, [string]$Field, [double]$Expected, [string]$Label, [switch]$AllowMissing, [switch]$AllowMismatch)
  if (-not (Has-Prop $Meta $Field)) {
    if ($AllowMissing) { Write-Warning "$Label missing metadata field: $Field"; return }
    throw "$Label missing metadata field: $Field"
  }
  $actual = [double]$Meta.$Field
  $diff = [Math]::Abs($actual - $Expected)
  if ($diff -gt 1e-12) {
    $msg = "$Label $Field mismatch: actual=$actual expected=$Expected"
    if ($AllowMismatch) { Write-Warning $msg } else { throw $msg }
  }
}

function Assert-Stage4Reusable {
  param(
    [string]$Stage4Dir,
    [int]$ExpectedEpochs,
    [int]$ExpectedBatchSize,
    [int]$ExpectedSeed,
    [int]$ExpectedMagnitudeQuantile,
    [bool]$ExpectedClassBalanced,
    [double]$ExpectedClassBalanceBeta,
    [double]$ExpectedClassWeightCap,
    [bool]$ExpectedFamilyBalanced = $false,
    [double]$ExpectedFamilyBalancePower = 0.5,
    [double]$ExpectedFamilyWeightCap = 3.0,
    [double]$ExpectedCombinedWeightCap = 5.0
  )

  Assert-Exists $Stage4Dir "Stage4 directory for -SkipStage4"
  Assert-Exists (Join-Path $Stage4Dir "metadata.json") "Stage4 metadata.json for -SkipStage4"
  Assert-NonEmptyDir $Stage4Dir "Stage4 artifact directory for -SkipStage4"

  $candidateFiles = @(Get-ChildItem -LiteralPath $Stage4Dir -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match '(?i)(\.pt$|\.pth$|checkpoint|model|bc)'
  })
  if (-not $candidateFiles -or $candidateFiles.Count -eq 0) {
    throw "Stage4 skip requested but no plausible checkpoint/model artifact found under $Stage4Dir"
  }

  $meta = Read-JsonFile (Join-Path $Stage4Dir "metadata.json")
  Check-NumericMetadata $meta "epochs" $ExpectedEpochs "Stage4(reused)"
  if (Has-Prop $meta "batch_size") { Check-NumericMetadata $meta "batch_size" $ExpectedBatchSize "Stage4(reused)" }
  if (Has-Prop $meta "seed") { Check-NumericMetadata $meta "seed" $ExpectedSeed "Stage4(reused)" }
  if (Has-Prop $meta "magnitude_quantile") { Check-NumericMetadata $meta "magnitude_quantile" $ExpectedMagnitudeQuantile "Stage4(reused)" }
  if (Has-Prop $meta "class_balanced") {
    $actualCb = [System.Convert]::ToBoolean($meta.class_balanced)
    if ($actualCb -ne $ExpectedClassBalanced) {
      throw "Stage4(reused) class_balanced mismatch: actual=$actualCb expected=$ExpectedClassBalanced"
    }
  }
  if (Has-Prop $meta "class_balance_beta") { Check-NumericMetadata $meta "class_balance_beta" $ExpectedClassBalanceBeta "Stage4(reused)" }
  if (Has-Prop $meta "class_weight_cap") { Check-NumericMetadata $meta "class_weight_cap" $ExpectedClassWeightCap "Stage4(reused)" }

  $fbField = $null
  if (Has-Prop $meta "family_balanced_loss") { $fbField = "family_balanced_loss" }
  elseif (Has-Prop $meta "family_balanced") { $fbField = "family_balanced" }
  if ($null -ne $fbField) {
    $actualFb = [System.Convert]::ToBoolean($meta.$fbField)
    if ($actualFb -ne $ExpectedFamilyBalanced) {
      throw "Stage4(reused) family_balanced mismatch: actual=$actualFb expected=$ExpectedFamilyBalanced. If reusing a family-balanced Stage4 artifact, pass -FamilyBalanced to the runner."
    }
  } elseif ($ExpectedFamilyBalanced) {
    throw "Stage4(reused) metadata has no family_balanced_loss/family_balanced field, but -FamilyBalanced was requested."
  }
  if ($ExpectedFamilyBalanced) {
    if (Has-Prop $meta "family_balance_power") { Check-NumericMetadata $meta "family_balance_power" $ExpectedFamilyBalancePower "Stage4(reused)" }
    if (Has-Prop $meta "family_weight_cap") { Check-NumericMetadata $meta "family_weight_cap" $ExpectedFamilyWeightCap "Stage4(reused)" }
    if (Has-Prop $meta "combined_weight_cap") { Check-NumericMetadata $meta "combined_weight_cap" $ExpectedCombinedWeightCap "Stage4(reused)" }
    Assert-Exists (Join-Path $Stage4Dir "family_balance_audit.csv") "Stage4 family_balance_audit.csv for -SkipStage4 -FamilyBalanced"
    Assert-Exists (Join-Path $Stage4Dir "stage4_label_quality_audit.csv") "Stage4 stage4_label_quality_audit.csv for -SkipStage4 -FamilyBalanced"
  }

  Write-Host "[Stage4 reusable config verified] epochs=$ExpectedEpochs batch_size=$ExpectedBatchSize seed=$ExpectedSeed p=$ExpectedMagnitudeQuantile class_balanced=$ExpectedClassBalanced family_balanced=$ExpectedFamilyBalanced"
}

function Assert-Stage5EffectiveConfig {
  param(
    [string]$Stage5Dir,
    [double]$ExpectedLearningRate,
    [double]$ExpectedWeightDecay,
    [string]$ExpectedCriticHeadArch,
    [int]$ExpectedCrossAttnBlocks,
    [int]$ExpectedCrossAttnHeads,
    [double]$ExpectedCrossAttnDropout
  )
  $metaPath = Join-Path $Stage5Dir "metadata.json"
  Assert-Exists $metaPath "Stage5 metadata.json"
  $meta = Read-JsonFile $metaPath
  Check-NumericMetadata $meta "learning_rate" $ExpectedLearningRate "Stage5 metadata"
  Check-NumericMetadata $meta "weight_decay" $ExpectedWeightDecay "Stage5 metadata"
  if (Has-Prop $meta "critic_head_arch") {
    if ([string]$meta.critic_head_arch -ne $ExpectedCriticHeadArch) {
      throw "Stage5 metadata critic_head_arch mismatch: actual=$($meta.critic_head_arch) expected=$ExpectedCriticHeadArch"
    }
  } else {
    throw "Stage5 metadata missing field: critic_head_arch"
  }
  if ($ExpectedCriticHeadArch -in @("cross_attention", "cross_attention_film")) {
    Check-NumericMetadata $meta "cross_attn_blocks" $ExpectedCrossAttnBlocks "Stage5 metadata"
    Check-NumericMetadata $meta "cross_attn_heads" $ExpectedCrossAttnHeads "Stage5 metadata"
    Check-NumericMetadata $meta "cross_attn_dropout" $ExpectedCrossAttnDropout "Stage5 metadata"
  }

  $logPath = Join-Path $Stage5Dir "training_log.csv"
  Assert-Exists $logPath "Stage5 training_log.csv"
  $rows = @(Import-Csv -LiteralPath $logPath)
  if ($rows.Count -le 0) { throw "Stage5 training_log.csv is empty: $logPath" }
  $lrVals = @($rows | Where-Object { $_.learning_rate -ne $null -and [string]$_.learning_rate -ne "" } | ForEach-Object { [double]$_.learning_rate } | Sort-Object -Unique)
  if ($lrVals.Count -ne 1) { throw "Stage5 training_log learning_rate must have exactly one unique value; got=$($lrVals -join ',')" }
  if ([Math]::Abs(([double]$lrVals[0]) - $ExpectedLearningRate) -gt 1e-12) {
    throw "Stage5 training_log learning_rate mismatch: actual=$($lrVals[0]) expected=$ExpectedLearningRate"
  }
  $wdVals = @($rows | Where-Object { $_.weight_decay -ne $null -and [string]$_.weight_decay -ne "" } | ForEach-Object { [double]$_.weight_decay } | Sort-Object -Unique)
  if ($wdVals.Count -eq 1 -and [Math]::Abs(([double]$wdVals[0]) - $ExpectedWeightDecay) -gt 1e-12) {
    throw "Stage5 training_log weight_decay mismatch: actual=$($wdVals[0]) expected=$ExpectedWeightDecay"
  }
  $stage5Artifacts = @(Get-ChildItem -LiteralPath $Stage5Dir -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match '(?i)(candidate_iql_policy|final_refit|fulltrain|\.pt$|\.pth$)'
  })
  if (-not $stage5Artifacts -or $stage5Artifacts.Count -eq 0) {
    throw "Stage5 reusable check failed: no plausible policy/checkpoint artifact found under $Stage5Dir"
  }
  Write-Host "[Stage5 effective config verified] learning_rate=$ExpectedLearningRate weight_decay=$ExpectedWeightDecay critic_head_arch=$ExpectedCriticHeadArch artifacts=$($stage5Artifacts.Count)"
}

function Print-Stage3Metadata {
  param([string]$Stage3Dir)
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "[Stage3 metadata check]"
  Write-Host "============================================================"
  Write-Host "Stage3Dir = $Stage3Dir"
  $meta = Join-Path $Stage3Dir "metadata.json"
  if (Test-Path -LiteralPath $meta) {
    $m = Read-JsonFile $meta
    foreach ($k in @("train_mode", "batch_size", "train_batch_size", "epochs", "seed", "masking_ratio", "learning_rate", "weight_decay", "best_val_loss", "feature_schema_hash", "stage3_backward_compat_final_refit_alias_written")) {
      if (Has-Prop $m $k) { Write-Host "$k = $($m.$k)" }
    }
  } else {
    Write-Warning "metadata.json not found: $meta"
  }

  foreach ($name in @("ssl_encoder.pt", "stage3_encoder_avs256_final_refit_fulltrain.pt", "stage3_encoder_avs256_innerdev_winner.pt")) {
    $p = Join-Path $Stage3Dir $name
    if (Test-Path -LiteralPath $p) {
      $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
      Write-Host "[CKPT] $name sha256=$h"
    } else {
      Write-Host "[MISSING] $name"
    }
  }
}

function Restore-Stage3FromSource {
  param([string]$Src, [string]$Dst)
  if (-not $Src -or $Src.Trim().Length -eq 0) { throw "Stage3Mode=restore requires -Stage3SourceDir" }
  Assert-NonEmptyDir $Src "Stage3SourceDir"
  if (Test-Path -LiteralPath $Dst) { Remove-Item -LiteralPath $Dst -Recurse -Force }
  Copy-TreeIfExists $Src $Dst
  Assert-Exists (Join-Path $Dst "ssl_encoder.pt") "restored Stage3 ssl_encoder.pt"
  Assert-Exists (Join-Path $Dst "metadata.json") "restored Stage3 metadata.json"
}

function Copy-PolicySummaryIfExists {
  param([string]$Stage6MultiDir, [string]$ArchiveRoot)
  $src = Join-Path $Stage6MultiDir "final_policy_summary.csv"
  if (Test-Path -LiteralPath $src) {
    Copy-Item -LiteralPath $src -Destination (Join-Path $ArchiveRoot "00_POLICY_SUMMARY_CORE.csv") -Force
  }
}

function Run-Verifier {
  param([string]$Root, [string]$Py, [string]$FF, [string]$VerifierMode, [string]$ArchiveRoot, [string]$LogDir)
  $status = [ordered]@{
    verifier_mode = $VerifierMode
    attempted = $false
    exit_code = $null
    verify_all_json = Join-Path $FF "ledgers\verify_all.json"
    archived_status_json = Join-Path $ArchiveRoot "00_VERIFIER_STATUS.json"
  }

  if ($VerifierMode -eq "skip") {
    $status.attempted = $false
    $status.status = "SKIPPED"
    $status | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path (Join-Path $ArchiveRoot "00_VERIFIER_STATUS.json")
    return $status
  }

  $status.attempted = $true
  $vargs = @("-m", "credit_recourse.verification.stage_boundary_contracts", "--project-root", $Root, "--stage", "all")
  try {
    Invoke-LoggedProcess -Name "Stage boundary verifier" -Exe $Py -CommandArgs $vargs -LogDir $LogDir | Out-Null
    $status.exit_code = 0
    $status.status = "PASS"
  } catch {
    $status.exit_code = 1
    $status.status = if ($VerifierMode -eq "strict") { "FAILED_STRICT" } else { "FAILED_WARNING" }
    $status.error = [string]$_.Exception.Message
    Write-Warning "Stage boundary verifier failed. Mode=$VerifierMode. Inspect $($status.verify_all_json)"
    if ($VerifierMode -eq "strict") {
      $status | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path (Join-Path $ArchiveRoot "00_VERIFIER_STATUS.json")
      throw
    }
  }

  if (Test-Path -LiteralPath $status.verify_all_json) {
    Copy-Item -LiteralPath $status.verify_all_json -Destination (Join-Path $ArchiveRoot "00_VERIFIER_STATUS.json") -Force
  } else {
    $status | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path (Join-Path $ArchiveRoot "00_VERIFIER_STATUS.json")
  }
  return $status
}

function Archive-CurrentCell {
  param(
    [string]$ArchiveRoot,
    [string]$CellId,
    [string]$Status,
    [string]$FailureMessage,
    [object]$CellConfig,
    [string]$Root,
    [string]$FF,
    [string]$Stage3Dir,
    [string]$Stage4Dir,
    [string]$Stage5Dir,
    [string]$Stage6SelectorDir,
    [string]$Stage6MultiDir,
    [switch]$NoSourceSnapshot,
    [switch]$ZipArchive
  )

  New-Item -ItemType Directory -Force -Path $ArchiveRoot | Out-Null
  $outputs = Join-Path $ArchiveRoot "outputs"
  New-Item -ItemType Directory -Force -Path $outputs | Out-Null

  Copy-TreeIfExists $Stage3Dir (Join-Path $outputs "stage3_acd_ssl")
  Copy-TreeIfExists $Stage4Dir (Join-Path $outputs "stage4_candidate_bc")
  Copy-TreeIfExists $Stage5Dir (Join-Path $outputs "stage5_candidate_iql")
  Copy-TreeIfExists $Stage6SelectorDir (Join-Path $outputs "stage6_candidate_selector_eval")
  Copy-TreeIfExists $Stage6MultiDir (Join-Path $outputs "stage6_multi_oracle_eval")

  Copy-TreeIfExists (Join-Path $FF "configs") (Join-Path $ArchiveRoot "configs")
  Copy-TreeIfExists (Join-Path $FF "ledgers") (Join-Path $ArchiveRoot "ledgers")
  Copy-TreeIfExists (Join-Path $FF "logs") (Join-Path $ArchiveRoot "logs")
  Copy-PolicySummaryIfExists -Stage6MultiDir $Stage6MultiDir -ArchiveRoot $ArchiveRoot

  if (-not $NoSourceSnapshot) {
    Copy-TreeIfExists (Join-Path $Root "src") (Join-Path $ArchiveRoot "src_snapshot")
    Copy-TreeIfExists (Join-Path $Root "tools") (Join-Path $ArchiveRoot "tools_snapshot")
  }

  $manifest = [ordered]@{
    runner_version = $RUNNER_VERSION
    cell_id = $CellId
    status = $Status
    failure_message = $FailureMessage
    created_utc = (Get-Date).ToUniversalTime().ToString("o")
    project_root = $Root
    archive_root = $ArchiveRoot
    config = $CellConfig
  }
  $manifest | ConvertTo-Json -Depth 30 | Set-Content -Encoding UTF8 -Path (Join-Path $ArchiveRoot "00_CELL_MANIFEST.json")

  $readmeLines = @(
    "# Unified RL Stage3/4/5/6 Cell Archive",
    "",
    "- runner_version: $RUNNER_VERSION",
    "- cell_id: $CellId",
    "- status: $Status",
    "- failure_message: $FailureMessage",
    "- archive_root: $ArchiveRoot",
    "",
    "## Config",
    "",
    '```text',
    "Stage3Mode        = $($CellConfig.stage3.mode)",
    "Stage3BatchSize   = $($CellConfig.stage3.batch_size)",
    "Stage3Epochs      = $($CellConfig.stage3.epochs)",
    "Stage3Seed        = $($CellConfig.stage3.seed)",
    "SeedLineage       = $($CellConfig.stage3.seed_lineage_id) [$($CellConfig.stage3.seed_lineage_mode)]",
    "MagnitudeQuantile = $($CellConfig.stage4.magnitude_quantile)",
    "Stage4Epochs      = $($CellConfig.stage4.epochs)",
    "Stage4BatchSize   = $($CellConfig.stage4.batch_size)",
    "Stage4Seed        = $($CellConfig.stage4.seed)",
    "ClassBalanced     = $($CellConfig.stage4.class_balanced)",
    "ClassBalanceBeta  = $($CellConfig.stage4.class_balance_beta)",
    "ClassWeightCap    = $($CellConfig.stage4.class_weight_cap)",
    "FamilyBalanced    = $($CellConfig.stage4.family_balanced)",
    "FamilyBalancePower= $($CellConfig.stage4.family_balance_power)",
    "FamilyWeightCap   = $($CellConfig.stage4.family_weight_cap)",
    "CombinedWeightCap = $($CellConfig.stage4.combined_weight_cap)",
    "Stage5Mode        = $($CellConfig.stage5.mode)",
    "Stage5Epochs      = $($CellConfig.stage5.epochs)",
    "Stage5BatchSize   = $($CellConfig.stage5.batch_size)",
    "Stage5Seed        = $($CellConfig.stage5.seed)",
    "Gamma             = $($CellConfig.stage5.gamma)",
    "ExpectileTau      = $($CellConfig.stage5.expectile_tau)",
    "Beta              = $($CellConfig.stage5.beta)",
    "CqlAlpha          = $($CellConfig.stage5.cql_alpha)",
    "Stage5LR          = $($CellConfig.stage5.learning_rate)",
    "Stage5WD          = $($CellConfig.stage5.weight_decay)",
    "DistillLambda     = $($CellConfig.stage5.distill_lambda)",
    "DistillMarginMin  = $($CellConfig.stage5.distill_margin_min)",
    "DistillTemp       = $($CellConfig.stage5.distill_temperature)",
    "CriticHeadArch    = $($CellConfig.stage5.critic_head_arch)",
    "CrossAttn         = blocks=$($CellConfig.stage5.cross_attn_blocks), heads=$($CellConfig.stage5.cross_attn_heads), dropout=$($CellConfig.stage5.cross_attn_dropout)",
    "Stage6Extras      = $($CellConfig.stage6.include_extras)",
    "AllowUnscored     = $($CellConfig.stage6.allow_unscored)",
    "VerifierMode      = $($CellConfig.verifier.mode)",
    '```'
  )
  $readme = ($readmeLines -join [Environment]::NewLine) + [Environment]::NewLine
  $readme | Set-Content -Encoding UTF8 -Path (Join-Path $ArchiveRoot "00_README_RUN_STATUS.md")

  Write-DirManifest -Root $ArchiveRoot -OutCsv (Join-Path $ArchiveRoot "ARCHIVE_SHA256_MANIFEST.csv")

  if ($ZipArchive) {
    $zip = "$ArchiveRoot.zip"
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    try {
      Compress-Archive -Path $ArchiveRoot -DestinationPath $zip -Force -ErrorAction Stop
    } catch {
      Write-Warning "ZIP archive failed but cell archive directory is preserved: $($_.Exception.Message)"
    }
  }
}

function Expand-Grid {
  param([object[]]$Cells, [string]$Name, [object[]]$Values)
  $out = @()
  foreach ($cell in $Cells) {
    foreach ($v in $Values) {
      $clone = [ordered]@{}
      foreach ($k in $cell.Keys) { $clone[$k] = $cell[$k] }
      $clone[$Name] = $v
      $out += $clone
    }
  }
  return @($out)
}

$Root = (Resolve-Path $ProjectRoot).Path
$FF = Join-Path $Root "data\final_freeze"
$Py = if ($env:REPRO_PYTHON_EXE) { $env:REPRO_PYTHON_EXE } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Src = Join-Path $Root "src"
$Stage3Dir = Join-Path $FF "stage3_acd_ssl"
$Stage4Dir = Join-Path $FF "stage4_candidate_bc"
$Stage5Dir = Join-Path $FF "stage5_candidate_iql"
$Stage6SelectorDir = Join-Path $FF "stage6_candidate_selector_eval"
$Stage6MultiDir = Join-Path $FF "stage6_multi_oracle_eval"

Assert-Exists $Py "Python venv"
Assert-Exists $Src "src"
$env:PYTHONPATH = $Src
Set-Location $Root

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLabelSafe = if ($RunLabel -and $RunLabel.Trim().Length -gt 0) { ($RunLabel -replace '[^A-Za-z0-9_.-]+', '_') } else { "run" }
if ($runLabelSafe.Length -gt 48) {
  $runLabelHash = Get-ShortHash -Text $runLabelSafe -Length 8
  $runLabelSafe = $runLabelSafe.Substring(0, 39).Trim('_') + "_" + $runLabelHash
}
$RunRoot = if ($ExistingArchiveRoot -and $ExistingArchiveRoot.Trim().Length -gt 0) { $ExistingArchiveRoot } else { Join-Path $FF "rl_unified_${runLabelSafe}_${timestamp}" }
$RunLogDir = Join-Path $RunRoot "run_logs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

# Reduce CUDA fragmentation for Stage5 cross-attention runs. This does not alter
# model semantics; it only helps PyTorch's allocator on 8GB laptop GPUs.
if (-not $env:PYTORCH_CUDA_ALLOC_CONF -or $env:PYTORCH_CUDA_ALLOC_CONF.Trim().Length -eq 0) {
  $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}
New-Item -ItemType Directory -Force -Path $RunLogDir | Out-Null

# Read live CLI support from the actual local src. This makes the runner work with both
# patched local code and older uploaded src snapshots that do not expose lr/wd CLI args.
$Stage2Help = Get-ModuleHelpText "credit_recourse.rl.pipelines.final_stage2_candidate_projection.pipeline"
$Stage3Help = Get-ModuleHelpText "credit_recourse.rl.pipelines.final_stage3_acd_ssl.pipeline"
$Stage4Help = Get-ModuleHelpText "credit_recourse.rl.pipelines.final_stage4_candidate_bc.pipeline"
$Stage5Help = Get-ModuleHelpText "credit_recourse.rl.pipelines.final_stage5_candidate_iql.pipeline"
$CounterfactualHelp = Get-ModuleHelpText "credit_recourse.rl.pipelines.final_stage2_counterfactual_transitions.pipeline"
$Stage6SelectorHelp = Get-ModuleHelpText "credit_recourse.eval.final_stage6_candidate_selector_eval.pipeline"
$Stage6MultiHelp = Get-ModuleHelpText "credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline"

$CliSupport = [ordered]@{
  stage2_rho = (Test-CliOption $Stage2Help.text "--rho")
  stage2_merton_lambda = (Test-CliOption $Stage2Help.text "--merton-lambda")
  stage2_fcff_lambda = (Test-CliOption $Stage2Help.text "--fcff-lambda")
  stage2_liquidity_lambda = (Test-CliOption $Stage2Help.text "--liquidity-lambda")
  stage3_learning_rate = (Test-CliOption $Stage3Help.text "--learning-rate")
  stage3_weight_decay = (Test-CliOption $Stage3Help.text "--weight-decay")
  stage4_class_balanced = (Test-CliOption $Stage4Help.text "--class-balanced")
  stage4_family_balanced = (Test-CliOption $Stage4Help.text "--family-balanced")
  stage4_family_balance_power = (Test-CliOption $Stage4Help.text "--family-balance-power")
  stage4_family_weight_cap = (Test-CliOption $Stage4Help.text "--family-weight-cap")
  stage4_combined_weight_cap = (Test-CliOption $Stage4Help.text "--combined-weight-cap")
  stage5_learning_rate = (Test-CliOption $Stage5Help.text "--learning-rate")
  stage5_weight_decay = (Test-CliOption $Stage5Help.text "--weight-decay")
  stage5_actor_distill = (Test-CliOption $Stage5Help.text "--actor-distill-lambda")
  stage5_critic_head_arch = (Test-CliOption $Stage5Help.text "--critic-head-arch")
  stage5_cross_attn_blocks = (Test-CliOption $Stage5Help.text "--cross-attn-blocks")
  stage5_cross_attn_heads = (Test-CliOption $Stage5Help.text "--cross-attn-heads")
  stage5_cross_attn_dropout = (Test-CliOption $Stage5Help.text "--cross-attn-dropout")
  stage5_transition_source = (Test-CliOption $Stage5Help.text "--transition-source")
  stage5_selection_metric = (Test-CliOption $Stage5Help.text "--selection-metric")
  stage5_actor_extraction_mode = (Test-CliOption $Stage5Help.text "--actor-extraction-mode")
  stage5_actor_finetune_steps = (Test-CliOption $Stage5Help.text "--actor-finetune-steps")
  stage5_actor_head_arch = (Test-CliOption $Stage5Help.text "--actor-head-arch")
  counterfactual_available = ($CounterfactualHelp.ok -and (Test-CliOption $CounterfactualHelp.text "--reward-mode"))
  counterfactual_fidelity_gate = (Test-CliOption $CounterfactualHelp.text "--fidelity-gate")
  counterfactual_sim_business_plan_mode = (Test-CliOption $CounterfactualHelp.text "--sim-business-plan-mode")
  counterfactual_preserve_residual = (Test-CliOption $CounterfactualHelp.text "--preserve-current-non-current-residual")
  stage6_include_extras = (Test-CliOption $Stage6SelectorHelp.text "--include-stage6-extras")
  stage6_allow_unscored = (Test-CliOption $Stage6MultiHelp.text "--allow-unscored")
  stage6_preserve_residual = (Test-CliOption $Stage6MultiHelp.text "--preserve-current-non-current-residual")
  stage6_sim_business_plan_mode = (Test-CliOption $Stage6MultiHelp.text "--sim-business-plan-mode")
  stage6_deploy_qargmax = (Test-CliOption $Stage6MultiHelp.text "--deploy-qargmax-as-policy")
}
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$CliSupport | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path (Join-Path $RunRoot "CLI_SUPPORT_DETECTED.json")
if ($RunStage2 -and (-not $CliSupport.stage2_rho)) { throw "Stage2 CLI does not expose --rho; aborting to prevent manifest-only rho drift." }
if ($RunStage2 -and [math]::Abs($Stage2MertonLambda) -gt 1e-12 -and (-not $CliSupport.stage2_merton_lambda)) { throw "Stage2 CLI does not expose --merton-lambda; aborting to prevent manifest-only Merton reward drift." }
if ($RunStage2 -and [math]::Abs($Stage2FcffLambda) -gt 1e-12 -and (-not $CliSupport.stage2_fcff_lambda)) { throw "Stage2 CLI does not expose --fcff-lambda; aborting to prevent manifest-only FCFF reward drift." }
if ($RunStage2 -and [math]::Abs($Stage2LiquidityLambda) -gt 1e-12 -and (-not $CliSupport.stage2_liquidity_lambda)) { throw "Stage2 CLI does not expose --liquidity-lambda; aborting to prevent manifest-only liquidity reward drift." }
if (-not $CliSupport.stage5_learning_rate) { throw "Stage5 CLI does not expose --learning-rate; aborting to prevent manifest-only LR drift." }
if (-not $CliSupport.stage5_weight_decay) { throw "Stage5 CLI does not expose --weight-decay; aborting to prevent manifest-only WD drift." }
if (($Stage5CriticHeadArch -eq "cross_attention" -or $Stage5CriticHeadArchGrid -match "cross_attention") -and (-not $CliSupport.stage5_critic_head_arch)) {
  throw "Stage5 CLI does not expose --critic-head-arch; aborting to prevent manifest-only critic-head drift."
}
if (($Stage5CriticHeadArch -eq "cross_attention_film" -or $Stage5CriticHeadArchGrid -match "cross_attention_film") -and ($Stage5Help.text -notmatch "cross_attention_film")) {
  throw "Stage5 CLI exposes --critic-head-arch but does not accept cross_attention_film; update src before running FiLM critic."
}
if ($CounterfactualTransitions -and (-not $CliSupport.counterfactual_available)) { throw "CounterfactualTransitions requested but counterfactual stage CLI is unavailable." }
if ($Stage5UseCounterfactualTransitions -and (-not $CliSupport.stage5_transition_source)) { throw "Stage5 counterfactual transition source requested but Stage5 CLI does not expose --transition-source." }
if (($Stage5SelectionMetric -ne "actor_policy_q") -and (-not $CliSupport.stage5_selection_metric)) { throw "Stage5SelectionMetric requested but Stage5 CLI does not expose --selection-metric." }
if (($ActorExtractionMode -ne "awr") -and (-not $CliSupport.stage5_actor_extraction_mode)) { throw "ActorExtractionMode requested but Stage5 CLI does not expose --actor-extraction-mode." }
if (($ActorHeadArch -ne "linear") -and (-not $CliSupport.stage5_actor_head_arch)) { throw "ActorHeadArch requested but Stage5 CLI does not expose --actor-head-arch." }
if ($PreserveCurrentNonCurrentResidual -and (-not $CliSupport.stage6_preserve_residual)) { throw "PreserveCurrentNonCurrentResidual requested but Stage6 CLI does not expose --preserve-current-non-current-residual." }
if (($SimBusinessPlanMode -ne "default") -and (-not $CliSupport.stage6_sim_business_plan_mode)) { throw "SimBusinessPlanMode requested but Stage6 CLI does not expose --sim-business-plan-mode." }

$Stage3BatchValues = @(Parse-IntGrid $Stage3BatchGrid $Stage3BatchSize)
$Stage3EpochValues = @(Parse-IntGrid $Stage3EpochGrid $Stage3Epochs)
$Stage3SeedValues = @(Parse-IntGrid $Stage3SeedGrid $Stage3Seed)
$Stage3MaskValues = @(Parse-DoubleGrid $Stage3MaskingRatioGrid $Stage3MaskingRatio)
$Stage3LRValues = @(Parse-DoubleGrid $Stage3LearningRateGrid $Stage3LearningRate)
$Stage3WDValues = @(Parse-DoubleGrid $Stage3WeightDecayGrid $Stage3WeightDecay)

$PValues = @(Parse-IntGrid $PGrid $MagnitudeQuantile)
$Stage4EpochValues = @(Parse-IntGrid $Stage4EpochGrid $Stage4Epochs)
$Stage4BatchValues = @(Parse-IntGrid $Stage4BatchGrid $Stage4BatchSize)
$Stage4SeedValues = @(Parse-IntGrid $Stage4SeedGrid $Stage4Seed)
$ClassBalancedValues = @(Parse-BoolGrid $ClassBalancedGrid (-not $NoClassBalanced))
$ClassBetaValues = @(Parse-DoubleGrid $ClassBalanceBetaGrid $ClassBalanceBeta)
$ClassCapValues = @(Parse-DoubleGrid $ClassWeightCapGrid $ClassWeightCap)
$FamilyBalancedValues = @([bool]$FamilyBalanced)
$FamilyPowerValues = @([double]$FamilyBalancePower)
$FamilyCapValues = @([double]$FamilyWeightCap)
$CombinedCapValues = @([double]$CombinedWeightCap)

$Stage5EpochValues = @(Parse-IntGrid $Stage5EpochGrid $Stage5Epochs)
$Stage5BatchValues = @(Parse-IntGrid $Stage5BatchGrid $Stage5BatchSize)
$Stage5SeedValues = @(Parse-IntGrid $Stage5SeedGrid $Stage5Seed)
$GammaValues = @(Parse-DoubleGrid $GammaGrid $Gamma)
$TauValues = @(Parse-DoubleGrid $ExpectileTauGrid $ExpectileTau)
$BetaValues = @(Parse-DoubleGrid $BetaGrid $Beta)
$CqlValues = @(Parse-DoubleGrid $CqlAlphaGrid $CqlAlpha)
$Stage5LRValues = @(Parse-DoubleGrid $Stage5LearningRateGrid $Stage5LearningRate)
$Stage5WDValues = @(Parse-DoubleGrid $Stage5WeightDecayGrid $Stage5WeightDecay)
$DistillValues = @(Parse-DoubleGrid $DistillGrid $DistillLambda)
$DistillMarginValues = @(Parse-DoubleGrid $DistillMarginMinGrid $DistillMarginMin)
$DistillTempValues = @(Parse-DoubleGrid $DistillTemperatureGrid $DistillTemperature)
$CriticHeadArchValues = @(Parse-StringGrid $Stage5CriticHeadArchGrid $Stage5CriticHeadArch @("linear", "cross_attention", "cross_attention_film"))
$CrossAttnBlocksValues = @(Parse-IntGrid $CrossAttnBlocksGrid $CrossAttnBlocks)
$CrossAttnHeadsValues = @(Parse-IntGrid $CrossAttnHeadsGrid $CrossAttnHeads)
$CrossAttnDropoutValues = @(Parse-DoubleGrid $CrossAttnDropoutGrid $CrossAttnDropout)
$Stage6ExtrasValues = @(Parse-BoolGrid $Stage6IncludeExtrasGrid ([bool]$IncludeStage6Extras))
$Stage6AllowUnscoredValues = @(Parse-BoolGrid $Stage6AllowUnscoredGrid ([bool]$AllowStage6Unscored))

if ($Stage3Mode -ne "train") {
  $stage3GridCounts = @($Stage3BatchValues.Count,$Stage3EpochValues.Count,$Stage3SeedValues.Count,$Stage3MaskValues.Count,$Stage3LRValues.Count,$Stage3WDValues.Count)
  if (@($stage3GridCounts | Where-Object { $_ -gt 1 }).Count -gt 0) {
    throw "Stage3 grid values require -Stage3Mode train. With skip/restore, use one Stage3 lineage value and sweep downstream variables only."
  }
}

$UseAlignedSeeds = ([bool]$AlignStageSeeds) -or ($AlignedSeedGrid -and $AlignedSeedGrid.Trim().Length -gt 0)
$AlignedSeedValues = @()
if ($UseAlignedSeeds) {
  $alignedDefault = $Stage3Seed
  $AlignedSeedValues = @(Parse-IntGrid $AlignedSeedGrid $alignedDefault)
  if ($Stage3Mode -ne "train") { throw "-AlignStageSeeds/-AlignedSeedGrid require -Stage3Mode train because each cell owns its Stage3 lineage." }
  if (($Stage3SeedGrid -and $Stage3SeedGrid.Trim().Length -gt 0) -or ($Stage4SeedGrid -and $Stage4SeedGrid.Trim().Length -gt 0) -or ($Stage5SeedGrid -and $Stage5SeedGrid.Trim().Length -gt 0)) {
    throw "Use either -AlignedSeedGrid for all-stage seed cells OR individual Stage3/4/5 seed grids, not both. This prevents accidental Cartesian seed explosion."
  }
  $Stage3SeedValues = @($Stage3Seed)
  $Stage4SeedValues = @($Stage4Seed)
  $Stage5SeedValues = @($Stage5Seed)
}

$cells = @([ordered]@{})
$gridDimensions = @(
  @{n="s3_batch"; v=$Stage3BatchValues},
  @{n="s3_epochs"; v=$Stage3EpochValues},
  @{n="s3_mask"; v=$Stage3MaskValues},
  @{n="s3_lr"; v=$Stage3LRValues},
  @{n="s3_wd"; v=$Stage3WDValues},
  @{n="p"; v=$PValues},
  @{n="s4_epochs"; v=$Stage4EpochValues},
  @{n="s4_batch"; v=$Stage4BatchValues},
  @{n="s4_cb"; v=$ClassBalancedValues},
  @{n="s4_cb_beta"; v=$ClassBetaValues},
  @{n="s4_cap"; v=$ClassCapValues},
  @{n="s4_family_balanced"; v=$FamilyBalancedValues},
  @{n="s4_family_power"; v=$FamilyPowerValues},
  @{n="s4_family_cap"; v=$FamilyCapValues},
  @{n="s4_combined_cap"; v=$CombinedCapValues},
  @{n="s5_epochs"; v=$Stage5EpochValues},
  @{n="s5_batch"; v=$Stage5BatchValues},
  @{n="gamma"; v=$GammaValues},
  @{n="tau"; v=$TauValues},
  @{n="beta"; v=$BetaValues},
  @{n="cql"; v=$CqlValues},
  @{n="s5_lr"; v=$Stage5LRValues},
  @{n="s5_wd"; v=$Stage5WDValues},
  @{n="distill"; v=$DistillValues},
  @{n="distill_margin"; v=$DistillMarginValues},
  @{n="distill_temp"; v=$DistillTempValues},
  @{n="critic_head_arch"; v=$CriticHeadArchValues},
  @{n="cross_attn_blocks"; v=$CrossAttnBlocksValues},
  @{n="cross_attn_heads"; v=$CrossAttnHeadsValues},
  @{n="cross_attn_dropout"; v=$CrossAttnDropoutValues},
  @{n="s6_extras"; v=$Stage6ExtrasValues},
  @{n="s6_allow_unscored"; v=$Stage6AllowUnscoredValues}
)

if (-not $UseAlignedSeeds) {
  $gridDimensions += @(
    @{n="s3_seed"; v=$Stage3SeedValues},
    @{n="s4_seed"; v=$Stage4SeedValues},
    @{n="s5_seed"; v=$Stage5SeedValues}
  )
}

foreach ($d in $gridDimensions) {
  $cells = @(Expand-Grid -Cells $cells -Name $d["n"] -Values $d["v"])
}

if ($UseAlignedSeeds) {
  $seedAlignedCells = @()
  foreach ($cell in $cells) {
    foreach ($seed in $AlignedSeedValues) {
      $clone = [ordered]@{}
      foreach ($k in $cell.Keys) { $clone[$k] = $cell[$k] }
      $clone["s3_seed"] = $seed
      $clone["s4_seed"] = $seed
      $clone["s5_seed"] = $seed
      $clone["seed_lineage_mode"] = "aligned"
      $clone["seed_lineage_id"] = "S${seed}_${seed}_${seed}"
      $seedAlignedCells += $clone
    }
    if ($IncludeExactAnchorCell) {
      $clone = [ordered]@{}
      foreach ($k in $cell.Keys) { $clone[$k] = $cell[$k] }
      $clone["s3_seed"] = $AnchorStage3Seed
      $clone["s4_seed"] = $AnchorStage4Seed
      $clone["s5_seed"] = $AnchorStage5Seed
      $clone["seed_lineage_mode"] = "exact_anchor"
      $clone["seed_lineage_id"] = "S${AnchorStage3Seed}_${AnchorStage4Seed}_${AnchorStage5Seed}"
      $seedAlignedCells += $clone
    }
  }
  $cells = @($seedAlignedCells)
} else {
  foreach ($cell in $cells) {
    $cell["seed_lineage_mode"] = "cartesian"
    $cell["seed_lineage_id"] = "S$($cell["s3_seed"])_$($cell["s4_seed"])_$($cell["s5_seed"])"
  }
}

$totalCells = $cells.Count
if ($MaxCells -gt 0 -and $totalCells -gt $MaxCells) {
  throw "Grid expands to $totalCells cells, above -MaxCells $MaxCells. Narrow grids or increase -MaxCells."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[Unified RL Stage3/4/5/6 All-Grid Runner]"
Write-Host "============================================================"
Write-Host "RUNNER_VERSION = $RUNNER_VERSION"
Write-Host "Root           = $Root"
Write-Host "RunRoot        = $RunRoot"
Write-Host "Stage2         = $(if ($RunStage2) { 'run' } else { 'skip' })"
Write-Host "Stage2Rho      = $Stage2Rho"
Write-Host "Stage2Merton   = $Stage2MertonLambda"
Write-Host "Stage2FCFF     = $Stage2FcffLambda"
Write-Host "Stage2Liquidity= $Stage2LiquidityLambda"
Write-Host "Stage3Mode     = $Stage3Mode"
Write-Host "SeedLineage    = $(if ($UseAlignedSeeds) { 'aligned' } else { 'cartesian' })"
if ($UseAlignedSeeds) { Write-Host "AlignedSeeds   = $($AlignedSeedValues -join ',')" }
Write-Host "TotalCells     = $totalCells"
Write-Host "VerifierMode   = $VerifierMode"
Write-Host "ContinueOnFail = $ContinueOnCellFailure"
Write-Host "NoClean        = $NoClean"
Write-Host "SkipStage4     = $SkipStage4"
Write-Host "SkipStage5     = $SkipStage5"
Write-Host "NoSourceSnap   = $NoSourceSnapshot"
Write-Host "CLI support    = $(Join-Path $RunRoot 'CLI_SUPPORT_DETECTED.json')"
Write-Host "============================================================"

if ($Stage3Mode -eq "check") {
  Print-Stage3Metadata -Stage3Dir $Stage3Dir
  exit 0
}

Invoke-LoggedProcess -Name "Compile source" -Exe $Py -CommandArgs @("-m", "compileall", (Join-Path $Root "src"), "-q") -LogDir $RunLogDir | Out-Null

if ($RunStage2) {
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "[Stage2 reward refresh]"
  Write-Host "============================================================"
  Write-Host ("Rebuilding Stage2A/InputSplits/CandidateProjection with rho={0}, merton_lambda={1}, fcff_lambda={2}, liquidity_lambda={3}. Reward changes require downstream Stage4/5/6 regeneration; Stage3 encoder may be reused." -f $Stage2Rho, $Stage2MertonLambda, $Stage2FcffLambda, $Stage2LiquidityLambda)
  $Stage2RawAllDir = $RawAllDir
  if (-not $Stage2RawAllDir -or $Stage2RawAllDir.Trim().Length -eq 0) { $Stage2RawAllDir = Join-Path $Root "data\raw\raw_all" }
  $stage2aArgs = @(
    "-m", "credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline",
    "--project-root", $Root,
    "--raw-all-dir", $Stage2RawAllDir
  )
  Invoke-LoggedProcess -Name "Stage2A Raw Action Source Precompute" -Exe $Py -CommandArgs $stage2aArgs -LogDir $RunLogDir | Out-Null
  $stage2SplitArgs = @(
    "-m", "credit_recourse.rl.pipelines.final_stage2_input_splits.pipeline",
    "--project-root", $Root,
    "--seed", "$Stage4Seed"
  )
  if ($JoinCashFlowSubstrate) {
    $stage2SplitArgs += @("--join-cash-flow-substrate", "--cash-flow-encoder-mode", $CashFlowEncoderMode)
    if ($CashFlowPanel -and $CashFlowPanel.Trim().Length -gt 0) {
      $stage2SplitArgs += @("--cash-flow-panel", $CashFlowPanel)
    }
  }
  Invoke-LoggedProcess -Name "Stage2 Input Splits" -Exe $Py -CommandArgs $stage2SplitArgs -LogDir $RunLogDir | Out-Null
  $stage2Args = @(
    "-m", "credit_recourse.rl.pipelines.final_stage2_candidate_projection.pipeline",
    "--project-root", $Root,
    "--sector-phi"
  )
  $stage2Args = Add-OptionalArg -ArgsIn $stage2Args -HelpText $Stage2Help.text -Option "--rho" -Value $Stage2Rho -Label "Stage2"
  $stage2Args = Add-OptionalArg -ArgsIn $stage2Args -HelpText $Stage2Help.text -Option "--merton-lambda" -Value $Stage2MertonLambda -Label "Stage2"
  $stage2Args = Add-OptionalArg -ArgsIn $stage2Args -HelpText $Stage2Help.text -Option "--fcff-lambda" -Value $Stage2FcffLambda -Label "Stage2"
  $stage2Args = Add-OptionalArg -ArgsIn $stage2Args -HelpText $Stage2Help.text -Option "--liquidity-lambda" -Value $Stage2LiquidityLambda -Label "Stage2"
  Invoke-LoggedProcess -Name "Stage2 Candidate Projection rho=$Stage2Rho merton=$Stage2MertonLambda fcff=$Stage2FcffLambda liquidity=$Stage2LiquidityLambda" -Exe $Py -CommandArgs $stage2Args -LogDir $RunLogDir | Out-Null
  $stage2MetaPath = Join-Path $FF "stage2_candidate_projection\metadata.json"
  Assert-Exists $stage2MetaPath "Stage2 candidate projection metadata.json"
  $stage2Meta = Read-JsonFile $stage2MetaPath
  if (Has-Prop $stage2Meta "rho_main") { Check-NumericMetadata $stage2Meta "rho_main" $Stage2Rho "Stage2" }
  elseif (Has-Prop $stage2Meta "rho_used") { Check-NumericMetadata $stage2Meta "rho_used" $Stage2Rho "Stage2" }
  else { throw "Stage2 metadata missing rho_main/rho_used after RunStage2 refresh." }
  if (Has-Prop $stage2Meta "aux_reward_stats") {
    $auxMeta = $stage2Meta.aux_reward_stats
    if (Has-Prop $auxMeta "lambda_merton") { Check-NumericMetadata $auxMeta "lambda_merton" $Stage2MertonLambda "Stage2" }
    elseif ([math]::Abs($Stage2MertonLambda) -gt 1e-12) { throw "Stage2 metadata aux_reward_stats missing lambda_merton after Merton reward refresh." }
    if (Has-Prop $auxMeta "lambda_fcff") { Check-NumericMetadata $auxMeta "lambda_fcff" $Stage2FcffLambda "Stage2" }
    elseif ([math]::Abs($Stage2FcffLambda) -gt 1e-12) { throw "Stage2 metadata aux_reward_stats missing lambda_fcff after FCFF reward refresh." }
    if (Has-Prop $auxMeta "lambda_liquidity") { Check-NumericMetadata $auxMeta "lambda_liquidity" $Stage2LiquidityLambda "Stage2" }
    elseif ([math]::Abs($Stage2LiquidityLambda) -gt 1e-12) { throw "Stage2 metadata aux_reward_stats missing lambda_liquidity after liquidity reward refresh." }
  } elseif ([math]::Abs($Stage2MertonLambda) -gt 1e-12 -or [math]::Abs($Stage2FcffLambda) -gt 1e-12 -or [math]::Abs($Stage2LiquidityLambda) -gt 1e-12) {
    throw "Stage2 metadata missing aux_reward_stats after nonzero Merton/FCFF/liquidity reward refresh."
  }
}


if ($Stage3Mode -eq "restore") {
  Write-Host "[Stage3 restore] $Stage3SourceDir -> $Stage3Dir"
  if (-not $NoClean) { Remove-DirIfExists $Stage3Dir }
  Restore-Stage3FromSource -Src $Stage3SourceDir -Dst $Stage3Dir
}

if ($Stage3Mode -in @("skip","restore","archive_only")) {
  Assert-Exists (Join-Path $Stage3Dir "ssl_encoder.pt") "Stage3 ssl_encoder.pt"
  Assert-Exists (Join-Path $Stage3Dir "metadata.json") "Stage3 metadata.json"
  $stage3Meta = Read-JsonFile (Join-Path $Stage3Dir "metadata.json")
  Check-NumericMetadata $stage3Meta "epochs" $Stage3EpochValues[0] "Stage3(reused)" -AllowMissing:$AllowMissingStage3Metadata -AllowMismatch:$AllowStage3MetadataMismatch
  if (Has-Prop $stage3Meta "batch_size") {
    Check-NumericMetadata $stage3Meta "batch_size" $Stage3BatchValues[0] "Stage3(reused)" -AllowMissing:$AllowMissingStage3Metadata -AllowMismatch:$AllowStage3MetadataMismatch
  } elseif (Has-Prop $stage3Meta "train_batch_size") {
    Check-NumericMetadata $stage3Meta "train_batch_size" $Stage3BatchValues[0] "Stage3(reused)" -AllowMissing:$AllowMissingStage3Metadata -AllowMismatch:$AllowStage3MetadataMismatch
  } elseif (-not $AllowMissingStage3Metadata) {
    throw "Stage3(reused) missing batch_size/train_batch_size metadata"
  }
  Print-Stage3Metadata -Stage3Dir $Stage3Dir
}

$summaryRows = @()
$cellIndex = 0

foreach ($cfg in $cells) {
  $cellIndex += 1

  $gammaToken = Normalize-Token $cfg["gamma"]
  $tauToken = Normalize-Token $cfg["tau"]
  $betaToken = Normalize-Token $cfg["beta"]
  $lamToken = Normalize-Token $cfg["distill"]
  $lr5Token = Normalize-Token $cfg["s5_lr"]
  $s3ModeToken = $Stage3Mode.ToUpperInvariant()
  $cbToken = if ([bool]$cfg["s4_cb"]) { "CB" } else { "NO_CB" }
  $fbToken = if ([bool]$cfg["s4_family_balanced"]) { "FB" } else { "NO_FB" }
  $criticToken = ([string]$cfg["critic_head_arch"]).ToUpperInvariant()
  $seedLineage = if ($cfg.Contains("seed_lineage_id")) { $cfg["seed_lineage_id"] } else { "S$($cfg["s3_seed"])_$($cfg["s4_seed"])_$($cfg["s5_seed"])" }
  $CellId = "B$($cfg["s3_batch"])_E$($cfg["s3_epochs"])_$($s3ModeToken)_$($seedLineage)_P$($cfg["p"])_$($cbToken)_$($fbToken)_$($criticToken)_DISTILL$($lamToken)_G$($gammaToken)_TAU$($tauToken)_BETA$($betaToken)_LR5$($lr5Token)"
  $CellFolder = "cell_{0:000}_P{1}_L{2}_G{3}_T{4}_B{5}_{6}" -f $cellIndex, $cfg["p"], $lamToken, $gammaToken, $tauToken, $betaToken, (Get-ShortHash -Text $CellId -Length 8)
  $ArchiveRoot = Join-Path $RunRoot $CellFolder
  $CellLogDir = Join-Path $ArchiveRoot "run_logs"
  New-Item -ItemType Directory -Force -Path $CellLogDir | Out-Null

  $cellConfig = [ordered]@{
    runner_version = $RUNNER_VERSION
    stage2 = [ordered]@{
      mode = $(if ($RunStage2) { "run_before_stage3" } else { "skip" })
      rho = $Stage2Rho
      merton_lambda = $Stage2MertonLambda
      fcff_lambda = $Stage2FcffLambda
      liquidity_lambda = $Stage2LiquidityLambda
      join_cash_flow_substrate = [bool]$JoinCashFlowSubstrate
      cash_flow_encoder_mode = $CashFlowEncoderMode
      cash_flow_panel = $CashFlowPanel
      raw_all_dir = $(if ($RawAllDir -and $RawAllDir.Trim().Length -gt 0) { $RawAllDir } else { Join-Path $Root "data\raw\raw_all" })
      input_split_seed = $null
      blast_radius = "Stage2 reward changes require Stage4/5/6 regeneration; Stage3 encoder is reward-independent and may be reused."
    }
    stage3 = [ordered]@{
      mode = $Stage3Mode
      batch_size = $cfg["s3_batch"]
      epochs = $cfg["s3_epochs"]
      seed = $cfg["s3_seed"]
      seed_lineage_mode = $cfg["seed_lineage_mode"]
      seed_lineage_id = $cfg["seed_lineage_id"]
      masking_ratio = $cfg["s3_mask"]
      learning_rate = $cfg["s3_lr"]
      weight_decay = $cfg["s3_wd"]
      source_dir = $Stage3SourceDir
    }
    stage4 = [ordered]@{
      mode = $(if ($SkipStage4) { "skip_reuse_active" } else { "train" })
      magnitude_quantile = $cfg["p"]
      epochs = $cfg["s4_epochs"]
      batch_size = $cfg["s4_batch"]
      seed = $cfg["s4_seed"]
      class_balanced = $cfg["s4_cb"]
      class_balance_beta = $cfg["s4_cb_beta"]
      class_weight_cap = $cfg["s4_cap"]
      family_balanced = $cfg["s4_family_balanced"]
      family_balance_power = $cfg["s4_family_power"]
      family_weight_cap = $cfg["s4_family_cap"]
      combined_weight_cap = $cfg["s4_combined_cap"]
    }
    stage5 = [ordered]@{
      mode = $(if ($SkipStage5) { "skip_reuse_active" } else { "train" })
      epochs = $cfg["s5_epochs"]
      batch_size = $cfg["s5_batch"]
      seed = $cfg["s5_seed"]
      gamma = $cfg["gamma"]
      expectile_tau = $cfg["tau"]
      beta = $cfg["beta"]
      cql_alpha = $cfg["cql"]
      learning_rate = $cfg["s5_lr"]
      weight_decay = $cfg["s5_wd"]
      distill_lambda = $cfg["distill"]
      distill_margin_min = $cfg["distill_margin"]
      distill_temperature = $cfg["distill_temp"]
      critic_head_arch = $cfg["critic_head_arch"]
      cross_attn_blocks = $cfg["cross_attn_blocks"]
      cross_attn_heads = $cfg["cross_attn_heads"]
      cross_attn_dropout = $cfg["cross_attn_dropout"]
    }
    stage6 = [ordered]@{
      include_extras = $cfg["s6_extras"]
      allow_unscored = $cfg["s6_allow_unscored"]
    }
    verifier = [ordered]@{ mode = $VerifierMode }
    cli_support = $CliSupport
  }

  $status = "RUNNING"
  $failure = ""
  try {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[CELL $cellIndex / $totalCells] $CellId"
    Write-Host "============================================================"
    Write-Host "ArchiveRoot = $ArchiveRoot"

    if ($Stage3Mode -eq "archive_only") {
      Assert-Exists (Join-Path $Stage4Dir "metadata.json") "active Stage4 metadata.json"
      Assert-Exists (Join-Path $Stage5Dir "metadata.json") "active Stage5 metadata.json"
      Assert-Exists (Join-Path $Stage6MultiDir "final_policy_summary.csv") "active Stage6 final_policy_summary.csv"
      $status = "ARCHIVE_ONLY"
    } else {
      if ($ReuseCounterfactualTransitionsForStage5 -and -not $CounterfactualTransitions) {
        $cfOut = Join-Path $FF ("stage2_candidate_projection\phase3_iql_counterfactual_candidate__P{0}.parquet" -f $cfg["p"])
        Assert-Exists $cfOut "reused Stage2 counterfactual transition parquet"
        Write-Host "[Stage2 counterfactual reuse] Preserving active artifact: $cfOut"
      }

      if ($CounterfactualTransitions) {
        $cfArgs = @(
          "-m", "credit_recourse.rl.pipelines.final_stage2_counterfactual_transitions.pipeline",
          "--project-root", $Root,
          "--magnitude-quantile", "$($cfg["p"])",
          "--reward-mode", $CounterfactualRewardMode,
          "--done-mode", $CounterfactualDoneMode,
          "--fidelity-gate", $CounterfactualFidelityGate,
          "--max-rel-err-assets", "$CounterfactualMaxRelErrAssets",
          "--sim-business-plan-mode", $SimBusinessPlanMode
        )
        if ($PreserveCurrentNonCurrentResidual) { $cfArgs += @("--preserve-current-non-current-residual") }
        Invoke-LoggedProcess -Name "Stage2 Counterfactual Transitions P$($cfg["p"]) $CellId" -Exe $Py -CommandArgs $cfArgs -LogDir $CellLogDir | Out-Null
        $cfOut = Join-Path $FF ("stage2_candidate_projection\phase3_iql_counterfactual_candidate__P{0}.parquet" -f $cfg["p"])
        Assert-Exists $cfOut "Stage2 counterfactual transition parquet"
      }

      if ($Stage3Mode -eq "train") {
        if (-not $NoClean) { Remove-DirIfExists $Stage3Dir }
        $stage3Args = @(
          "-m", "credit_recourse.rl.pipelines.final_stage3_acd_ssl.pipeline",
          "--project-root", $Root,
          "--train-mode", "final_refit",
          "--batch-size", "$($cfg["s3_batch"])",
          "--epochs", "$($cfg["s3_epochs"])",
          "--seed", "$($cfg["s3_seed"])",
          "--masking-ratio", "$($cfg["s3_mask"])"
        )
        $stage3Args = Add-OptionalArg -ArgsIn $stage3Args -HelpText $Stage3Help.text -Option "--learning-rate" -Value $cfg["s3_lr"] -Label "Stage3"
        $stage3Args = Add-OptionalArg -ArgsIn $stage3Args -HelpText $Stage3Help.text -Option "--weight-decay" -Value $cfg["s3_wd"] -Label "Stage3"
        Invoke-LoggedProcess -Name "Stage3 ACD SSL encoder train $CellId" -Exe $Py -CommandArgs $stage3Args -LogDir $CellLogDir | Out-Null
        Assert-Exists (Join-Path $Stage3Dir "ssl_encoder.pt") "Stage3 ssl_encoder.pt"
        Assert-Exists (Join-Path $Stage3Dir "metadata.json") "Stage3 metadata.json"
        $stage3Meta = Read-JsonFile (Join-Path $Stage3Dir "metadata.json")
        Check-NumericMetadata $stage3Meta "epochs" $cfg["s3_epochs"] "Stage3" -AllowMismatch:$AllowStage3MetadataMismatch
        if (Has-Prop $stage3Meta "batch_size") {
          Check-NumericMetadata $stage3Meta "batch_size" $cfg["s3_batch"] "Stage3" -AllowMismatch:$AllowStage3MetadataMismatch
        }
        if ((Test-CliOption $Stage3Help.text "--learning-rate") -and (Has-Prop $stage3Meta "learning_rate")) {
          Check-NumericMetadata $stage3Meta "learning_rate" $cfg["s3_lr"] "Stage3" -AllowMismatch:$AllowStage3MetadataMismatch
        }
        if ((Test-CliOption $Stage3Help.text "--weight-decay") -and (Has-Prop $stage3Meta "weight_decay")) {
          Check-NumericMetadata $stage3Meta "weight_decay" $cfg["s3_wd"] "Stage3" -AllowMismatch:$AllowStage3MetadataMismatch
        }
      }

      if (-not $NoClean) {
        if (-not $SkipStage4) {
          Remove-DirIfExists $Stage4Dir
        } else {
          Write-Host "[Stage4 skip] Preserving active Stage4 artifacts: $Stage4Dir"
        }
        if (-not $SkipStage5) {
          Remove-DirIfExists $Stage5Dir
        } else {
          Write-Host "[Stage5 skip] Preserving active Stage5 artifacts: $Stage5Dir"
        }
        Remove-DirIfExists $Stage6SelectorDir
        Remove-DirIfExists $Stage6MultiDir
      } else {
        Write-Warning "-NoClean set: active Stage4/5/6 artifacts may be reused."
      }

      if ($SkipStage4) {
        Assert-Stage4Reusable -Stage4Dir $Stage4Dir -ExpectedEpochs ([int]$cfg["s4_epochs"]) -ExpectedBatchSize ([int]$cfg["s4_batch"]) -ExpectedSeed ([int]$cfg["s4_seed"]) -ExpectedMagnitudeQuantile ([int]$cfg["p"]) -ExpectedClassBalanced ([bool]$cfg["s4_cb"]) -ExpectedClassBalanceBeta ([double]$cfg["s4_cb_beta"]) -ExpectedClassWeightCap ([double]$cfg["s4_cap"]) -ExpectedFamilyBalanced ([bool]$cfg["s4_family_balanced"]) -ExpectedFamilyBalancePower ([double]$cfg["s4_family_power"]) -ExpectedFamilyWeightCap ([double]$cfg["s4_family_cap"]) -ExpectedCombinedWeightCap ([double]$cfg["s4_combined_cap"])
      } else {
        $stage4Args = @(
          "-m", "credit_recourse.rl.pipelines.final_stage4_candidate_bc.pipeline",
          "--project-root", $Root,
          "--train-mode", "final_refit",
          "--epochs", "$($cfg["s4_epochs"])",
          "--seeds", "$($cfg["s4_seed"])",
          "--batch-size", "$($cfg["s4_batch"])",
          "--magnitude-quantile", "$($cfg["p"])"
        )
        if ([bool]$cfg["s4_cb"]) {
          if (Test-CliOption $Stage4Help.text "--class-balanced") {
            $stage4Args += @("--class-balanced", "--class-balance-beta", "$($cfg["s4_cb_beta"])", "--class-weight-cap", "$($cfg["s4_cap"])")
          } else {
            Write-Warning "Stage4 CLI does not support class-balanced flags; requested class-balanced=True ignored."
          }
        }
        if ([bool]$cfg["s4_family_balanced"]) {
          if ($CliSupport.stage4_family_balanced -and $CliSupport.stage4_family_balance_power -and $CliSupport.stage4_family_weight_cap -and $CliSupport.stage4_combined_weight_cap) {
            $stage4Args += @("--family-balanced", "--family-balance-power", "$($cfg["s4_family_power"])", "--family-weight-cap", "$($cfg["s4_family_cap"])", "--combined-weight-cap", "$($cfg["s4_combined_cap"])")
          } else {
            throw "Stage4 CLI does not expose family-balanced flags but -FamilyBalanced was requested."
          }
        }
        Invoke-LoggedProcess -Name "Stage4 Candidate BC $CellId" -Exe $Py -CommandArgs $stage4Args -LogDir $CellLogDir | Out-Null
        Assert-Exists (Join-Path $Stage4Dir "metadata.json") "Stage4 metadata.json"
      }

      $stage5Args = @(
        "-m", "credit_recourse.rl.pipelines.final_stage5_candidate_iql.pipeline",
        "--project-root", $Root,
        "--train-mode", "final_refit",
        "--epochs", "$($cfg["s5_epochs"])",
        "--seeds", "$($cfg["s5_seed"])",
        "--batch-size", "$($cfg["s5_batch"])",
        "--magnitude-quantile", "$($cfg["p"])",
        "--gamma", "$($cfg["gamma"])",
        "--expectile-tau", "$($cfg["tau"])",
        "--beta", "$($cfg["beta"])",
        "--cql-alpha", "$($cfg["cql"])"
      )
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--learning-rate" -Value $cfg["s5_lr"] -Label "Stage5"
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--weight-decay" -Value $cfg["s5_wd"] -Label "Stage5"
      if (Test-CliOption $Stage5Help.text "--actor-distill-lambda") {
        $stage5Args += @(
          "--actor-distill-mode", "ce",
          "--actor-distill-lambda", "$($cfg["distill"])",
          "--actor-distill-margin-min", "$($cfg["distill_margin"])",
          "--actor-distill-temperature", "$($cfg["distill_temp"])"
        )
      } else {
        Write-Warning "Stage5 CLI does not support actor distillation flags; distill settings recorded in manifest only."
      }
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--critic-head-arch" -Value $cfg["critic_head_arch"] -Label "Stage5"
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--cross-attn-blocks" -Value $cfg["cross_attn_blocks"] -Label "Stage5"
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--cross-attn-heads" -Value $cfg["cross_attn_heads"] -Label "Stage5"
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--cross-attn-dropout" -Value $cfg["cross_attn_dropout"] -Label "Stage5"
      if ($Stage5UseCounterfactualTransitions) { $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--transition-source" -Value "counterfactual" -Label "Stage5" }
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--selection-metric" -Value $Stage5SelectionMetric -Label "Stage5"
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--actor-extraction-mode" -Value $ActorExtractionMode -Label "Stage5"
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--actor-finetune-steps" -Value $ActorFinetuneSteps -Label "Stage5"
      $stage5Args = Add-OptionalArg -ArgsIn $stage5Args -HelpText $Stage5Help.text -Option "--actor-head-arch" -Value $ActorHeadArch -Label "Stage5"
      if ($SkipStage5) {
        Write-Host "[Stage5 skip] Reusing active Stage5 artifacts: $Stage5Dir"
        Assert-NonEmptyDir $Stage5Dir "Stage5 artifact directory for -SkipStage5"
        Assert-Stage5EffectiveConfig -Stage5Dir $Stage5Dir -ExpectedLearningRate ([double]$cfg["s5_lr"]) -ExpectedWeightDecay ([double]$cfg["s5_wd"]) -ExpectedCriticHeadArch ([string]$cfg["critic_head_arch"]) -ExpectedCrossAttnBlocks ([int]$cfg["cross_attn_blocks"]) -ExpectedCrossAttnHeads ([int]$cfg["cross_attn_heads"]) -ExpectedCrossAttnDropout ([double]$cfg["cross_attn_dropout"])
      } else {
        Invoke-LoggedProcess -Name "Stage5 Candidate IQL $CellId" -Exe $Py -CommandArgs $stage5Args -LogDir $CellLogDir | Out-Null
        Assert-Exists (Join-Path $Stage5Dir "metadata.json") "Stage5 metadata.json"
        Assert-Stage5EffectiveConfig -Stage5Dir $Stage5Dir -ExpectedLearningRate ([double]$cfg["s5_lr"]) -ExpectedWeightDecay ([double]$cfg["s5_wd"]) -ExpectedCriticHeadArch ([string]$cfg["critic_head_arch"]) -ExpectedCrossAttnBlocks ([int]$cfg["cross_attn_blocks"]) -ExpectedCrossAttnHeads ([int]$cfg["cross_attn_heads"]) -ExpectedCrossAttnDropout ([double]$cfg["cross_attn_dropout"])
      }

      $stage6SelectorArgs = @(
        "-m", "credit_recourse.eval.final_stage6_candidate_selector_eval.pipeline",
        "--project-root", $Root
      )
      $stage6SelectorArgs = Add-OptionalSwitch -ArgsIn $stage6SelectorArgs -HelpText $Stage6SelectorHelp.text -Option "--include-stage6-extras" -Enabled ([bool]$cfg["s6_extras"]) -Label "Stage6 selector"
      Invoke-LoggedProcess -Name "Stage6 Candidate Selector $CellId" -Exe $Py -CommandArgs $stage6SelectorArgs -LogDir $CellLogDir | Out-Null
      Assert-NonEmptyDir $Stage6SelectorDir "Stage6 candidate selector eval"

      $stage6MultiArgs = @(
        "-m", "credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline",
        "--project-root", $Root
      )
      $stage6MultiArgs = Add-OptionalSwitch -ArgsIn $stage6MultiArgs -HelpText $Stage6MultiHelp.text -Option "--allow-unscored" -Enabled ([bool]$cfg["s6_allow_unscored"]) -Label "Stage6 multi-oracle"
      $stage6MultiArgs = Add-OptionalSwitch -ArgsIn $stage6MultiArgs -HelpText $Stage6MultiHelp.text -Option "--preserve-current-non-current-residual" -Enabled ([bool]$PreserveCurrentNonCurrentResidual) -Label "Stage6 multi-oracle"
      $stage6MultiArgs = Add-OptionalArg -ArgsIn $stage6MultiArgs -HelpText $Stage6MultiHelp.text -Option "--sim-business-plan-mode" -Value $SimBusinessPlanMode -Label "Stage6 multi-oracle"
      $stage6MultiArgs = Add-OptionalSwitch -ArgsIn $stage6MultiArgs -HelpText $Stage6MultiHelp.text -Option "--deploy-qargmax-as-policy" -Enabled ([bool]$DeployQArgmaxAsPolicy) -Label "Stage6 multi-oracle"
      Invoke-LoggedProcess -Name "Stage6 Multi Oracle $CellId" -Exe $Py -CommandArgs $stage6MultiArgs -LogDir $CellLogDir | Out-Null
      Assert-Exists (Join-Path $Stage6MultiDir "final_policy_summary.csv") "Stage6 final_policy_summary.csv"

      if (-not $SkipStage6Inference) {
        Invoke-LoggedProcess -Name "Stage6 Statistical Inference $CellId" -Exe $Py -CommandArgs @(
          "-m", "credit_recourse.eval.final_stage6_statistical_inference",
          "--project-root", $Root
        ) -LogDir $CellLogDir | Out-Null
      }

      Run-Verifier -Root $Root -Py $Py -FF $FF -VerifierMode $VerifierMode -ArchiveRoot $ArchiveRoot -LogDir $CellLogDir | Out-Null
      $status = if ($VerifierMode -eq "warn") { "COMPLETED_VERIFIER_WARN_ALLOWED" } else { "COMPLETED" }
    }
  } catch {
    $status = "FAILED_PARTIAL"
    $failure = [string]$_.Exception.Message
    Write-Warning "CELL FAILED: $CellId"
    Write-Warning $failure
    if (-not $ContinueOnCellFailure) {
      throw
    }
  } finally {
    Archive-CurrentCell -ArchiveRoot $ArchiveRoot -CellId $CellId -Status $status -FailureMessage $failure -CellConfig $cellConfig -Root $Root -FF $FF -Stage3Dir $Stage3Dir -Stage4Dir $Stage4Dir -Stage5Dir $Stage5Dir -Stage6SelectorDir $Stage6SelectorDir -Stage6MultiDir $Stage6MultiDir -NoSourceSnapshot:$NoSourceSnapshot -ZipArchive:$ZipArchive
    $summaryRows += [pscustomobject]@{
      cell_index = $cellIndex
      cell_id = $CellId
      status = $status
      failure_message = $failure
      archive_root = $ArchiveRoot
      stage3_mode = $Stage3Mode
      stage3_batch_size = $cfg["s3_batch"]
      stage3_epochs = $cfg["s3_epochs"]
      stage3_seed = $cfg["s3_seed"]
      seed_lineage_mode = $cfg["seed_lineage_mode"]
      seed_lineage_id = $cfg["seed_lineage_id"]
      stage3_masking_ratio = $cfg["s3_mask"]
      stage3_learning_rate = $cfg["s3_lr"]
      stage3_weight_decay = $cfg["s3_wd"]
      magnitude_quantile = $cfg["p"]
      stage4_mode = $(if ($SkipStage4) { "skip_reuse_active" } else { "train" })
      stage4_epochs = $cfg["s4_epochs"]
      stage4_batch_size = $cfg["s4_batch"]
      stage4_seed = $cfg["s4_seed"]
      stage4_class_balanced = $cfg["s4_cb"]
      stage4_class_balance_beta = $cfg["s4_cb_beta"]
      stage4_class_weight_cap = $cfg["s4_cap"]
      stage4_family_balanced = $cfg["s4_family_balanced"]
      stage4_family_balance_power = $cfg["s4_family_power"]
      stage4_family_weight_cap = $cfg["s4_family_cap"]
      stage4_combined_weight_cap = $cfg["s4_combined_cap"]
      stage5_mode = $(if ($SkipStage5) { "skip_reuse_active" } else { "train" })
      stage5_epochs = $cfg["s5_epochs"]
      stage5_batch_size = $cfg["s5_batch"]
      stage5_seed = $cfg["s5_seed"]
      gamma = $cfg["gamma"]
      expectile_tau = $cfg["tau"]
      beta = $cfg["beta"]
      cql_alpha = $cfg["cql"]
      stage5_learning_rate = $cfg["s5_lr"]
      stage5_weight_decay = $cfg["s5_wd"]
      distill_lambda = $cfg["distill"]
      distill_margin_min = $cfg["distill_margin"]
      distill_temperature = $cfg["distill_temp"]
      critic_head_arch = $cfg["critic_head_arch"]
      cross_attn_blocks = $cfg["cross_attn_blocks"]
      cross_attn_heads = $cfg["cross_attn_heads"]
      cross_attn_dropout = $cfg["cross_attn_dropout"]
      stage6_include_extras = $cfg["s6_extras"]
      stage6_allow_unscored = $cfg["s6_allow_unscored"]
    }
    $summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $RunRoot "UNIFIED_RUN_SUMMARY.csv")
  }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[DONE] Unified all-grid runner finished"
Write-Host "============================================================"
Write-Host "RunRoot = $RunRoot"
Write-Host "Summary = $(Join-Path $RunRoot 'UNIFIED_RUN_SUMMARY.csv')"
