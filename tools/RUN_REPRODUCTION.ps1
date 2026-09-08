[CmdletBinding()]
param(
    [ValidateSet('FrozenReplay','OracleClean','OracleRLClean','OracleRLLLMClean','FullClean')]
    [string]$Mode,
    [string]$RunId,
    [string]$ProjectRoot,
    [string]$PythonExe,
    [string]$PythonSitePackagesRoot,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Select-Mode {
    Write-Host ''
    Write-Host '논문 재현 실행'
    Write-Host '  1. FrozenReplay       보존 산출물 -> V10 분석'
    Write-Host '  2. OracleClean        raw -> Oracle'
    Write-Host '  3. OracleRLClean      raw -> Oracle -> RL'
    Write-Host '  4. OracleRLLLMClean   raw -> Oracle -> RL -> live LLM'
    Write-Host '  5. FullClean          위 전체 -> V10 분석'
    switch (Read-Host '실행할 번호 [1-5]') {
        '1' { return 'FrozenReplay' }
        '2' { return 'OracleClean' }
        '3' { return 'OracleRLClean' }
        '4' { return 'OracleRLLLMClean' }
        '5' { return 'FullClean' }
        default { throw '1에서 5 사이의 번호를 선택하세요.' }
    }
}

function Import-LocalEnvironment([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed -notmatch '=') { continue }
        $parts = $trimmed.Split('=', 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name -match '^[A-Za-z_][A-Za-z0-9_]*$' -and -not [Environment]::GetEnvironmentVariable($name, 'Process')) {
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}

function Test-PythonExecutable([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate)) { return $null }
    $resolved = $null
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        $resolved = (Resolve-Path -LiteralPath $Candidate).Path
    } else {
        $command = Get-Command $Candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command -and $command.Source -and (Test-Path -LiteralPath $command.Source -PathType Leaf)) {
            $resolved = (Resolve-Path -LiteralPath $command.Source).Path
        }
    }
    if (-not $resolved) { return $null }
    try {
        $probeLines = @(& $resolved -c "import sys; print(sys.executable)" 2>$null)
        $probeExit = $LASTEXITCODE
        $probe = ($probeLines -join [Environment]::NewLine).Trim()
        if ($probeExit -eq 0 -and -not [string]::IsNullOrWhiteSpace($probe)) { return $resolved }
    } catch { }
    return $null
}

function Resolve-Python([string]$Explicit, [string]$Root) {
    foreach ($candidate in @($Explicit, $env:REPRO_PYTHON, (Join-Path $Root '.venv\Scripts\python.exe'), 'python')) {
        $resolved = Test-PythonExecutable $candidate
        if ($resolved) { return $resolved }
    }
    throw '실행 가능한 Python을 찾지 못했습니다. 저장소 .venv를 만들거나 -PythonExe로 실제 python.exe를 지정하세요.'
}

function Invoke-Checked([string]$Label, [string[]]$Command) {
    Write-Host ''
    Write-Host "==== $Label ====" -ForegroundColor Cyan
    & $Command[0] @($Command[1..($Command.Count - 1)])
    if ($LASTEXITCODE -ne 0) { throw "$Label 실행 실패 (exit=$LASTEXITCODE)" }
}

function Test-JsonReadable([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try { Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json | Out-Null; return $true } catch { return $false }
}

function Test-ParquetReadable([string]$Path, [string]$Python) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    & $Python -c "import pyarrow.parquet as p,sys; p.ParquetFile(sys.argv[1]).schema" $Path 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Test-OracleReady([string]$Root, [string]$Python) {
    $final = Join-Path $Root 'data\final_freeze'
    return (
        (Test-ParquetReadable (Join-Path $final 'stage0_oracle_foundation\canonical_panel\stage0_canonical_panel.parquet') $Python) -and
        (Test-ParquetReadable (Join-Path $final 'stage0_oracle_foundation\canonical_panel\statement_items_panel.parquet') $Python) -and
        (Test-JsonReadable (Join-Path $final 'stage0_oracle_foundation\stage0_manifest.json')) -and
        (Test-ParquetReadable (Join-Path $final 'stage1_oracle_inputs\alpha_vanilla_input_candidate.parquet') $Python) -and
        (Test-JsonReadable (Join-Path $final 'stage1_oracle_inputs\stage00_01_rating_statement_integration\stage00_01_metadata.json')) -and
        (Test-ParquetReadable (Join-Path $final 'stage1_oracle_backends\alpha\oracle_firm_year_output_alpha.parquet') $Python) -and
        (Test-ParquetReadable (Join-Path $final 'stage1_oracle_backends\beta\benchmark_firm_year_output_beta.parquet') $Python) -and
        (Test-ParquetReadable (Join-Path $final 'stage1_oracle_backends\gamma\benchmark_firm_year_output_gamma.parquet') $Python) -and
        (Test-Path -LiteralPath (Join-Path $final 'configs\oracle_backend_registry.yaml') -PathType Leaf)
    )
}

function Test-RLReady([string]$Root, [string]$Python) {
    $final = Join-Path $Root 'data\final_freeze'
    return (
        (Test-ParquetReadable (Join-Path $final 'stage2_candidate_projection\phase_eval_candidate.parquet') $Python) -and
        (Test-ParquetReadable (Join-Path $final 'stage2_candidate_projection\phase3_iql_candidate__P50.parquet') $Python) -and
        (Test-JsonReadable (Join-Path $final 'stage2_candidate_projection\counterfactual_transitions_metadata__P50.json')) -and
        (Test-Path -LiteralPath (Join-Path $final 'stage3_acd_ssl\stage3_encoder_avs256_final_refit_fulltrain.pt') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $final 'stage4_candidate_bc\candidate_bc_policy.pt') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $final 'stage5_candidate_iql\candidate_iql_policy.pt') -PathType Leaf) -and
        (Test-ParquetReadable (Join-Path $final 'stage6_candidate_selector_eval\policy_actions.parquet') $Python) -and
        (Test-ParquetReadable (Join-Path $final 'stage6_candidate_selector_eval\multi_oracle_policy_eval.parquet') $Python) -and
        (Test-ParquetReadable (Join-Path $final 'stage6_multi_oracle_eval\multi_oracle_policy_eval.parquet') $Python)
    )
}

function Test-LLMReady([string]$Root, [string]$Python) {
    $runs = Join-Path $Root 'data\final_freeze\llm_runs'
    if (-not (Test-Path -LiteralPath $runs -PathType Container)) { return $false }
    $complete = 0
    foreach ($dir in Get-ChildItem -LiteralPath $runs -Directory) {
        if (
            (Test-ParquetReadable (Join-Path $dir.FullName 'stage7_llm_action_generation\llm_stage7_action_table.parquet') $Python) -and
            (Test-ParquetReadable (Join-Path $dir.FullName 'stage8_llm_multi_oracle_eval\llm_stage8_multi_oracle_scores.parquet') $Python) -and
            (Test-ParquetReadable (Join-Path $dir.FullName 'stage9_llm_rl_comparison\llm_stage9_llm_rl_comparison.parquet') $Python)
        ) { $complete++ }
    }
    return ($complete -ge 15)
}

function Test-FrozenPostFreezeAnalysisReady([string]$Root) {
    $analysisRoot = Join-Path $Root 'data\analysis\paper_repro'
    $manifestPath = Join-Path $analysisRoot '00_manifest\paper_repro_analysis_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return $false }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([string]$manifest.status -ne 'PASS') { return $false }
        if ([string]$manifest.analysis_profile -ne 'historical_20260715') { return $false }
    } catch { return $false }
    foreach ($relative in @(
        '01_substrate_validation',
        '02_llm_hypothesis_tests',
        '03_output_contract_diagnostics',
        '04_paper_assets',
        '06_thesis_registry'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $analysisRoot $relative) -PathType Container)) { return $false }
    }
    return $true
}

function Move-WorkingSetAside([string]$Root, [string]$RunRoot, [string]$Reason) {
    $saved = Join-Path $RunRoot 'saved_previous_working_set'
    New-Item -ItemType Directory -Path $saved -Force | Out-Null
    $ownerMarker = Join-Path $Root 'data\final_freeze\.working_copy_run_id.txt'
    $previousRunId = if (Test-Path -LiteralPath $ownerMarker -PathType Leaf) {
        (Get-Content -LiteralPath $ownerMarker -Raw).Trim()
    } else { '' }
    $moved = @{}
    foreach ($name in @('final_freeze','analysis')) {
        $source = Join-Path $Root "data\$name"
        if (-not (Test-Path -LiteralPath $source -PathType Container)) { continue }
        if ((Get-ChildItem -LiteralPath $source -Force | Measure-Object).Count -eq 0) { continue }
        $destination = Join-Path $saved $name
        if (Test-Path -LiteralPath $destination) { throw "기존 작업본 보관 위치가 이미 있습니다: $destination" }
        Move-Item -LiteralPath $source -Destination $destination
        $moved[$name] = $destination
        New-Item -ItemType Directory -Path $source -Force | Out-Null
    }
    if ($previousRunId -and $moved.ContainsKey('final_freeze') -and $moved.ContainsKey('analysis')) {
        $previousInfo = Join-Path $Root "data\runs\$previousRunId\RUN_INFO.txt"
        if (Test-Path -LiteralPath $previousInfo -PathType Leaf) {
            $updated = foreach ($line in Get-Content -LiteralPath $previousInfo -Encoding UTF8) {
                if ($line -like 'final_freeze=*') { "final_freeze=$($moved['final_freeze'])" }
                elseif ($line -like 'analysis=*') { "analysis=$($moved['analysis'])" }
                else { $line }
            }
            Set-Content -LiteralPath $previousInfo -Value $updated -Encoding UTF8
        }
    }
    Set-Content -LiteralPath (Join-Path $saved 'REASON.txt') -Value $Reason -Encoding UTF8
}

function Initialize-FrozenWorkingCopy([string]$Root, [string]$RunRoot) {
    $source = Join-Path $Root 'frozen_outputs\final_freeze'
    $target = Join-Path $Root 'data\final_freeze'
    if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw "보존 산출물이 없습니다: $source" }
    $marker = Join-Path $target '.working_copy_source.txt'
    $kind = if (Test-Path -LiteralPath $marker) { (Get-Content -LiteralPath $marker -Raw).Trim() } else { '' }
    if ($kind -ne 'FROZEN_SNAPSHOT' -and (Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) {
        Move-WorkingSetAside $Root $RunRoot 'FrozenReplay 시작 전 기존 clean/unknown 작업본 보관'
    }
    if ((Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
        Write-Host '보존 산출물을 data/final_freeze 작업본으로 복사합니다.' -ForegroundColor Cyan
        & robocopy.exe $source $target /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /XJ /NFL /NDL /NJH /NJS /NP
        if ($LASTEXITCODE -ge 8) { throw "보존 산출물 복사 실패 (robocopy=$LASTEXITCODE)" }
        Set-Content -LiteralPath $marker -Value 'FROZEN_SNAPSHOT' -Encoding ASCII
    }
    Set-Content -LiteralPath (Join-Path $target '.working_copy_run_id.txt') -Value (Split-Path -Leaf $RunRoot) -Encoding UTF8
}

function Initialize-CleanWorkingCopy([string]$Root, [string]$RunRoot) {
    $target = Join-Path $Root 'data\final_freeze'
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    $marker = Join-Path $target '.working_copy_source.txt'
    $kind = if (Test-Path -LiteralPath $marker) { (Get-Content -LiteralPath $marker -Raw).Trim() } else { '' }
    if ($kind -eq 'FROZEN_SNAPSHOT') {
        Move-WorkingSetAside $Root $RunRoot 'clean run 시작 전 FrozenReplay 작업본 보관'
    }
    Set-Content -LiteralPath (Join-Path $target '.working_copy_source.txt') -Value 'CLEAN_FROM_RAW' -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $target '.working_copy_run_id.txt') -Value (Split-Path -Leaf $RunRoot) -Encoding UTF8
}

function Invoke-OriginalOracle([string]$Root, [string]$Python) {
    Invoke-Checked '원본 Oracle Stage0-1' @(
        'powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
        (Join-Path $Root 'tools\internal\original\run_oracle_stage0_stage1.ps1'),
        '-ProjectRoot',$Root,'-PythonExe',$Python,
        '-RawAllDir',(Join-Path $Root 'data\raw\raw_all'),
        '-RawRatingDir',(Join-Path $Root 'data\raw\rating_sample'),
        '-RunVerifiers'
    )
}

function Invoke-OriginalRL([string]$Root, [string]$Python) {
    $cashFlow = Join-Path $Root 'data\final_freeze\stage1_oracle_inputs\stage00_01_rating_statement_integration\cleaned_statement_panels\현금흐름표_clean.parquet'
    Invoke-Checked '원본 Stage2A raw action source' @(
        $Python,'-m','credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline',
        '--project-root',$Root,'--raw-all-dir',(Join-Path $Root 'data\raw\raw_all')
    )
    Invoke-Checked '원본 Stage2 input splits' @(
        $Python,'-m','credit_recourse.rl.pipelines.final_stage2_input_splits.pipeline',
        '--project-root',$Root,'--join-cash-flow-substrate','--cash-flow-encoder-mode','reward_only','--cash-flow-panel',$cashFlow
    )
    Invoke-Checked '원본 Stage2 candidate projection' @(
        $Python,'-m','credit_recourse.rl.pipelines.final_stage2_candidate_projection.pipeline',
        '--project-root',$Root,'--rho','0.0','--merton-lambda','0.05','--fcff-lambda','0.45','--liquidity-lambda','0.45'
    )
    Invoke-Checked '원본 Stage2 counterfactual handoff' @(
        $Python,'-m','credit_recourse.rl.pipelines.final_stage2_counterfactual_transitions.pipeline',
        '--project-root',$Root,'--magnitude-quantile','50','--reward-mode','phi_merton_fcff_liquidity',
        '--done-mode','terminal','--fidelity-gate','warn','--max-rel-err-assets','0.05',
        '--sim-business-plan-mode','calibrated','--preserve-current-non-current-residual'
    )
    Invoke-Checked '원본 LoopA/B2 substrate' @(
        'powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
        (Join-Path $Root 'tools\internal\original\run_loopA_loopB2_stage2_extension.ps1'),
        '-ProjectRoot',$Root,'-SimBusinessPlanMode','calibrated','-MagnitudeQuantile','50',
        '-MaxRelErrAssets','0.10'
    )
    Invoke-Checked '원본 RL Stage2-6 final-paper preset' @(
        'powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
        (Join-Path $Root 'tools\internal\original\run_rl_unified_stage3456.ps1'),
        '-ProjectRoot',$Root,
        '-RunLabel','sector0_m0p05_f0p45_liq0p45_s2_s3train_g0p60_tau0p85_b15_cql0_distill1p5_qpareto',
        '-RunStage2','-Stage2Rho','0.0','-Stage2MertonLambda','0.05','-Stage2FcffLambda','0.45','-Stage2LiquidityLambda','0.45',
        '-JoinCashFlowSubstrate','-CashFlowEncoderMode','reward_only','-CashFlowPanel',$cashFlow,
        '-CounterfactualTransitions','-CounterfactualRewardMode','phi_merton_fcff_liquidity','-CounterfactualDoneMode','terminal',
        '-CounterfactualFidelityGate','warn','-CounterfactualMaxRelErrAssets','0.05','-SimBusinessPlanMode','calibrated',
        '-PreserveCurrentNonCurrentResidual',
        '-Stage3Mode','train','-Stage3BatchSize','512','-Stage3Epochs','30','-Stage3Seed','2',
        '-Stage3MaskingRatio','0.15','-Stage3LearningRate','3e-4','-Stage3WeightDecay','1e-5',
        '-MagnitudeQuantile','50','-Stage4Epochs','80','-Stage4BatchSize','512','-Stage4Seed','2',
        '-FamilyBalanced','-FamilyBalancePower','0.5','-FamilyWeightCap','3.0','-CombinedWeightCap','5.0',
        '-Stage5Epochs','80','-Stage5BatchSize','128','-Stage5Seed','2','-Stage5LearningRate','1e-4','-Stage5WeightDecay','2e-3',
        '-Gamma','0.60','-ExpectileTau','0.85','-Beta','15.0','-CqlAlpha','0.0',
        '-DistillLambda','1.5','-DistillMarginMin','0.01','-DistillTemperature','1.0',
        '-Stage5CriticHeadArch','cross_attention','-CrossAttnBlocks','2','-CrossAttnHeads','4','-CrossAttnDropout','0.25',
        '-ActorExtractionMode','awr','-ActorHeadArch','linear','-Stage5SelectionMetric','q_pareto_knee','-VerifierMode','warn'
    )
}

function Invoke-OriginalLLM([string]$Root, [string]$Python) {
    # Key checks occur only here, after reusable Oracle/RL work is safely kept.
    if (-not $env:OPENAI_API_KEY) { throw '첫 live API 호출 직전 중단: OPENAI_API_KEY가 없습니다.' }
    $tag = Get-Date -Format 'yyyyMMdd_HHmmss'
    $matrix = Join-Path $Root 'tools\internal\original\run_llm789_fresh_all_single_repo.ps1'
    foreach ($ic in @('IC-a','IC-b','IC-c')) {
        Invoke-Checked "원본 primary GPT-5.4-mini $ic" @('powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',$matrix,'-ProjectRoot',$Root,'-RunRole','paper_primary_gpt54','-Backends','openai:gpt-5.4-mini','-InformationConditions',$ic,'-RunLabelPrefix','paper','-DateTag',$tag,'-OpenAIApiMode','responses','-OpenAIReasoningEffort','low','-OpenAIMaxOutputTokens','1200','-Fresh')
    }
    Invoke-Checked '원본 supplementary GPT-4.1-mini IC-b' @('powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',$matrix,'-ProjectRoot',$Root,'-RunRole','paper_supplementary_gpt41','-Backends','openai:gpt-4.1-mini','-InformationConditions','IC-b','-RunLabelPrefix','paper','-DateTag',$tag,'-Fresh')
    Invoke-Checked '원본 N5 L1=1.27 IC-a/b/c' @('powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $Root 'tools\internal\original\run_n5_main_gpt54mini_icab.ps1'),'-ProjectRoot',$Root,'-DateTag',$tag)
    Invoke-Checked '원본 N5M IC-b budget frontier' @('powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $Root 'tools\internal\original\run_n5_budget_frontier_icb.ps1'),'-ProjectRoot',$Root,'-DateTag',$tag,'-BudgetC4')
    $probeLabel = "paper_paper_icc_probe_ICc_gpt54mini_seed1_$tag"
    Invoke-Checked '원본 IC-c probe' @($Python,'-m','credit_recourse.rl.pipelines.final_stage7_llm_action_generation.icc_probe_runner','--project-root',$Root,'--run-label',$probeLabel,'--run-role','paper_icc_probe','--base-run-role','paper_primary_gpt54','--icc-probe-tolerance','0.20','--max-retries','3','--retry-sleep-seconds','20')
    if (-not $env:ANTHROPIC_API_KEY) { throw 'Anthropic live API 호출 직전 중단: ANTHROPIC_API_KEY가 없습니다. 앞서 완료된 OpenAI 결과는 유지됩니다.' }
    Invoke-Checked '원본 supplementary Haiku 4.5 IC-b' @('powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',$matrix,'-ProjectRoot',$Root,'-RunRole','paper_supplementary_haiku45','-Backends','anthropic:claude-haiku-4-5-20251001','-InformationConditions','IC-b','-RunLabelPrefix','paper','-DateTag',$tag,'-Fresh')
}

function Invoke-V10Analysis([string]$Root, [string]$Python, [string]$SitePackages, [string]$RunRoot, [bool]$FrozenReplay) {
    $environmentConfig = Join-Path $Root 'src\credit_recourse\configs\frozen_python_environment.json'
    $contractId = [string](Get-Content -LiteralPath $environmentConfig -Raw -Encoding UTF8 | ConvertFrom-Json).contract_id
    New-Item -ItemType Directory -Path (Join-Path $RunRoot 'tmp') -Force | Out-Null
    $env:CREDIT_RECOURSE_ARTIFACT_ROOT = Join-Path $Root 'data\final_freeze'
    $env:CREDIT_RECOURSE_ANALYSIS_ROOT = Join-Path $Root 'data\analysis'
    $env:CREDIT_RECOURSE_RAW_ROOT = Join-Path $Root 'data\raw'
    $workingConfigRoot = Join-Path $Root 'data\final_freeze\configs'
    New-Item -ItemType Directory -Path $workingConfigRoot -Force | Out-Null
    foreach ($analysisConfig in @('n5m_adaptive_selection_contract.json')) {
        Copy-Item -LiteralPath (Join-Path $Root "src\credit_recourse\configs\$analysisConfig") -Destination (Join-Path $workingConfigRoot $analysisConfig) -Force
    }
    $env:CREDIT_RECOURSE_RUN_HISTORICAL_N5M = '1'
    # The preserved V10 extension checkpoint catalog is a fixed part of the
    # historical thesis analysis profile.  Set its process-local context here
    # so the public runner does not depend on undocumented caller state.
    $env:CREDIT_RECOURSE_CHECKPOINT_CATALOG_COUNT = '21'
    $env:CREDIT_RECOURSE_CHECKPOINT_PROFILE = 'historical_20260715'
    $env:CREDIT_RECOURSE_CHECKPOINT_REPOSITORY_ROOT = $Root
    $env:CREDIT_RECOURSE_CHECKPOINT_STORE = Join-Path $RunRoot 'extension_checkpoints'
    $common = @(
        '-ProjectRoot',$Root,'-PythonExe',$Python,
        '-PythonSourceRoot',(Join-Path $Root 'src'),
        '-PythonSitePackagesRoot',$SitePackages,
        '-PythonEnvironmentConfig',$environmentConfig,
        '-PythonEnvironmentContractId',$contractId,
        '-ExecutionRunRoot',$RunRoot,
        '-TemporaryRoot',(Join-Path $RunRoot 'tmp')
    )
    $analysisArgs = @(
        'powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
        (Join-Path $Root 'tools\internal\v10\run_postfreeze_analysis.ps1')
    ) + $common + @('-RawRoot',(Join-Path $Root 'data\raw'))
    $analysisProfile = if ($FrozenReplay) { 'historical_20260715' } else { 'current_comprehensive' }
    $resumePartial = $null
    $partialArchiveRoot = Join-Path $Root 'data\reproduction\archive\paper_repro_rebuilds'
    if (Test-Path -LiteralPath $partialArchiveRoot -PathType Container) {
        foreach ($candidate in Get-ChildItem -LiteralPath $partialArchiveRoot -Directory | Sort-Object LastWriteTime -Descending) {
            if ($candidate.Name -notlike 'failed_*') { continue }
            $manifestPath = Join-Path $candidate.FullName '00_manifest\paper_repro_analysis_manifest.json'
            if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { continue }
            try { $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { continue }
            if ([string]$manifest.status -eq 'FAIL' -and [string]$manifest.analysis_profile -eq $analysisProfile) {
                $resumePartial = $candidate.FullName
                break
            }
        }
    }
    if ($resumePartial) {
        Write-Host "보존된 부분 계산에서 이어갑니다: $resumePartial" -ForegroundColor Yellow
        $analysisArgs += @('-ResumePartialAnalysisPath',$resumePartial)
    } else {
        $analysisArgs += '-ReplaceOutput'
    }
    if ($FrozenReplay) {
        $eligibleRunCatalog = Join-Path $Root 'frozen_outputs\analysis\paper_repro\00_manifest\archived_llm_run_catalog.csv'
        if (-not (Test-Path -LiteralPath $eligibleRunCatalog -PathType Leaf)) {
            throw "V10 역사적 LLM 실행 선택 목록이 없습니다: $eligibleRunCatalog"
        }
        $analysisArgs += @('-AnalysisProfile',$analysisProfile,'-EligibleRunCatalog',$eligibleRunCatalog)
    } else {
        $analysisArgs += @('-AnalysisProfile',$analysisProfile)
    }
    if ($FrozenReplay -and (Test-FrozenPostFreezeAnalysisReady $Root)) {
        Write-Host '완료된 FrozenReplay post-freeze 분석을 이어서 사용합니다.' -ForegroundColor Green
    } else {
        Invoke-Checked 'V10 post-freeze 분석' $analysisArgs
    }
    Invoke-Checked 'V10 E2/E3/E4 확장 분석' (@('powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $Root 'tools\internal\v10\run_analysis_extensions.ps1')) + $common + @('-MaxParallelAxisTasks','1','-ReplaceOutput'))
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { throw '스크립트 위치에서 저장소 루트를 결정하지 못했습니다. -ProjectRoot를 지정하세요.' }
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) { throw "저장소 루트를 찾지 못했습니다: $ProjectRoot" }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (-not $Mode) { $Mode = Select-Mode }
if (-not $RunId) { $RunId = '{0}_{1}' -f $Mode,(Get-Date -Format 'yyyyMMdd_HHmmss') }
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'RunId에는 영문, 숫자, 점, 밑줄, 하이픈만 사용할 수 있습니다.' }
$PythonExe = Resolve-Python $PythonExe $ProjectRoot
if (-not $PythonSitePackagesRoot) {
    $purelibLines = @(& $PythonExe -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>$null)
    $purelibExit = $LASTEXITCODE
    $PythonSitePackagesRoot = ($purelibLines -join [Environment]::NewLine).Trim()
    if ($purelibExit -ne 0 -or [string]::IsNullOrWhiteSpace($PythonSitePackagesRoot)) {
        throw "Python site-packages 경로를 확인하지 못했습니다: $PythonExe"
    }
}
if (-not (Test-Path -LiteralPath $PythonSitePackagesRoot -PathType Container)) { throw "Python site-packages 경로를 찾지 못했습니다: $PythonSitePackagesRoot" }
$PythonSitePackagesRoot = (Resolve-Path -LiteralPath $PythonSitePackagesRoot).Path
Import-LocalEnvironment (Join-Path $ProjectRoot '.env.local')
$env:REPRO_PYTHON_EXE = $PythonExe
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = (Join-Path $ProjectRoot 'src') + [System.IO.Path]::PathSeparator + $PythonSitePackagesRoot

if ($PlanOnly) {
    Write-Host ''
    Write-Host "실행 계획: $Mode" -ForegroundColor Cyan
    if ($Mode -eq 'FrozenReplay') {
        $snapshot = Join-Path $ProjectRoot 'frozen_outputs\final_freeze'
        $catalog = Join-Path $ProjectRoot 'frozen_outputs\analysis\paper_repro\00_manifest\archived_llm_run_catalog.csv'
        if (-not (Test-Path -LiteralPath $snapshot -PathType Container)) { throw "보존 산출물이 없습니다: $snapshot" }
        if (-not (Test-Path -LiteralPath $catalog -PathType Leaf)) { throw "역사적 LLM 실행 목록이 없습니다: $catalog" }
        Write-Host '  READ frozen_outputs/final_freeze'
        Write-Host '  RUN  V10 post-freeze analysis (학습/API 호출 없음)'
        Write-Host '  RUN  V10 E2/E3/E4 extension analysis'
        exit 0
    }

    foreach ($raw in @('data\raw\raw_all','data\raw\rating_sample')) {
        $path = Join-Path $ProjectRoot $raw
        if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "clean run 원자료가 없습니다: $path" }
    }
    $marker = Join-Path $ProjectRoot 'data\final_freeze\.working_copy_source.txt'
    $cleanWorkingCopy = (Test-Path -LiteralPath $marker -PathType Leaf) -and ((Get-Content -LiteralPath $marker -Raw).Trim() -eq 'CLEAN_FROM_RAW')
    $oracleReady = $cleanWorkingCopy -and (Test-OracleReady $ProjectRoot $PythonExe)
    $rlReady = $cleanWorkingCopy -and (Test-RLReady $ProjectRoot $PythonExe)
    $llmReady = $cleanWorkingCopy -and (Test-LLMReady $ProjectRoot $PythonExe)
    if ($oracleReady) { Write-Host '  SKIP Oracle (현재 clean 산출물 사용 가능)' }
    else { Write-Host '  RUN  raw -> 원본 Oracle Stage0-1' }
    if ($Mode -in @('OracleRLClean','OracleRLLLMClean','FullClean')) {
        if ($rlReady) { Write-Host '  SKIP RL (현재 clean 산출물 사용 가능)' }
        else { Write-Host '  RUN  Oracle -> 원본 RL Stage2-6' }
    }
    if ($Mode -in @('OracleRLLLMClean','FullClean')) {
        if ($llmReady) { Write-Host '  SKIP LLM Stage7-9 (현재 clean 산출물 사용 가능)' }
        else { Write-Host '  RUN  RL -> live LLM Stage7-9; API key는 첫 호출 직전에만 확인' }
    }
    if ($Mode -eq 'FullClean') {
        Write-Host '  RUN  clean Stage0-9 산출물 -> V10 analysis'
    }
    exit 0
}

$runRoot = Join-Path $ProjectRoot "data\runs\$RunId"
New-Item -ItemType Directory -Path $runRoot,(Join-Path $ProjectRoot 'data\analysis'),(Join-Path $ProjectRoot 'data\thesis_outputs') -Force | Out-Null
$started = Get-Date
$exitStatus = 1
try {
    if ($Mode -eq 'FrozenReplay') {
        Initialize-FrozenWorkingCopy $ProjectRoot $runRoot
        if (-not (Test-OracleReady $ProjectRoot $PythonExe) -or -not (Test-RLReady $ProjectRoot $PythonExe) -or -not (Test-LLMReady $ProjectRoot $PythonExe)) {
            throw '보존 작업본에서 Oracle/RL/LLM 필수 산출물을 열 수 없습니다.'
        }
        Invoke-V10Analysis $ProjectRoot $PythonExe $PythonSitePackagesRoot $runRoot $true
    } else {
        Initialize-CleanWorkingCopy $ProjectRoot $runRoot
        if (Test-OracleReady $ProjectRoot $PythonExe) { Write-Host 'Oracle 산출물을 열 수 있어 재사용합니다.' -ForegroundColor Green }
        else { Invoke-OriginalOracle $ProjectRoot $PythonExe }
        if ($Mode -in @('OracleRLClean','OracleRLLLMClean','FullClean')) {
            if (Test-RLReady $ProjectRoot $PythonExe) { Write-Host 'RL 산출물을 열 수 있어 재사용합니다.' -ForegroundColor Green }
            else { Invoke-OriginalRL $ProjectRoot $PythonExe }
        }
        if ($Mode -in @('OracleRLLLMClean','FullClean')) {
            if (Test-LLMReady $ProjectRoot $PythonExe) { Write-Host 'LLM Stage7-9 산출물을 열 수 있어 재사용합니다.' -ForegroundColor Green }
            else { Invoke-OriginalLLM $ProjectRoot $PythonExe }
        }
        if ($Mode -eq 'FullClean') { Invoke-V10Analysis $ProjectRoot $PythonExe $PythonSitePackagesRoot $runRoot $false }
    }
    $exitStatus = 0
} finally {
    $finished = Get-Date
    $lines = @(
        "run_id=$RunId",
        "mode=$Mode",
        "started=$($started.ToUniversalTime().ToString('o'))",
        "finished=$($finished.ToUniversalTime().ToString('o'))",
        "exit_status=$exitStatus",
        "final_freeze=$ProjectRoot\data\final_freeze",
        "analysis=$ProjectRoot\data\analysis"
    )
    Set-Content -LiteralPath (Join-Path $runRoot 'RUN_INFO.txt') -Value $lines -Encoding UTF8
}

Write-Host ''
Write-Host "완료: $Mode / $RunId" -ForegroundColor Green
Write-Host "산출물: $ProjectRoot\data\final_freeze, $ProjectRoot\data\analysis"
exit 0
