$ErrorActionPreference = "Stop"

# UTF-8 logging guard for Windows PowerShell/cp949 consoles
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}

$bundle = "C:\Users\Demian\Desktop\credit_rating_llm_thesis_template\00_claude_did\stage4_alpha_bundle"
Set-Location $bundle

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python.exe" }

New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = ".\logs\stage4_alpha_pareto_$ts.out.log"
$errLog = ".\logs\stage4_alpha_pareto_$ts.err.log"
$verLog = ".\logs\verify_alpha_pareto_$ts.log"
$repairLog = ".\logs\repair_stage4_inputs_$ts.log"

function Show-LogTail($path, $title, $n = 80) {
    Write-Host ""
    Write-Host "---- $title : $path ----" -ForegroundColor Yellow
    if (Test-Path $path) {
        Get-Content -Path $path -Tail $n
    } else {
        Write-Host "(not found)"
    }
    Write-Host "---- end $title ----" -ForegroundColor Yellow
}

function Test-RequiredInputs() {
    $required = @(
        ".\inputs\stage1b\firm_year_panel_v1.parquet",
        ".\inputs\stage2\engineered_financial_ratios.parquet",
        ".\inputs\stage3_v2\selected_variables_v2.json",
        ".\inputs\stage3_v2\direction_encoding_v2.json"
    )
    $stage1cOk = (Test-Path ".\inputs\stage1c_v3\nonfinancial_metadata_panel.parquet") -or (Test-Path ".\inputs\stage1c\nonfinancial_metadata_panel.parquet")
    $missing = @()
    foreach ($p in $required) {
        if (-not (Test-Path $p)) { $missing += $p }
    }
    if (-not $stage1cOk) { $missing += ".\inputs\stage1c_v3\nonfinancial_metadata_panel.parquet OR .\inputs\stage1c\nonfinancial_metadata_panel.parquet" }
    return $missing
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Stage 4 alpha Pareto - Vanilla Isotonic Run (v5)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Bundle : $bundle"
Write-Host "  Output : .\outputs_pareto"
Write-Host "  Script : .\stage4_alpha_pipeline_pareto.py"
Write-Host "  Verify : .\verify_alpha_metrics_pareto.py"
Write-Host ""
Write-Host "  Logs:"
Write-Host "    repair: $repairLog"
Write-Host "    stdout: $outLog"
Write-Host "    stderr: $errLog"
Write-Host "    verify: $verLog"
Write-Host ""

$missingBefore = Test-RequiredInputs
if ($missingBefore.Count -gt 0) {
    Write-Host "[INFO] Missing required inputs. Running repair script..." -ForegroundColor Yellow
    Write-Host "  Missing:" -ForegroundColor Yellow
    $missingBefore | ForEach-Object { Write-Host "    - $_" -ForegroundColor Yellow }

    if (-not (Test-Path ".\scripts\repair_stage4_alpha_inputs.ps1")) {
        throw "Repair script not found: .\scripts\repair_stage4_alpha_inputs.ps1"
    }
    & ".\scripts\repair_stage4_alpha_inputs.ps1" -BundleRoot $bundle 1> $repairLog 2> ($repairLog + ".err")
    Show-LogTail $repairLog "repair log" 120
    if (Test-Path ($repairLog + ".err")) { Show-LogTail ($repairLog + ".err") "repair stderr" 80 }
}

$missingAfter = Test-RequiredInputs
if ($missingAfter.Count -gt 0) {
    Write-Host "[FAIL] Required inputs are still missing:" -ForegroundColor Red
    $missingAfter | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    throw "Input preflight failed. Run .\scripts\repair_stage4_alpha_inputs.ps1 manually or copy the missing files."
}
Write-Host "[OK] Input preflight passed" -ForegroundColor Green

if (Test-Path ".\outputs_pareto") {
    Write-Host "[INFO] Existing outputs_pareto found. Keeping it; files with same names may be overwritten." -ForegroundColor Yellow
}

Write-Host "[RUN] Pareto pipeline..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $py -ArgumentList @(".\stage4_alpha_pipeline_pareto.py") -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$pipeCode = $proc.ExitCode

if ($pipeCode -ne 0) {
    Write-Host "[FAIL] pipeline exit=$pipeCode" -ForegroundColor Red
    Show-LogTail $errLog "stderr tail" 160
    Show-LogTail $outLog "stdout tail" 100
    throw "Stage 4 alpha Pareto pipeline failed. See $errLog"
}

Write-Host "[OK] pipeline completed" -ForegroundColor Green
Write-Host "[RUN] verifier..." -ForegroundColor Cyan
$proc2 = Start-Process -FilePath $py -ArgumentList @(".\verify_alpha_metrics_pareto.py") -NoNewWindow -Wait -PassThru -RedirectStandardOutput $verLog -RedirectStandardError ($verLog + ".err")
$verifyCode = $proc2.ExitCode

Show-LogTail $verLog "verify log" 120
if (Test-Path ($verLog + ".err")) { Show-LogTail ($verLog + ".err") "verify stderr" 60 }

if ($verifyCode -eq 0) {
    Write-Host "[PASS] acceptance checks passed" -ForegroundColor Green
} else {
    Write-Host "[FAIL] verifier exit=$verifyCode - check .\outputs_pareto\acceptance_alpha_pareto.json" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Key outputs:" -ForegroundColor Cyan
Write-Host "  .\outputs_pareto\hyperparameter_pareto_search_alpha_pareto.csv"
Write-Host "  .\outputs_pareto\pareto_search_summary_alpha_pareto.json"
Write-Host "  .\outputs_pareto\oracle_firm_year_output_alpha_pareto.parquet"
Write-Host "  .\outputs_pareto\preliminary_dev_oot_metrics_alpha_pareto.csv"
Write-Host "  .\outputs_pareto\per_grade_em_alpha_pareto.csv"
Write-Host "  .\outputs_pareto\acceptance_alpha_pareto.json"
