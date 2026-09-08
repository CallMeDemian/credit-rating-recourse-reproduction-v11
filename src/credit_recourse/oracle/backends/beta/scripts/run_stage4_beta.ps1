# =============================================================================
# Stage 4 beta ordered-logit -- Project-root runner
# Current thesis_repo layout supported:
#   data/final_freeze/stage00_01_rating_statement_integration
#   data/final_freeze/stage00_02_financial_ratio_engineering
#   data/final_freeze/stage00_03_nonfinancial_metadata
#   data/final_freeze/stage00_04_variable_selection
#   data/final_freeze/stage01_alpha_vanilla
# Outputs are written outside the source bundle:
#   data/final_freeze/stage01_beta_ordered_logit
# =============================================================================
param(
    [string]$ProjectRoot = "",
    [switch]$ForceRefreshInputs
)

$ErrorActionPreference = 'Stop'

chcp 65001 | Out-Null
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Stop-Runner($message) {
    Write-Error $message
    exit 2
}

function Resolve-ProjectRoot([string]$ExplicitRoot, [string]$StartDir) {
    if (-not [string]::IsNullOrWhiteSpace($ExplicitRoot)) {
        $root = (Resolve-Path $ExplicitRoot).Path
        if (-not (Test-Path (Join-Path $root "data\final_freeze"))) {
            Stop-Runner "ProjectRoot does not contain data\final_freeze: $root"
        }
        return $root
    }

    $dir = Get-Item $StartDir
    while ($null -ne $dir) {
        if (Test-Path (Join-Path $dir.FullName "data\final_freeze")) {
            return $dir.FullName
        }
        $dir = $dir.Parent
    }

    $cwd = (Get-Location).Path
    if (Test-Path (Join-Path $cwd "data\final_freeze")) { return $cwd }

    Stop-Runner "Could not infer project root. Re-run with: -ProjectRoot C:\Users\Demian\Desktop\thesis_repo"
}

function Copy-RequiredFile([string]$Source, [string]$Dest) {
    if (-not (Test-Path $Source)) {
        Stop-Runner "Required input missing: $Source"
    }
    $destDir = Split-Path -Parent $Dest
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    if ((Test-Path $Dest) -and (-not $ForceRefreshInputs)) {
        Write-Host "  [KEEP] $Dest" -ForegroundColor DarkGray
        return
    }
    Copy-Item $Source $Dest -Force
    Write-Host "  [COPY] $Source"
    Write-Host "      -> $Dest"
}

function Resolve-Python([string]$Root, [string]$BundleDir) {
    $projectPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $projectPython) { return $projectPython }

    $bundleVenv = Join-Path $BundleDir ".venv"
    $bundlePython = Join-Path $bundleVenv "Scripts\python.exe"
    if (-not (Test-Path $bundlePython)) {
        Write-Host "  [venv] project .venv not found; creating bundle-local .venv" -ForegroundColor Yellow
        python -m venv $bundleVenv
        if ($LASTEXITCODE -ne 0) { Stop-Runner "python -m venv failed" }
    }
    return $bundlePython
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Stage 4 beta ordered-logit" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$BundleDir  = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Resolve-ProjectRoot $ProjectRoot $BundleDir
$InterimDir = Join-Path $ProjectRoot "data\final_freeze"
$InputsDir  = Join-Path $BundleDir "inputs"
$OutputsDir = Join-Path $InterimDir "stage01_beta_ordered_logit"
$ConfigPath = Join-Path $BundleDir "configs\stage4_beta_config.yaml"

Write-Host "Project root : $ProjectRoot"
Write-Host "Bundle dir   : $BundleDir"
Write-Host "Inputs dir   : $InputsDir"
Write-Host "Outputs dir  : $OutputsDir"

New-Item -ItemType Directory -Path $InputsDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputsDir -Force | Out-Null

Write-Host "`nPreparing inputs from data\final_freeze..." -ForegroundColor Yellow
Copy-RequiredFile `
    (Join-Path $InterimDir "stage00_01_rating_statement_integration\firm_year_panel_v1.parquet") `
    (Join-Path $InputsDir "stage1b\firm_year_panel_v1.parquet")
Copy-RequiredFile `
    (Join-Path $InterimDir "stage00_02_financial_ratio_engineering\engineered_financial_ratios.parquet") `
    (Join-Path $InputsDir "stage2\engineered_financial_ratios.parquet")
Copy-RequiredFile `
    (Join-Path $InterimDir "stage00_03_nonfinancial_metadata\nonfinancial_metadata_panel.parquet") `
    (Join-Path $InputsDir "stage1c_v3\nonfinancial_metadata_panel.parquet")
Copy-RequiredFile `
    (Join-Path $InterimDir "stage00_04_variable_selection\selected_variables_v2.json") `
    (Join-Path $InputsDir "stage3_v2\selected_variables_v2.json")
Copy-RequiredFile `
    (Join-Path $InterimDir "stage00_04_variable_selection\direction_encoding_v2.json") `
    (Join-Path $InputsDir "stage3_v2\direction_encoding_v2.json")
Copy-RequiredFile `
    (Join-Path $InterimDir "stage01_alpha_vanilla\oracle_alpha_params.json") `
    (Join-Path $InputsDir "stage4_alpha\oracle_alpha_params.json")
Copy-RequiredFile `
    (Join-Path $InterimDir "stage01_alpha_vanilla\oracle_firm_year_output_alpha.parquet") `
    (Join-Path $InputsDir "stage4_alpha\oracle_firm_year_output_alpha.parquet")

$PythonExe = Resolve-Python $ProjectRoot $BundleDir
Write-Host "`nPython      : $PythonExe"
& $PythonExe -m pip install -r (Join-Path $BundleDir "requirements.txt") --prefer-binary
if ($LASTEXITCODE -ne 0) { Stop-Runner "pip install failed" }
& $PythonExe -c "import yaml; print('PyYAML import OK')"
if ($LASTEXITCODE -ne 0) { Stop-Runner "PyYAML import check failed" }

$env:ORACLE_INPUT_DIR  = $InputsDir
$env:ORACLE_OUTPUT_DIR = $OutputsDir
$env:ORACLE_CONFIG     = $ConfigPath

Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Running Stage 4 beta ordered-logit pipeline..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
& $PythonExe -X utf8 (Join-Path $BundleDir "pipeline.py")
if ($LASTEXITCODE -ne 0) { Stop-Runner "beta pipeline failed" }

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Stage 4 beta ordered-logit complete." -ForegroundColor Green
Write-Host "Outputs: $OutputsDir" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
