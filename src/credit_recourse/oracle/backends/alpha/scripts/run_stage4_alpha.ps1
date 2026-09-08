# =============================================================================
# Stage 4 alpha -- Local Windows Reproduction Runner (definitive)
# =============================================================================
$ErrorActionPreference = 'Stop'

chcp 65001 | Out-Null
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Stage 4 alpha -- Performance-Optimized Scorecard Oracle" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$BundleDir    = Split-Path -Parent $PSScriptRoot
$InputsDir    = Join-Path $BundleDir "inputs"
$OutputsDir   = Join-Path $BundleDir "outputs"
$SiblingsRoot = Split-Path -Parent $BundleDir
Write-Host "Bundle dir    : $BundleDir"
Write-Host "Siblings root : $SiblingsRoot"
New-Item -ItemType Directory -Path $InputsDir  -Force | Out-Null
New-Item -ItemType Directory -Path $OutputsDir -Force | Out-Null

# -- Helpers ------------------------------------------------------------------
function Stop-Runner($message) {
    Write-Error $message
    exit 2
}

function Find-SiblingOutputs($pattern) {
    $candidates = @(Get-ChildItem -Path $SiblingsRoot -Directory -Filter $pattern -ErrorAction SilentlyContinue)
    if ($candidates.Count -eq 0) { return $null }
    $bundle = ($candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    $outDir = Join-Path $bundle "outputs"
    if (-not (Test-Path $outDir)) {
        Stop-Runner "Bundle found ($bundle) but outputs/ missing. Run that stage first."
    }
    return $outDir
}

# Recursive search under outputs/, EXCLUDES paths containing \inputs\
# This prevents picking up stale cached files from a sibling bundle's inputs/.
function Resolve-OutputFile($srcOutputsDir, $filename, [string[]]$PreferSubpaths = @()) {
    $direct = Join-Path $srcOutputsDir $filename
    if (Test-Path $direct) { return (Get-Item $direct) }

    $matches = @(Get-ChildItem -Path $srcOutputsDir -Recurse -File -Filter $filename `
                  -ErrorAction SilentlyContinue |
                 Where-Object { $_.FullName -notmatch '\\inputs\\' })
    if ($matches.Count -eq 0) {
        Stop-Runner "Required file not found under $srcOutputsDir : $filename"
    }
    foreach ($pref in $PreferSubpaths) {
        $preferred = @($matches | Where-Object { $_.FullName -like "*$pref*" } |
                       Sort-Object LastWriteTime -Descending)
        if ($preferred.Count -gt 0) { return $preferred[0] }
    }
    if ($matches.Count -gt 1) {
        Write-Host "  [WARN] $($matches.Count) candidates for $filename; using newest:" -ForegroundColor Yellow
        foreach ($m in ($matches | Sort-Object LastWriteTime -Descending | Select-Object -First 5)) {
            Write-Host "         $($m.FullName)"
        }
    }
    return ($matches | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
}

function Copy-FromOutputs($srcOutputsDir, $destSubdir, $filename, [string[]]$PreferSubpaths = @()) {
    $dest = Join-Path $InputsDir $destSubdir
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    $srcFile = Resolve-OutputFile $srcOutputsDir $filename $PreferSubpaths
    Copy-Item $srcFile.FullName (Join-Path $dest $filename) -Force
    Write-Host "  copied: $destSubdir\$filename"
    Write-Host "          from: $($srcFile.FullName)"
}

# Robust JSON parser that handles PS 5.1 array quirks
function Get-SelectedVariableIds($jsonPath) {
    try {
        $raw = Get-Content -Path $jsonPath -Raw -Encoding UTF8
        $data = $raw | ConvertFrom-Json
        $items = $null
        if ($data -is [System.Collections.IEnumerable] -and $data -isnot [string]) {
            $items = @($data)
        } elseif ($null -ne $data.selected_variables) {
            $items = @($data.selected_variables)
        } elseif ($null -ne $data.variables) {
            $items = @($data.variables)
        } else {
            $items = @($data)
        }
        $ids = @()
        foreach ($item in $items) {
            if ($null -ne $item -and $null -ne $item.variable_id) {
                $ids += [string]$item.variable_id
            }
        }
        return $ids
    } catch {
        Write-Host "  [WARN] Failed to parse JSON $jsonPath : $_" -ForegroundColor Yellow
        return @()
    }
}

function Test-SameSet([string[]]$A, [string[]]$B) {
    if ($A.Count -ne $B.Count) { return $false }
    $sa = @($A | Sort-Object)
    $sb = @($B | Sort-Object)
    for ($i = 0; $i -lt $sa.Count; $i++) {
        if ($sa[$i] -ne $sb[$i]) { return $false }
    }
    return $true
}

function Copy-CanonicalStage3($stage3OutputsDir) {
    $dest = Join-Path $InputsDir "stage3_v2"
    New-Item -ItemType Directory -Path $dest -Force | Out-Null

    $canonical = @(
        "R006","R064","R086","R131","R157","R185",
        "industry_bad_grade_share_lag1_self_excl",
        "cap_change_count_3y","log_assets",
        "operating_loss_freq_3y","financial_data_completeness"
    )
    $forbidden = @("kospi_dummy","sector_7","industry_avg_rating",
                   "sector_year_rating_count","sector_year_rating_count_lag1")

    $candidates = @(Get-ChildItem -Path $stage3OutputsDir -Recurse -File `
                     -Filter "selected_variables_v2.json" -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -notmatch '\\inputs\\' })
    if ($candidates.Count -eq 0) {
        Stop-Runner "No selected_variables_v2.json under $stage3OutputsDir"
    }

    Write-Host "  Stage 3 selected_variables_v2.json candidates:"
    $valid = @()
    foreach ($c in ($candidates | Sort-Object LastWriteTime -Descending)) {
        $ids = @(Get-SelectedVariableIds $c.FullName)
        $bad = @($ids | Where-Object { $forbidden -contains $_ })
        $dirPath = Join-Path $c.DirectoryName "direction_encoding_v2.json"
        $hasDir = Test-Path $dirPath
        $canon = (Test-SameSet $ids $canonical)

        $tags = @()
        if ($canon)  { $tags += "CANONICAL" } else { $tags += "NON_CANONICAL" }
        if ($bad.Count -gt 0) { $tags += "FORBIDDEN=$($bad -join ',')" }
        if (-not $hasDir) { $tags += "NO_DIRECTION" }

        Write-Host "    - $($c.FullName) [$($tags -join '; ')]"
        Write-Host "      ids ($($ids.Count)): $($ids -join ', ')"

        if ($canon -and $hasDir -and $bad.Count -eq 0) { $valid += $c }
    }

    if ($valid.Count -eq 0) {
        Stop-Runner "No canonical Stage 3 v2 selected_variables_v2.json found. Re-run Stage 3 with current Stage 1C v3.2 inputs."
    }

    $chosen = ($valid | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    Write-Host "  [CHOSEN Stage 3] $($chosen.DirectoryName)" -ForegroundColor Green
    foreach ($name in @("selected_variables_v2.json","direction_encoding_v2.json")) {
        Copy-Item (Join-Path $chosen.DirectoryName $name) (Join-Path $dest $name) -Force
        Write-Host "  copied: stage3_v2\$name"
    }
}

# -- Locate ------------------------------------------------------------------
$Out1B = Find-SiblingOutputs "stage1b_*"
$Out1C = Find-SiblingOutputs "stage1c_v3_*"
$Out2  = Find-SiblingOutputs "stage2_v3_*"
$Out3  = Find-SiblingOutputs "stage3_v2_*"
foreach ($pair in @(@($Out1B,"stage1b_*"), @($Out1C,"stage1c_v3_*"),
                    @($Out2,"stage2_v3_*"), @($Out3,"stage3_v2_*"))) {
    if (-not $pair[0]) { Stop-Runner "Bundle not found: $($pair[1]) under $SiblingsRoot" }
    Write-Host "  outputs: $($pair[0])"
}

# -- Copy --------------------------------------------------------------------
Write-Host "`nFetching inputs..."
Copy-FromOutputs $Out1B "stage1b"    "firm_year_panel_v1.parquet"          @("05_outputs","stage1b_v1")
Copy-FromOutputs $Out2  "stage2"     "engineered_financial_ratios.parquet" @("05_outputs","outputs")
Copy-FromOutputs $Out1C "stage1c_v3" "nonfinancial_metadata_panel.parquet" @("05_outputs","stage1c_v3_2","outputs")
Copy-CanonicalStage3 $Out3

# -- Python venv -------------------------------------------------------------
Write-Host "`nSetting up Python environment..."
$VenvDir    = Join-Path $BundleDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Stop-Runner "Failed to create venv" }
}
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $BundleDir "requirements.txt") --prefer-binary --quiet
if ($LASTEXITCODE -ne 0) { Stop-Runner "pip install (requirements.txt) failed" }

# optuna: binary-only install (avoids C++ Build Tools requirement on Windows)
Write-Host "  Installing optuna (binary-only)..."
& $VenvPython -m pip install "optuna>=3.6.0" --only-binary=:all: --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  --only-binary failed, trying without constraint..." -ForegroundColor Yellow
    & $VenvPython -m pip install "optuna>=3.6.0" --prefer-binary --quiet
    if ($LASTEXITCODE -ne 0) { Stop-Runner "optuna install failed" }
}

# -- Run ---------------------------------------------------------------------
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Running Stage 4 alpha pipeline..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
& $VenvPython -X utf8 (Join-Path $BundleDir "stage4_alpha_pipeline.py")
if ($LASTEXITCODE -ne 0) { Stop-Runner "Pipeline failed (exit code $LASTEXITCODE)" }

# -- Verify ------------------------------------------------------------------
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "Verifying acceptance gates..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
& $VenvPython -X utf8 (Join-Path $BundleDir "verify_alpha_metrics.py")
$verifyExit = $LASTEXITCODE

Write-Host "`n============================================================" -ForegroundColor Cyan
if ($verifyExit -eq 0) {
    Write-Host "Stage 4 alpha complete + all acceptance gates passed" -ForegroundColor Green
} else {
    Write-Host "Acceptance check failed -- review outputs/acceptance_alpha.csv" -ForegroundColor Yellow
}
Write-Host "Outputs: $OutputsDir" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
exit $verifyExit
