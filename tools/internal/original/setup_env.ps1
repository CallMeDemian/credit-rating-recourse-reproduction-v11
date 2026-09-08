<#
.SYNOPSIS
    setup_env.ps1 - Create the Python virtual environment and install all dependencies
    for the credit-recourse (Oracle -> RL -> LLM) pipeline.

.DESCRIPTION
    Run this ONCE, before runner_play_script.

    It creates the venv at <ProjectRoot>\.venv -- the exact path the pipeline runners
    auto-detect (run_oracle_stage0_stage1.ps1, run_rl_unified_stage3456.ps1,
    run_llm789_fresh_all_single_repo.ps1 all fall back to .venv\Scripts\python.exe).
    No activation is needed afterwards; the runners pick it up automatically.

    The dependency set was derived directly from the package source (the only third-party
    modules imported are numpy, pandas, scipy, scikit-learn, joblib, PyYAML, torch) plus the
    runtime-only engines pandas needs (pyarrow for parquet; python-calamine / openpyxl for
    Excel) and the LLM SDKs (openai, anthropic) used by Stage 7. Core data-stack
    dependencies are pinned because pandas 3.x + pyarrow 24.x can crash on Windows
    during the no-regression preflight via Arrow-backed string CSV parsing.

    Reproducibility note: torch + CUDA is the main driver of RL numeric reproducibility.
    To reproduce the *reported* numbers as closely as possible, pin the exact torch build
    you originally used via -TorchVersion / -TorchIndexUrl. A requirements.lock.txt is
    written at the end so the resolved environment can be committed with the package.

.PARAMETER ProjectRoot
    Repository root. Default: auto-detected from this script's location (place this file in
    <root>\tools\). Recommended to pass it explicitly, exactly like the play-script runners.

.PARAMETER Cpu
    Install a CPU-only torch build instead of the CUDA build. Use only if you have no GPU;
    the reference runs used an RTX 4060 (CUDA).

.PARAMETER TorchVersion
    Pin a specific torch version. Default is "2.6.0", matching the final-freeze
    reproduction environment. Use "" only if you intentionally want the latest stable
    for the chosen index; exact reproduction should keep this pinned.

.PARAMETER TorchIndexUrl
    PyTorch wheel index. Default is CUDA 12.4 (works on Ada GPUs such as the RTX 4060).
    Override to match your original CUDA build, e.g. .../whl/cu121 or .../whl/cu126.
    Ignored when -Cpu is set.

.PARAMETER PythonExe
    Explicit base interpreter to build the venv from (e.g. "C:\Python311\python.exe").
    Empty (default) auto-detects via the py launcher, then `python` on PATH.

.PARAMETER RawArchivesDir
    Optional directory containing exactly raw_all.zip, raw_nonfinancial.zip, and
    rating_sample.zip. When the canonical data\raw folders are absent, the
    script also searches ProjectRoot and ProjectRoot\data\raw_archives and
    extracts these archives atomically into the canonical raw layout.

.PARAMETER Recreate
    Delete and rebuild an existing .venv.

.EXAMPLE
    # Recommended: absolute -File path + explicit root (runs from any directory,
    # mirrors how the play-script invokes every runner).
    $Root = "C:\Users\Demian\Desktop\thesis_repo_reproduction"
    powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\tools\setup_env.ps1" -ProjectRoot $Root

.EXAMPLE
    # Pin the exact build used for the reported results:
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_env.ps1 `
        -TorchVersion 2.4.1 -TorchIndexUrl https://download.pytorch.org/whl/cu124

.EXAMPLE
    # CPU-only fallback (slow; not the reference environment):
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_env.ps1 -Cpu
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [switch]$Cpu,
    [string]$TorchVersion = "2.6.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124",
    [string]$PythonExe = "",
    [string]$RawArchivesDir = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "     $msg" -ForegroundColor Gray }
function Write-Warn2($msg) { Write-Host "     WARNING: $msg" -ForegroundColor Yellow }

# ----------------------------------------------------------------------------
# 0. Resolve and sanity-check the project root
# ----------------------------------------------------------------------------
Write-Step "Resolving project root"
if ($ProjectRoot -eq "") {
    # Default: auto-detect the repo root. This script is meant to live in <root>\tools\,
    # so the root is normally the parent of the script's own directory. Also handle the
    # case where the script was placed directly in <root>.
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "src\credit_recourse")) {
        $ProjectRoot = $PSScriptRoot                       # script placed directly in <root>
    } else {
        $ProjectRoot = (Split-Path -Parent $PSScriptRoot)  # script placed in <root>\tools (expected)
    }
}
$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
Write-Info "ProjectRoot = $Root"

$pkgDir = Join-Path $Root "src\credit_recourse"
if (-not (Test-Path -LiteralPath $pkgDir)) {
    throw "Expected package not found: $pkgDir. Put src and tools directly under ProjectRoot."
}

# ----------------------------------------------------------------------------
# 0b. Materialize the canonical raw-data layout from the three distributed
#     archives when necessary. Extraction is staged and renamed only after the
#     expected top-level directory is verified, so an interrupted setup cannot
#     silently leave a half-populated canonical input directory.
# ----------------------------------------------------------------------------
Write-Step "Materializing canonical raw-data layout"
$RawRoot = Join-Path $Root "data\raw"
New-Item -ItemType Directory -Path $RawRoot -Force | Out-Null

$archiveSearchDirs = New-Object System.Collections.Generic.List[string]
if ($RawArchivesDir -ne "") {
    if (-not (Test-Path -LiteralPath $RawArchivesDir -PathType Container)) {
        throw "RawArchivesDir does not exist: $RawArchivesDir"
    }
    $archiveSearchDirs.Add((Resolve-Path -LiteralPath $RawArchivesDir).Path)
}
foreach ($candidate in @($Root, (Join-Path $Root "data\raw_archives"))) {
    if ((Test-Path -LiteralPath $candidate -PathType Container) -and -not $archiveSearchDirs.Contains($candidate)) {
        $archiveSearchDirs.Add($candidate)
    }
}

$rawSpecs = @(
    @{ Name = "raw_all"; Archive = "raw_all.zip"; MinXlsx = 6 },
    @{ Name = "raw_nonfinancial"; Archive = "raw_nonfinancial.zip"; MinXlsx = 4 },
    @{ Name = "rating_sample"; Archive = "rating_sample.zip"; MinXlsx = 1 }
)
foreach ($spec in $rawSpecs) {
    $target = Join-Path $RawRoot $spec.Name
    $existingCount = 0
    if (Test-Path -LiteralPath $target -PathType Container) {
        $existingCount = @(Get-ChildItem -LiteralPath $target -Recurse -File -Filter "*.xlsx").Count
    }
    if ($existingCount -ge [int]$spec.MinXlsx) {
        Write-Info "$($spec.Name): using existing canonical directory ($existingCount xlsx files)."
        continue
    }
    if (Test-Path -LiteralPath $target) {
        throw "Canonical raw directory exists but is incomplete: $target ($existingCount xlsx files). Remove it or restore the complete data before setup."
    }

    $hits = @()
    foreach ($dir in $archiveSearchDirs) {
        $candidate = Join-Path $dir $spec.Archive
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $hits += (Resolve-Path -LiteralPath $candidate).Path }
    }
    $hits = @($hits | Select-Object -Unique)
    if ($hits.Count -ne 1) {
        throw "Expected exactly one $($spec.Archive) because $target is absent; found $($hits.Count): $($hits -join '; ')"
    }

    $stageParent = Join-Path $Root ("data\_raw_extract_" + $spec.Name + "_" + [Guid]::NewGuid().ToString("N"))
    try {
        New-Item -ItemType Directory -Path $stageParent -Force | Out-Null
        Write-Info "$($spec.Name): extracting $($hits[0])"
        Expand-Archive -LiteralPath $hits[0] -DestinationPath $stageParent -Force
        $staged = Join-Path $stageParent $spec.Name
        if (-not (Test-Path -LiteralPath $staged -PathType Container)) {
            throw "Archive $($hits[0]) does not contain expected top-level directory '$($spec.Name)'."
        }
        $stagedCount = @(Get-ChildItem -LiteralPath $staged -Recurse -File -Filter "*.xlsx").Count
        if ($stagedCount -lt [int]$spec.MinXlsx) {
            throw "Archive $($hits[0]) yielded only $stagedCount xlsx files; expected at least $($spec.MinXlsx)."
        }
        Move-Item -LiteralPath $staged -Destination $target
        Write-Info "$($spec.Name): materialized $stagedCount xlsx files at $target"
    } finally {
        if (Test-Path -LiteralPath $stageParent) {
            Remove-Item -LiteralPath $stageParent -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# ----------------------------------------------------------------------------
# 1. Locate a base Python interpreter (3.10-3.12 recommended)
# ----------------------------------------------------------------------------
Write-Step "Locating a base Python interpreter"

function Test-PythonCandidate([string]$exe, [string[]]$prefixArgs) {
    try {
        $v = & $exe @prefixArgs -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return $v.Trim() }
    } catch { }
    return $null
}

$basePython = $null
$baseArgs   = @()

if ($PythonExe -ne "") {
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw "PythonExe '$PythonExe' not found." }
    $basePython = $PythonExe
} else {
    # Prefer the Windows py launcher pinned to a known-good minor version, then bare python.
    $cands = @(
        @{ exe = "py";     args = @("-3.12") },
        @{ exe = "py";     args = @("-3.11") },
        @{ exe = "py";     args = @("-3.10") },
        @{ exe = "python"; args = @() },
        @{ exe = "python3";args = @() }
    )
    foreach ($c in $cands) {
        if (Get-Command $c.exe -ErrorAction SilentlyContinue) {
            $ver = Test-PythonCandidate $c.exe $c.args
            if ($ver) { $basePython = $c.exe; $baseArgs = $c.args; break }
        }
    }
}

if (-not $basePython) {
    throw "No usable Python interpreter found. Install Python 3.10-3.12 and re-run, or pass -PythonExe."
}

$pyVer = Test-PythonCandidate $basePython $baseArgs
Write-Info "Base interpreter: $basePython $($baseArgs -join ' ')  ->  Python $pyVer"

$verParts = $pyVer.Split('.')
$maj = [int]$verParts[0]; $min = [int]$verParts[1]
if ($maj -ne 3 -or $min -lt 10) {
    Write-Warn2 "Python $pyVer detected. 3.10-3.12 is recommended; older versions may lack matching wheels."
}
if ($maj -eq 3 -and $min -gt 12) {
    Write-Warn2 "Python $pyVer is newer than 3.12; some wheels (esp. torch CUDA) may be unavailable. Consider 3.11/3.12."
}

# ----------------------------------------------------------------------------
# 2. Create the virtual environment at <root>\.venv (runner-detected path)
# ----------------------------------------------------------------------------
Write-Step "Creating virtual environment"
$VenvDir = Join-Path $Root ".venv"
$Py      = Join-Path $VenvDir "Scripts\python.exe"

if (Test-Path -LiteralPath $VenvDir) {
    if ($Recreate) {
        Write-Info "Removing existing .venv (-Recreate)..."
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    } else {
        Write-Warn2 ".venv already exists at '$VenvDir'. Reusing it. Pass -Recreate for a clean rebuild."
    }
}

if (-not (Test-Path -LiteralPath $Py)) {
    & $basePython @baseArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
}
if (-not (Test-Path -LiteralPath $Py)) { throw "venv python not found at '$Py' after creation." }
Write-Info "venv python = $Py"

# ----------------------------------------------------------------------------
# 3. Upgrade pip tooling
# ----------------------------------------------------------------------------
Write-Step "Upgrading pip / setuptools / wheel"
& $Py -m pip install --upgrade --disable-pip-version-check pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip tooling upgrade failed." }

# ----------------------------------------------------------------------------
# 4. Install torch first (CUDA by default), from the PyTorch wheel index
# ----------------------------------------------------------------------------
Write-Step "Installing PyTorch"
if ($Cpu) {
    $torchIndex = "https://download.pytorch.org/whl/cpu"
    Write-Info "CPU-only build (not the reference environment)."
} else {
    $torchIndex = $TorchIndexUrl
    Write-Info "CUDA build from $torchIndex"
}
$torchSpec = if ($TorchVersion -ne "") { "torch==$TorchVersion" } else { "torch" }
Write-Info "Installing $torchSpec"
& $Py -m pip install --disable-pip-version-check $torchSpec --index-url $torchIndex
if ($LASTEXITCODE -ne 0) {
    throw "torch install failed. Try a different -TorchIndexUrl (e.g. .../whl/cu121, .../whl/cu126) or -Cpu."
}

# ----------------------------------------------------------------------------
# 5. Install the remaining dependencies (from PyPI)
#    Provenance:
#      - Imported by the package : numpy, pandas, scipy, scikit-learn, joblib, PyYAML
#      - Parquet engine (pandas)  : pyarrow              (used pervasively; required)
#      - Excel engines (pandas)   : python-calamine (preferred), openpyxl (fallback/writer)
#      - Declared in backend reqs : statsmodels, tabulate, greenlet
#      - Stage 7 LLM backends     : openai, anthropic    (imported lazily at call time)
#
#    IMPORTANT FOR FINAL-FREEZE REPRODUCTION:
#      Do NOT use open-ended floors such as pandas>=2 or pyarrow>=14 here.
#      On Windows, pandas 3.x + pyarrow 24.x can segfault/access-violate during
#      pandas.read_csv() through the Arrow-backed string path. That crash prevents
#      run_oracle_stage0_stage1.ps1 from passing its no-regression preflight.
#      The pinned stack below keeps pandas on the stable 2.x object-string path.
# ----------------------------------------------------------------------------
Write-Step "Installing pinned pipeline dependencies"
$deps = @(
    "numpy==2.3.3",
    "pandas==2.3.3",
    "scipy==1.16.1",
    "scikit-learn==1.6.1",
    "joblib==1.5.3",
    "PyYAML==6.0.3",
    "pyarrow==20.0.0",
    "openpyxl==3.1.5",
    "python-calamine==0.7.0",
    "tabulate==0.10.0",
    "statsmodels==0.14.5",
    "greenlet==3.5.2",
    "openai==2.43.0",
    "anthropic==0.111.0",
    "matplotlib==3.10.6"
)
& $Py -m pip install --upgrade --disable-pip-version-check @deps
if ($LASTEXITCODE -ne 0) { throw "dependency install failed." }

# ----------------------------------------------------------------------------
# 6. Verify the environment imports cleanly and report CUDA availability
# ----------------------------------------------------------------------------
Write-Step "Verifying the environment"
$verify = @'
import importlib, sys
from io import StringIO

expected = {
    "numpy": "2.3.3",
    "pandas": "2.3.3",
    "scipy": "1.16.1",
    "sklearn": "1.6.1",
    "joblib": "1.5.3",
    "yaml": "6.0.3",
    "pyarrow": "20.0.0",
    "openpyxl": "3.1.5",
    "matplotlib": "3.10.6",
}

core = ["numpy","pandas","scipy","sklearn","joblib","yaml","pyarrow","torch","openpyxl","matplotlib"]
fail = []
for m in core:
    try:
        mod = importlib.import_module(m)
        version = getattr(mod, "__version__", "?")
        print(f"  OK    {m:16s} {version}")
        if m in expected and version != expected[m]:
            print(f"  FAIL  {m:16s} expected {expected[m]}, got {version}")
            fail.append(m)
    except Exception as e:
        print(f"  FAIL  {m:16s} {e}")
        fail.append(m)

for m, label in [("python_calamine","Excel engine"), ("openai","LLM SDK"), ("anthropic","LLM SDK")]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m:16s} {getattr(mod,'__version__','?')} ({label})")
    except Exception as e:
        print(f"  FAIL  {m:16s} not importable ({label}): {e}")
        fail.append(m)

# Regression guard for the Windows crash observed with pandas 3.x + pyarrow 24.x.
# The no-regression preflight reads a tiny CSV selected-variable master; this smoke
# test catches Arrow-backed string CSV crashes before the user reaches Oracle.
try:
    import pandas as pd
    import pyarrow as pa
    df = pd.read_csv(StringIO("variable_id\nR006\nR064\n"))
    assert df["variable_id"].tolist() == ["R006", "R064"]
    _ = pa.table({"variable_id": df["variable_id"].astype(str).tolist()})
    print("  OK    pandas_csv_smoke PANDAS_CSV_OK")
    print("  OK    pyarrow_smoke    PYARROW_STRING_OK")
except Exception as e:
    print(f"  FAIL  pandas/pyarrow smoke test failed: {e}")
    fail.append("pandas_csv_smoke")

import torch
avail = torch.cuda.is_available()
print(f"  torch {torch.__version__}  | CUDA available = {avail}")
if avail:
    print(f"  CUDA device     = {torch.cuda.get_device_name(0)}")
    print(f"  torch CUDA build= {torch.version.cuda}")
else:
    print("  (No CUDA device visible to torch. RL stages will run on CPU -- slow, and not the reference setup.)")

if fail:
    print("CORE IMPORT/SMOKE FAILURES:", fail)
    sys.exit(1)
print("ENV_OK")
'@
$verify | & $Py -
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed (see FAIL lines above)." }

# Validate the fresh-workspace and raw-input layout with the same profile used by all runners.
Write-Step "Verifying fresh-workspace and raw-input contract"
$env:PYTHONPATH = (Join-Path $Root "src")
& $Py -m credit_recourse.verification.verify_reproduction_workspace --project-root $Root
if ($LASTEXITCODE -ne 0) { throw "Fresh-workspace/raw-input contract failed." }

# ----------------------------------------------------------------------------
# 7. Freeze a lockfile for reproducibility
# ----------------------------------------------------------------------------
Write-Step "Writing requirements.lock.txt"
$lock = Join-Path $Root "requirements.lock.txt"
& $Py -m pip freeze | Out-File -FilePath $lock -Encoding utf8
Write-Info "Wrote $lock  (commit this with the reproduction package)"

# ----------------------------------------------------------------------------
# 8. Done
# ----------------------------------------------------------------------------
Write-Step "Setup complete"
Write-Host @"
The virtual environment is ready at:
    $VenvDir
The pipeline runners detect it automatically (no activation needed).

Run the canonical workflow:
  1. Set your API keys in that shell session (used by the LLM stages):
         `$env:OPENAI_API_KEY    = "<your OpenAI key>"
         `$env:ANTHROPIC_API_KEY = "<your Anthropic key>"
  2. Ensure raw data is in place under $Root\data\raw\
         (raw_all\, raw_nonfinancial\{kospi_kosdaq,konex_optional}\, and rating_sample\).
  3. Run tools\run_thesis_repro.ps1 in order: Oracle -> RL -> LLM -> Analysis.

To use this environment manually instead:
    & "$VenvDir\Scripts\Activate.ps1"
"@ -ForegroundColor Green
