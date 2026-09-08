param(
    [string]$BundleRoot,
    [string]$SearchRoot,
    [switch]$Overwrite,
    [switch]$CopyFiles
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}

if ([string]::IsNullOrWhiteSpace($BundleRoot)) {
    $BundleRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $BundleRoot = (Resolve-Path $BundleRoot).Path
}

if ([string]::IsNullOrWhiteSpace($SearchRoot)) {
    $SearchRoot = (Resolve-Path (Join-Path $BundleRoot "..")).Path
} else {
    $SearchRoot = (Resolve-Path $SearchRoot).Path
}

Set-Location $BundleRoot
New-Item -ItemType Directory -Force -Path ".\inputs" | Out-Null

function Show-Header([string]$Text) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Is-BadCandidate([System.IO.FileInfo]$File) {
    $p = $File.FullName.ToLowerInvariant()
    $badFragments = @(
        "\outputs_pareto\",
        "\logs\",
        "\.venv\",
        "\__pycache__\",
        "__backup",
        "backup_before",
        "backup",
        "\archive\",
        "\old\",
        "\tmp\",
        "\temp\"
    )
    foreach ($frag in $badFragments) {
        if ($p.Contains($frag.ToLowerInvariant())) { return $true }
    }
    if ($File.Length -le 0) { return $true }
    return $false
}

function Get-DirScore([string]$PathText, [string[]]$PreferFragments) {
    $pl = $PathText.ToLowerInvariant()
    $score = 0
    foreach ($frag in $PreferFragments) {
        if (-not [string]::IsNullOrWhiteSpace($frag) -and $pl.Contains($frag.ToLowerInvariant())) { $score += 1000 }
    }
    if ($pl.Contains("\outputs\")) { $score += 500 }
    if ($pl.Contains("\05_outputs\")) { $score += 500 }
    if ($pl.Contains("\inputs\stage1b")) { $score += 300 }
    if ($pl.Contains("\inputs\stage2")) { $score += 300 }
    if ($pl.Contains("\inputs\stage1c")) { $score += 300 }
    if ($pl.Contains("\inputs\stage3")) { $score += 300 }
    if ($pl.Contains("\stage3_v2")) { $score += 250 }
    if ($pl.Contains("\stage2_v3")) { $score += 80 }
    if ($pl.Contains("\stage1c_v3")) { $score += 80 }
    if ($pl.Contains("\stage3_v3")) { $score += 60 }
    if ($pl.Contains("backup")) { $score -= 5000 }
    return $score
}

function Find-BestFile([string]$Label, [string]$FileName, [string[]]$PreferFragments) {
    Write-Host "[SEARCH] $Label :: $FileName under $SearchRoot" -ForegroundColor Cyan

    $all = @(Get-ChildItem -LiteralPath $SearchRoot -Recurse -File -Filter $FileName -ErrorAction SilentlyContinue)
    $candidates = @($all | Where-Object { -not (Is-BadCandidate $_) })

    if (-not $candidates -or $candidates.Count -eq 0) {
        Write-Host "  No clean candidate found. Raw candidates:" -ForegroundColor Yellow
        @($all | Select-Object -First 20) | ForEach-Object { Write-Host "    - $($_.FullName)" -ForegroundColor DarkGray }
        throw "Required input not found: $Label ($FileName). SearchRoot=$SearchRoot"
    }

    $ranked = $candidates | ForEach-Object {
        $score = Get-DirScore $_.FullName $PreferFragments
        [PSCustomObject]@{ File = $_; Score = $score }
    } | Sort-Object @{Expression="Score";Descending=$true}, @{Expression={$_.File.LastWriteTime};Descending=$true}, @{Expression={$_.File.Length};Descending=$true}

    $selected = $ranked[0].File
    Write-Host "  Selected file: $($selected.FullName)" -ForegroundColor Green
    Write-Host "  Source dir   : $($selected.Directory.FullName)" -ForegroundColor Green
    Write-Host "  SizeMB       : $([Math]::Round($selected.Length / 1MB, 2)) | LastWrite: $($selected.LastWriteTime)"

    if ($ranked.Count -gt 1) {
        Write-Host "  Other clean candidates:" -ForegroundColor DarkGray
        $ranked | Select-Object -Skip 1 -First 6 | ForEach-Object {
            Write-Host "    - score=$($_.Score) $($_.File.FullName)" -ForegroundColor DarkGray
        }
    }
    return $selected
}

function Remove-PathSafely([string]$PathText) {
    if (-not (Test-Path -LiteralPath $PathText)) { return }
    $item = Get-Item -LiteralPath $PathText -Force
    Write-Host "[REMOVE] $PathText" -ForegroundColor Yellow
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        # Directory junction/symlink: rmdir removes link, not target.
        cmd /c rmdir "$PathText"
    } elseif ($item.PSIsContainer) {
        Remove-Item -LiteralPath $PathText -Recurse -Force
    } else {
        Remove-Item -LiteralPath $PathText -Force
    }
}

function Link-DirectoryOrCopyFile([System.IO.FileInfo]$SourceFile, [string]$TargetSubdir, [string]$RequiredFileName, [string]$Label) {
    $srcDir = $SourceFile.Directory.FullName
    $targetDir = Join-Path $BundleRoot (Join-Path "inputs" $TargetSubdir)
    $targetFile = Join-Path $targetDir $RequiredFileName

    if ((Test-Path -LiteralPath $targetFile) -and (-not $Overwrite)) {
        $existing = Get-Item -LiteralPath $targetFile
        if ($existing.Length -gt 0) {
            Write-Host "[OK] $Label already available: $targetFile ($([Math]::Round($existing.Length/1MB,2)) MB)" -ForegroundColor Green
            return
        }
    }

    if ($CopyFiles) {
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        Write-Host "[COPY] $Label" -ForegroundColor Cyan
        Write-Host "  From: $($SourceFile.FullName)"
        Write-Host "  To  : $targetFile"
        Copy-Item -LiteralPath $SourceFile.FullName -Destination $targetFile -Force -ErrorAction Stop
    } else {
        # Default: link whole input directory using junction. This avoids Copy-Item failures on Windows paths.
        if (Test-Path -LiteralPath $targetDir) {
            if ($Overwrite) {
                Remove-PathSafely $targetDir
            } else {
                # Existing dir without target file: do not merge silently; make user choose overwrite.
                throw "Target dir exists but required file missing or invalid: $targetDir. Re-run with -Overwrite or inspect manually."
            }
        }
        $parent = Split-Path -Parent $targetDir
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Write-Host "[LINK] $Label" -ForegroundColor Cyan
        Write-Host "  Junction: $targetDir"
        Write-Host "  Target  : $srcDir"
        New-Item -ItemType Junction -Path $targetDir -Target $srcDir | Out-Null
    }

    if (-not (Test-Path -LiteralPath $targetFile)) {
        throw "Required file still not visible after repair: $targetFile"
    }
    $item = Get-Item -LiteralPath $targetFile
    if ($item.Length -le 0) {
        throw "Required file has zero length after repair: $targetFile"
    }
    Write-Host "[PASS] $Label -> $targetFile ($([Math]::Round($item.Length/1MB,2)) MB)" -ForegroundColor Green
}

function Repair-RequiredInput([string]$Label, [string]$FileName, [string]$TargetSubdir, [string[]]$PreferFragments) {
    $targetFile = Join-Path $BundleRoot (Join-Path "inputs" (Join-Path $TargetSubdir $FileName))
    if ((Test-Path -LiteralPath $targetFile) -and (-not $Overwrite)) {
        $existing = Get-Item -LiteralPath $targetFile
        if ($existing.Length -gt 0) {
            Write-Host "[OK] $Label already exists: $targetFile ($([Math]::Round($existing.Length/1MB,2)) MB)" -ForegroundColor Green
            return
        }
    }
    $srcFile = Find-BestFile $Label $FileName $PreferFragments
    Link-DirectoryOrCopyFile $srcFile $TargetSubdir $FileName $Label
}

Show-Header "Repair Stage 4 alpha input bundle (v8 - junction-first)"
Write-Host "BundleRoot: $BundleRoot"
Write-Host "SearchRoot: $SearchRoot"
Write-Host "Overwrite : $Overwrite"
Write-Host "Mode      : $($(if ($CopyFiles) { 'copy files' } else { 'junction directories' }))"

Repair-RequiredInput "Stage 1B firm_year_panel_v1.parquet" "firm_year_panel_v1.parquet" "stage1b" @("stage2_v3_code_reproduction_bundle_fixed\inputs\stage1b", "stage1b_v", "stage1b")
Repair-RequiredInput "Stage 2 engineered_financial_ratios.parquet" "engineered_financial_ratios.parquet" "stage2" @("stage2_v3", "stage2", "outputs")
Repair-RequiredInput "Stage 1C nonfinancial_metadata_panel.parquet" "nonfinancial_metadata_panel.parquet" "stage1c_v3" @("stage1c_v3", "stage1c", "outputs")
Repair-RequiredInput "Stage 3 selected_variables_v2.json" "selected_variables_v2.json" "stage3_v2" @("stage3_v3", "stage3_v2", "stage3", "outputs")
Repair-RequiredInput "Stage 3 direction_encoding_v2.json" "direction_encoding_v2.json" "stage3_v2" @("stage3_v3", "stage3_v2", "stage3", "outputs")

Show-Header "Post-repair check"
$required = @(
    ".\inputs\stage1b\firm_year_panel_v1.parquet",
    ".\inputs\stage2\engineered_financial_ratios.parquet",
    ".\inputs\stage1c_v3\nonfinancial_metadata_panel.parquet",
    ".\inputs\stage3_v2\selected_variables_v2.json",
    ".\inputs\stage3_v2\direction_encoding_v2.json"
)
$missing = @()
foreach ($p in $required) {
    if (Test-Path -LiteralPath $p) {
        $item = Get-Item -LiteralPath $p
        Write-Host "[PASS] $p ($([Math]::Round($item.Length/1MB,2)) MB)" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $p" -ForegroundColor Red
        $missing += $p
    }
}
if ($missing.Count -gt 0) {
    throw "Input repair incomplete. Missing: $($missing -join ', ')"
}
Write-Host "[DONE] Stage 4 input files are ready." -ForegroundColor Green
