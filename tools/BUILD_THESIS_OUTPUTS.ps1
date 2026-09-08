[CmdletBinding()]
param(
    [string]$RunId,
    [string]$ThesisDocx,
    [string]$ProjectRoot,
    [string]$PythonExe,
    [string]$PythonSitePackagesRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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

function Read-RunInfo([string]$Path) {
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        if ($line -notmatch '=') { continue }
        $parts = $line.Split('=', 2)
        $values[$parts[0].Trim()] = $parts[1].Trim()
    }
    return $values
}

function Select-CompletedRun([string]$Root, [string]$Requested) {
    $runsRoot = Join-Path $Root 'data\runs'
    if ($Requested) {
        $infoPath = Join-Path $runsRoot "$Requested\RUN_INFO.txt"
        if (-not (Test-Path -LiteralPath $infoPath -PathType Leaf)) {
            throw "실행 기록을 찾지 못했습니다: $infoPath"
        }
        $info = Read-RunInfo $infoPath
        if ([string]$info.exit_status -ne '0') { throw "완료되지 않은 실행입니다: $Requested" }
        return $Requested
    }
    foreach ($file in Get-ChildItem -LiteralPath $runsRoot -Filter RUN_INFO.txt -Recurse -File | Sort-Object LastWriteTime -Descending) {
        $info = Read-RunInfo $file.FullName
        if ([string]$info.exit_status -eq '0') {
            if ($info.run_id) { return [string]$info.run_id }
            return $file.Directory.Name
        }
    }
    throw '완료된 실행이 없습니다. 먼저 RUN_REPRODUCTION.ps1을 실행하세요.'
}

function Move-PreviousOutputsAside([string]$OutputRoot, [string]$RunRoot) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    $children = @(Get-ChildItem -LiteralPath $OutputRoot -Force)
    if ($children.Count -eq 0) { return }
    $saved = Join-Path $RunRoot ('previous_thesis_outputs_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    New-Item -ItemType Directory -Path $saved -Force | Out-Null
    foreach ($child in $children) {
        Move-Item -LiteralPath $child.FullName -Destination $saved
    }
    Write-Host "기존 thesis_outputs는 보관했습니다: $saved" -ForegroundColor DarkGray
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { throw '스크립트 위치에서 저장소 루트를 결정하지 못했습니다. -ProjectRoot를 지정하세요.' }
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) { throw "저장소 루트를 찾지 못했습니다: $ProjectRoot" }
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$RunId = Select-CompletedRun $ProjectRoot $RunId
$runRoot = Join-Path $ProjectRoot "data\runs\$RunId"
$outputRoot = Join-Path $ProjectRoot 'data\thesis_outputs'

if (-not $ThesisDocx) { $ThesisDocx = Join-Path $ProjectRoot 'docs\thesis\canonical_thesis.docx' }
if (-not (Test-Path -LiteralPath $ThesisDocx -PathType Leaf)) {
    throw "논문 DOCX를 찾지 못했습니다: $ThesisDocx"
}
$ThesisDocx = (Resolve-Path -LiteralPath $ThesisDocx).Path

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
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonPaths = @((Join-Path $ProjectRoot 'src'))
$pythonPaths += $PythonSitePackagesRoot
$env:PYTHONPATH = $pythonPaths -join [System.IO.Path]::PathSeparator

Move-PreviousOutputsAside $outputRoot $runRoot

Write-Host ''
Write-Host "논문 DOCX에서 항목을 읽고 실제 실행 산출물을 계산합니다: $RunId" -ForegroundColor Cyan
& $PythonExe -m credit_recourse.reproduction.thesis_outputs.prepare `
    --project-root $ProjectRoot `
    --thesis-docx $ThesisDocx `
    --run-id $RunId `
    --output-root $outputRoot
if ($LASTEXITCODE -ne 0) { throw "논문 산출물 계산 준비 실패 (exit=$LASTEXITCODE)" }

$payload = Join-Path $outputRoot "_build\$RunId\build_payload.json"
if (-not (Test-Path -LiteralPath $payload -PathType Leaf)) { throw "계산 결과를 찾지 못했습니다: $payload" }
$builder = Join-Path $ProjectRoot 'src\credit_recourse\reproduction\thesis_outputs\build_workbooks.py'
& $PythonExe $builder --payload $payload --output-root $outputRoot
if ($LASTEXITCODE -ne 0) { throw "Excel 생성 실패 (exit=$LASTEXITCODE)" }

foreach ($required in @('THESIS_OUTPUT_INDEX.xlsx','NUMERIC_CLAIMS.xlsx')) {
    $path = Join-Path $outputRoot $required
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 Excel이 생성되지 않았습니다: $path" }
}

$buildRoot = Join-Path $outputRoot '_build'
if (Test-Path -LiteralPath $buildRoot -PathType Container) {
    $resolvedOutput = [System.IO.Path]::GetFullPath($outputRoot).TrimEnd('\') + '\'
    $resolvedBuild = [System.IO.Path]::GetFullPath($buildRoot)
    if (-not $resolvedBuild.StartsWith($resolvedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "임시 계산 경로가 thesis_outputs 밖입니다: $resolvedBuild"
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}

$tableCount = @(Get-ChildItem -LiteralPath (Join-Path $outputRoot 'tables') -Filter '*.xlsx' -File -ErrorAction SilentlyContinue).Count
$figureCount = @(Get-ChildItem -LiteralPath (Join-Path $outputRoot 'figures') -Filter '*.xlsx' -File -ErrorAction SilentlyContinue).Count
if ($tableCount -ne 57 -or $figureCount -ne 14) {
    throw "논문 DOCX 전체 항목이 생성되지 않았습니다: 표 $tableCount/57, 그림 $figureCount/14"
}
Write-Host ''
Write-Host "완료: 표 ${tableCount}개, 그림 ${figureCount}개, 논문 수치 요약 2개" -ForegroundColor Green
Write-Host "산출물: $outputRoot"
exit 0
