Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-ReproRoot {
  param([string]$ProjectRoot = '')
  if (-not [string]::IsNullOrWhiteSpace($ProjectRoot)) {
    return (Resolve-Path -LiteralPath $ProjectRoot).Path
  }
  $candidate = Split-Path -Parent $PSScriptRoot
  if (-not (Test-Path -LiteralPath (Join-Path $candidate 'src\credit_recourse') -PathType Container)) {
    throw "Cannot infer repro repo root. Pass -ProjectRoot."
  }
  return (Resolve-Path -LiteralPath $candidate).Path
}

function Get-ReproPython {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$PythonExe=''
  )
  $py = if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    Join-Path $Root '.venv\Scripts\python.exe'
  }
  else {
    [System.IO.Path]::GetFullPath($PythonExe)
  }
  if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "Python executable missing: $py"
  }
  return (Resolve-Path -LiteralPath $py).Path
}

function ConvertFrom-ReproProcessText {
  param([AllowEmptyString()][string]$Text)

  if ([string]::IsNullOrEmpty($Text)) { return @() }
  $trimmed = $Text.TrimEnd("`r", "`n")
  if ([string]::IsNullOrEmpty($trimmed)) { return @() }
  return @($trimmed -split "`r?`n")
}

function Invoke-ReproPythonStdinCaptured {
  param(
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][AllowEmptyString()][string]$ScriptText
  )

  if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable missing: $PythonExe"
  }

  # Windows PowerShell 5.1 reconstructs one native command-line string and can
  # consume embedded double quotes in a Python -c payload. Keep Python source
  # out of argv entirely and write UTF-8 directly to redirected stdin.
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $PythonExe
  $startInfo.Arguments = '-B -'
  $startInfo.UseShellExecute = $false
  $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardInput = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $startInfo.StandardOutputEncoding = $utf8
  $startInfo.StandardErrorEncoding = $utf8

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) { throw "Failed to start Python process: $PythonExe" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $stdinText = if ($ScriptText.EndsWith("`n")) { $ScriptText } else { $ScriptText + "`n" }
    $stdinBytes = $utf8.GetBytes($stdinText)
    $process.StandardInput.BaseStream.Write($stdinBytes, 0, $stdinBytes.Length)
    $process.StandardInput.BaseStream.Flush()
    $process.StandardInput.BaseStream.Close()
    $process.WaitForExit()
    $stdoutText = $stdoutTask.GetAwaiter().GetResult()
    $stderrText = $stderrTask.GetAwaiter().GetResult()
    $exitCode = [int]$process.ExitCode
  }
  finally {
    $process.Dispose()
  }

  $stdoutLines = @(ConvertFrom-ReproProcessText -Text $stdoutText)
  $stderrLines = @(ConvertFrom-ReproProcessText -Text $stderrText)
  $outputLines = @($stdoutLines) + @($stderrLines)
  $firstStderr = if ($stderrLines.Count -gt 0) { $stderrLines[0] } else { $null }

  return [pscustomobject]@{
    ExitCode = $exitCode
    Output = $outputLines
    StandardOutput = $stdoutLines
    StandardError = $stderrLines
    FirstStderr = $firstStderr
  }
}

function Initialize-ReproPythonEnvironment {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [string]$PythonSourceRoot='',
    [string]$PythonSitePackagesRoot='',
    [switch]$VerifyImport
  )

  $sourceRoot = if ([string]::IsNullOrWhiteSpace($PythonSourceRoot)) {
    Join-Path $Root 'src'
  }
  else {
    [System.IO.Path]::GetFullPath($PythonSourceRoot)
  }
  if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Repository Python source root is missing: $sourceRoot"
  }
  $sourceRoot = (Resolve-Path -LiteralPath $sourceRoot).Path
  $packageRoot = Join-Path $sourceRoot 'credit_recourse'
  if (-not (Test-Path -LiteralPath $packageRoot -PathType Container)) {
    throw "Repository Python package is missing: $packageRoot"
  }
  if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Repository Python executable is missing: $PythonExe"
  }

  $separator = [System.IO.Path]::PathSeparator
  $existing = @()
  if (-not [string]::IsNullOrWhiteSpace($PythonSitePackagesRoot)) {
    if (-not (Test-Path -LiteralPath $PythonSitePackagesRoot -PathType Container)) {
      throw "Python site-packages root is missing: $PythonSitePackagesRoot"
    }
    $existing = @((Resolve-Path -LiteralPath $PythonSitePackagesRoot).Path)
  }
  elseif (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $existing = @(
      $env:PYTHONPATH.Split(
        [char[]]@($separator),
        [System.StringSplitOptions]::RemoveEmptyEntries
      )
    )
  }

  $existing = @(
    $existing | Where-Object {
      -not [string]::Equals(
        $_.TrimEnd('\','/'),
        $sourceRoot.TrimEnd('\','/'),
        [System.StringComparison]::OrdinalIgnoreCase
      )
    }
  )
  if ($existing.Count -eq 0) {
    $env:PYTHONPATH = $sourceRoot
  }
  else {
    $env:PYTHONPATH = $sourceRoot + $separator + ($existing -join $separator)
  }

  $env:PYTHONUTF8 = '1'
  $env:PYTHONIOENCODING = 'utf-8'
  $env:PYTHONDONTWRITEBYTECODE = '1'

  if ($VerifyImport) {
    $previousExpectedSource = [Environment]::GetEnvironmentVariable(
      'CREDIT_RECOURSE_EXPECTED_SOURCE_ROOT',
      'Process'
    )
    $env:CREDIT_RECOURSE_EXPECTED_SOURCE_ROOT = $sourceRoot
    $bootstrapScript = @'
import os
import pathlib

import credit_recourse

actual = pathlib.Path(credit_recourse.__file__).resolve()
expected = pathlib.Path(os.environ["CREDIT_RECOURSE_EXPECTED_SOURCE_ROOT"]).resolve()
assert expected in actual.parents, f"import escaped source root: {actual} != {expected}"
print("credit_recourse import: PASS", actual)
'@
    try {
      $bootstrapResult = Invoke-ReproPythonStdinCaptured `
        -PythonExe $PythonExe `
        -ScriptText $bootstrapScript
    }
    finally {
      if ($null -eq $previousExpectedSource) {
        [Environment]::SetEnvironmentVariable(
          'CREDIT_RECOURSE_EXPECTED_SOURCE_ROOT',
          $null,
          'Process'
        )
      }
      else {
        $env:CREDIT_RECOURSE_EXPECTED_SOURCE_ROOT = $previousExpectedSource
      }
    }
    $bootstrapResult.Output | ForEach-Object { Write-Host $_ }
    if ($bootstrapResult.ExitCode -ne 0) {
      throw (
        "Repository Python bootstrap failed exit=$($bootstrapResult.ExitCode); " +
        "python=$PythonExe; source=$sourceRoot"
      )
    }
  }
}

function Initialize-ReproFrozenPythonEnvironment {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$PythonSourceRoot,
    [Parameter(Mandatory=$true)][string]$PythonSitePackagesRoot,
    [Parameter(Mandatory=$true)][string]$PythonEnvironmentConfig,
    [Parameter(Mandatory=$true)][string]$PythonEnvironmentContractId,
    [Parameter(Mandatory=$true)][string]$ExecutionRunRoot,
    [Parameter(Mandatory=$true)][string]$TemporaryRoot
  )

  $requiredValues = [ordered]@{
    PythonSourceRoot = $PythonSourceRoot
    PythonSitePackagesRoot = $PythonSitePackagesRoot
    PythonEnvironmentConfig = $PythonEnvironmentConfig
    PythonEnvironmentContractId = $PythonEnvironmentContractId
    ExecutionRunRoot = $ExecutionRunRoot
    TemporaryRoot = $TemporaryRoot
  }
  foreach ($requiredName in $requiredValues.Keys) {
    if ([string]::IsNullOrWhiteSpace([string]$requiredValues[$requiredName])) {
      throw "Frozen child environment requires explicit $requiredName."
    }
  }

  $resolvedRunRoot = [System.IO.Path]::GetFullPath($ExecutionRunRoot).TrimEnd('\','/')
  $resolvedTemporaryRoot = [System.IO.Path]::GetFullPath($TemporaryRoot).TrimEnd('\','/')
  $runPrefix = $resolvedRunRoot + [System.IO.Path]::DirectorySeparatorChar
  if (-not $resolvedTemporaryRoot.StartsWith(
    $runPrefix,
    [System.StringComparison]::OrdinalIgnoreCase
  )) {
    throw "Frozen child temporary root escapes RunRoot: $resolvedTemporaryRoot"
  }
  if (-not (Test-Path -LiteralPath $resolvedRunRoot -PathType Container)) {
    throw "Frozen child RunRoot is missing: $resolvedRunRoot"
  }
  if (-not (Test-Path -LiteralPath $resolvedTemporaryRoot -PathType Container)) {
    throw "Frozen child temporary root is missing: $resolvedTemporaryRoot"
  }
  $resolvedEnvironmentConfig = [System.IO.Path]::GetFullPath($PythonEnvironmentConfig)

  Initialize-ReproPythonEnvironment `
    -Root $Root `
    -PythonExe $PythonExe `
    -PythonSourceRoot $PythonSourceRoot `
    -PythonSitePackagesRoot $PythonSitePackagesRoot `
    -VerifyImport

  $env:TEMP = $resolvedTemporaryRoot
  $env:TMP = $resolvedTemporaryRoot
  $env:MPLCONFIGDIR = Join-Path $resolvedTemporaryRoot 'matplotlib'
  $env:NUMBA_CACHE_DIR = Join-Path $resolvedTemporaryRoot 'numba'
  $env:XDG_CACHE_HOME = Join-Path $resolvedTemporaryRoot 'xdg'
  $env:TORCH_HOME = Join-Path $resolvedTemporaryRoot 'torch'
  $env:HF_HOME = Join-Path $resolvedTemporaryRoot 'huggingface'
  $env:JOBLIB_TEMP_FOLDER = Join-Path $resolvedTemporaryRoot 'joblib'
  $env:CREDIT_RECOURSE_PYTHON_SOURCE_ROOT = (Resolve-Path -LiteralPath $PythonSourceRoot).Path
  $env:CREDIT_RECOURSE_PYTHON_SITE_PACKAGES_ROOT = (
    Resolve-Path -LiteralPath $PythonSitePackagesRoot
  ).Path
  $env:CREDIT_RECOURSE_PYTHON_ENVIRONMENT_CONFIG = $resolvedEnvironmentConfig
  $env:CREDIT_RECOURSE_PYTHON_ENVIRONMENT_CONTRACT_ID = $PythonEnvironmentContractId
  $env:CREDIT_RECOURSE_EXECUTION_RUN_ROOT = $resolvedRunRoot
  $env:CREDIT_RECOURSE_TEMPORARY_ROOT = $resolvedTemporaryRoot

}

function Assert-ReproId {
  param([Parameter(Mandatory=$true)][string]$Value,[string]$Label='ID')
  if ($Value -notmatch '^[A-Za-z][A-Za-z0-9_-]{1,31}$') {
    throw "Invalid $Label '$Value'. Use 2-32 characters: leading letter, then letters, digits, _ or -."
  }
}

function Assert-NewPath {
  param([Parameter(Mandatory=$true)][string]$Path,[string]$Label='Output')
  if (Test-Path -LiteralPath $Path) {
    throw "$Label already exists and will not be overwritten: $Path"
  }
}

function Invoke-ReproChecked {
  param([Parameter(Mandatory=$true)][string]$Label,[Parameter(Mandatory=$true)][string[]]$Command)
  Write-Host "`n==== $Label ====" -ForegroundColor Cyan
  Write-Host "CMD> $($Command -join ' ')" -ForegroundColor DarkGray
  if ($Command.Count -lt 1) { throw "Empty command for: $Label" }
  $exe = $Command[0]
  $invokeArgs = @()
  if ($Command.Count -gt 1) { $invokeArgs = @($Command[1..($Command.Count - 1)]) }
  & $exe @invokeArgs
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) { throw "FAILED: $Label exit=$exitCode" }
}

function New-ReproJunction {
  param(
    [Parameter(Mandatory=$true)][string]$Link,
    [Parameter(Mandatory=$true)][string]$Target,
    [Parameter(Mandatory=$true)][string]$CompatRoot
  )
  if (-not (Test-Path -LiteralPath $Target -PathType Container)) { throw "Junction target missing: $Target" }
  if (Test-Path -LiteralPath $Link) { throw "Junction link already exists: $Link" }
  $parent = Split-Path -Parent $Link
  New-Item -ItemType Directory -Path $parent -Force | Out-Null
  New-Item -ItemType Junction -Path $Link -Target $Target | Out-Null
  Add-Content -LiteralPath (Join-Path $CompatRoot '.junctions.txt') -Value $Link -Encoding UTF8
}

function Set-ReproTreeWritable {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [string]$Label='Generated output tree'
  )
  if (-not (Test-Path -LiteralPath $Path)) { return }

  $entries = @(
    Get-Item -LiteralPath $Path -Force
  ) + @(
    Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction Stop
  )

  foreach ($entry in $entries) {
    if (($entry.Attributes -band [System.IO.FileAttributes]::ReadOnly) -ne 0) {
      $entry.Attributes = (
        $entry.Attributes -bxor [System.IO.FileAttributes]::ReadOnly
      )
    }
  }

  $remaining = @(
    Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction Stop |
      Where-Object {
        ($_.Attributes -band [System.IO.FileAttributes]::ReadOnly) -ne 0
      }
  )
  if ($remaining.Count -gt 0) {
    throw (
      "${Label} still contains read-only entries: {0}"
    ) -f (($remaining | Select-Object -First 10 -ExpandProperty FullName) -join '; ')
  }
}

function Assert-ReproWritableDirectory {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [string]$Label='Writable directory'
  )
  New-Item -ItemType Directory -Path $Path -Force | Out-Null
  Set-ReproTreeWritable -Path $Path -Label $Label

  $probe = Join-Path $Path ('.repro_write_probe_{0}.tmp' -f $PID)
  try {
    [System.IO.File]::WriteAllText(
      $probe,
      'PASS',
      [System.Text.UTF8Encoding]::new($false)
    )
    [System.IO.File]::WriteAllText(
      $probe,
      'PASS_OVERWRITE',
      [System.Text.UTF8Encoding]::new($false)
    )
  }
  finally {
    if (Test-Path -LiteralPath $probe -PathType Leaf) {
      Remove-Item -LiteralPath $probe -Force
    }
  }
}

function Copy-ReproDirectoryWritable {
  param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$Label='Writable overlay'
  )
  if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
    throw "${Label} source directory is missing: $Source"
  }
  if (Test-Path -LiteralPath $Destination) {
    throw "${Label} destination already exists: $Destination"
  }
  New-Item -ItemType Directory -Path $Destination -Force | Out-Null

  $robocopy = Get-Command robocopy.exe -ErrorAction SilentlyContinue
  if ($null -ne $robocopy) {
    $previousPreference = $ErrorActionPreference
    try {
      $ErrorActionPreference = 'Continue'
      & $robocopy.Source `
        $Source `
        $Destination `
        /E /COPY:DT /DCOPY:DT /R:2 /W:1 /XJ /NFL /NDL /NJH /NJS /NP |
        Out-Null
      $exitCode = $LASTEXITCODE
    }
    finally {
      $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -gt 7) {
      throw "${Label} copy failed exit=$exitCode; source=$Source; destination=$Destination"
    }
  }
  else {
    Get-ChildItem -LiteralPath $Source -Force -ErrorAction Stop |
      ForEach-Object {
        Copy-Item -LiteralPath $_.FullName `
          -Destination $Destination -Recurse -Force -ErrorAction Stop
      }
  }

  Set-ReproTreeWritable -Path $Destination -Label $Label
  Assert-ReproWritableDirectory -Path $Destination -Label $Label
}

function Remove-ReproCompatView {
  param([Parameter(Mandatory=$true)][string]$CompatRoot)
  $ledger = Join-Path $CompatRoot '.junctions.txt'
  $errors = New-Object System.Collections.Generic.List[string]
  if (Test-Path -LiteralPath $ledger -PathType Leaf) {
    $links = @(
      Get-Content -LiteralPath $ledger -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    [array]::Reverse($links)
    foreach ($link in $links) {
      if (-not (Test-Path -LiteralPath $link)) { continue }
      try {
        $item = Get-Item -LiteralPath $link -Force
        $isReparse = (
          ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
        )
        if (-not $isReparse) {
          throw "Tracked compatibility path is not a junction/reparse point: $link"
        }
        & cmd.exe /d /c "rmdir `"$link`"" | Out-Null
        if ($LASTEXITCODE -ne 0 -or (Test-Path -LiteralPath $link)) {
          throw "Junction removal failed exit=${LASTEXITCODE}: $link"
        }
      }
      catch {
        $errors.Add($_.Exception.Message)
      }
    }
  }

  if (Test-Path -LiteralPath $CompatRoot) {
    try {
      Set-ReproTreeWritable -Path $CompatRoot -Label 'Compatibility view cleanup'
      Remove-Item -LiteralPath $CompatRoot -Recurse -Force
    }
    catch {
      $errors.Add($_.Exception.Message)
    }
  }
  if ($errors.Count -gt 0) {
    throw "Compatibility view cleanup failed: $($errors -join ' | ')"
  }
}

function Initialize-ReproCompatBase {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$CompatRoot,
    [Parameter(Mandatory=$true)][string]$AnalysisTarget,
    [Parameter(Mandatory=$true)][string]$ReproductionTarget,
    [switch]$DisableRawProjection
  )
  Assert-NewPath -Path $CompatRoot -Label 'Compatibility view'
  New-Item -ItemType Directory -Path $CompatRoot -Force | Out-Null
  Set-Content -LiteralPath (Join-Path $CompatRoot '.junctions.txt') -Value '' -Encoding UTF8
  New-ReproJunction -Link (Join-Path $CompatRoot 'src') -Target (Join-Path $Root 'src') -CompatRoot $CompatRoot
  New-ReproJunction -Link (Join-Path $CompatRoot 'tools') -Target (Join-Path $Root 'tools') -CompatRoot $CompatRoot
  # Python is supplied explicitly by the canonical runner.  Do not create a
  # .venv junction: an external interpreter remains read-only and provenance is
  # recorded by the run configuration.
  New-Item -ItemType Directory -Path (Join-Path $CompatRoot 'data') -Force | Out-Null
  if (-not $DisableRawProjection) {
    foreach ($dataName in @('raw','raw_all','raw_nonfinancial','rating_sample','financial_simulator')) {
      $dataTarget = Join-Path $Root "data\$dataName"
      if (Test-Path -LiteralPath $dataTarget -PathType Container) {
        New-ReproJunction -Link (Join-Path $CompatRoot "data\$dataName") -Target $dataTarget -CompatRoot $CompatRoot
      }
    }
  }
  # Frozen Replay disables legacy raw projection and passes an explicit
  # read-only RawRoot to the sole post-freeze consumer.
  if (Test-Path -LiteralPath (Join-Path $Root 'financial_simulator') -PathType Container) {
    New-ReproJunction -Link (Join-Path $CompatRoot 'financial_simulator') -Target (Join-Path $Root 'financial_simulator') -CompatRoot $CompatRoot
  }
  New-Item -ItemType Directory -Path $AnalysisTarget -Force | Out-Null
  New-Item -ItemType Directory -Path $ReproductionTarget -Force | Out-Null
  Set-ReproTreeWritable -Path $AnalysisTarget -Label 'Run-local analysis tree'
  Set-ReproTreeWritable -Path $ReproductionTarget -Label 'Run-local reproduction tree'
  Assert-ReproWritableDirectory -Path $AnalysisTarget -Label 'Run-local analysis tree'
  Assert-ReproWritableDirectory -Path $ReproductionTarget -Label 'Run-local reproduction tree'
  New-ReproJunction -Link (Join-Path $CompatRoot 'data\analysis') -Target $AnalysisTarget -CompatRoot $CompatRoot
  New-ReproJunction -Link (Join-Path $CompatRoot 'data\reproduction') -Target $ReproductionTarget -CompatRoot $CompatRoot
}

function Add-ReferenceArtifactView {
  param(
    [Parameter(Mandatory=$true)][string]$CompatRoot,
    [Parameter(Mandatory=$true)][string]$FrozenArtifactRoot,
    [Parameter(Mandatory=$true)][string]$ReferenceRoot,
    [string[]]$StageNames = @(
      'stage0_oracle_foundation','stage1_oracle_inputs','stage1_oracle_backends',
      'stage2_candidate_projection','stage2_substrate_loopA_loopB2','stage2_substrate_validation',
      'stage3_acd_ssl','stage4_candidate_bc','stage5_candidate_iql',
      'stage6_candidate_selector_eval','stage6_multi_oracle_eval','llm_runs'
    )
  )
  $ff = Join-Path $CompatRoot 'data\final_freeze'
  New-Item -ItemType Directory -Path $ff -Force | Out-Null
  foreach ($name in $StageNames) {
    $target = Join-Path $FrozenArtifactRoot $name
    if (Test-Path -LiteralPath $target -PathType Container) {
      New-ReproJunction -Link (Join-Path $ff $name) -Target $target -CompatRoot $CompatRoot
    }
  }
  # Stage artifacts remain immutable reference junctions.  Contract/config
  # directories are different: current analysis may materialize configs or
  # write derived ledgers/verifier outputs.  Seed local writable overlays
  # instead of exposing write paths through the sealed reference.
  foreach ($name in @('configs','ledgers','verification')) {
    $target = Join-Path $ReferenceRoot "contracts\$name"
    $overlay = Join-Path $ff $name
    if (Test-Path -LiteralPath $target -PathType Container) {
      Copy-ReproDirectoryWritable `
        -Source $target `
        -Destination $overlay `
        -Label "Reference contract overlay $name"
    }
    else {
      New-Item -ItemType Directory -Path $overlay -Force | Out-Null
      Assert-ReproWritableDirectory `
        -Path $overlay `
        -Label "Empty reference contract overlay $name"
    }
  }
}

function Set-ReproCompatibilityEnvironment {
  param(
    [Parameter(Mandatory=$true)][string]$CompatRoot,
    [Parameter(Mandatory=$true)][string]$RunRoot,
    [string]$ReferenceRoot='',
    [string]$RawRoot='',
    [string]$TemporaryRoot='',
    [string]$SourceRoot='',
    [string]$PythonSitePackagesRoot=''
  )
  $env:CREDIT_RECOURSE_COMPAT_VIEW = '1'
  $env:CREDIT_RECOURSE_RUN_ROOT = $RunRoot
  $env:CREDIT_RECOURSE_ARTIFACT_ROOT = (Join-Path $CompatRoot 'data\final_freeze')
  $env:CREDIT_RECOURSE_ANALYSIS_ROOT = (Join-Path $CompatRoot 'data\analysis')
  if (-not [string]::IsNullOrWhiteSpace($ReferenceRoot)) { $env:CREDIT_RECOURSE_REFERENCE_ROOT = $ReferenceRoot }
  if (-not [string]::IsNullOrWhiteSpace($RawRoot)) { $env:CREDIT_RECOURSE_RAW_ROOT = $RawRoot }
  if (-not [string]::IsNullOrWhiteSpace($TemporaryRoot)) {
    New-Item -ItemType Directory -Path $TemporaryRoot -Force | Out-Null
    $env:TEMP = $TemporaryRoot
    $env:TMP = $TemporaryRoot
    $env:MPLCONFIGDIR = (Join-Path $TemporaryRoot 'matplotlib')
    $env:NUMBA_CACHE_DIR = (Join-Path $TemporaryRoot 'numba')
    $env:XDG_CACHE_HOME = (Join-Path $TemporaryRoot 'xdg')
    $env:TORCH_HOME = (Join-Path $TemporaryRoot 'torch')
    $env:HF_HOME = (Join-Path $TemporaryRoot 'huggingface')
    $env:JOBLIB_TEMP_FOLDER = (Join-Path $TemporaryRoot 'joblib')
  }
  $pythonSourceRoot = if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    Join-Path $CompatRoot 'src'
  } else {
    [System.IO.Path]::GetFullPath($SourceRoot)
  }
  if ([string]::IsNullOrWhiteSpace($PythonSitePackagesRoot)) {
    $env:PYTHONPATH = $pythonSourceRoot
  }
  else {
    if (-not (Test-Path -LiteralPath $PythonSitePackagesRoot -PathType Container)) {
      throw "Python site-packages root is missing: $PythonSitePackagesRoot"
    }
    $resolvedSitePackages = (Resolve-Path -LiteralPath $PythonSitePackagesRoot).Path
    $env:PYTHONPATH = (
      $pythonSourceRoot + [System.IO.Path]::PathSeparator + $resolvedSitePackages
    )
  }
  $env:PYTHONUTF8 = '1'
  $env:PYTHONIOENCODING = 'utf-8'
  $env:PYTHONDONTWRITEBYTECODE = '1'
}

function Write-ReproStatus {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string]$Lineage,
    [Parameter(Mandatory=$true)][ValidateSet('INCOMPLETE','PASS','PASS_WITH_SKIPS','FAIL')][string]$Status,
    [string]$Message=''
  )
  $payload = [ordered]@{
    run_id = $RunId
    lineage = $Lineage
    status = $Status
    updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    required_tasks = $RequiredTasks
    optional_tasks = $OptionalTasks
  }
  foreach ($key in $AdditionalFields.Keys) {
    $payload[$key] = $AdditionalFields[$key]
  }
  if (-not [string]::IsNullOrWhiteSpace($Message)) { $payload.message = $Message }
  $payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Get-LongPathsEnabled {
  try {
    return [int](Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled
  } catch { return -1 }
}

function New-ReproArtifactManifest {
  param(
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$RunRoot,
    [Parameter(Mandatory=$true)][string[]]$RelativePaths,
    [string]$OutRelative='manifests\artifact_manifest.json'
  )
  $out = Join-Path $RunRoot $OutRelative
  New-Item -ItemType Directory -Path (Split-Path -Parent $out) -Force | Out-Null
  $cmd = @($PythonExe,'-B','-m','credit_recourse.verification.manifest','create-artifacts','--root',$RunRoot,'--out',$out)
  foreach ($rel in $RelativePaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $RunRoot $rel))) { throw "Manifest input missing: $rel" }
    $cmd += @('--path',$rel)
  }
  Invoke-ReproChecked 'create artifact manifest' $cmd
}

function Get-ReproInfrastructureState {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$InfrastructureRoot=''
  )

  $resolvedInfrastructureRoot = if ([string]::IsNullOrWhiteSpace($InfrastructureRoot)) {
    Join-Path $Root 'data\infrastructure'
  } else {
    [System.IO.Path]::GetFullPath($InfrastructureRoot)
  }
  $statePath = Join-Path $resolvedInfrastructureRoot 'status.json'
  if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    return $null
  }

  try {
    return Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
  }
  catch {
    throw ("Infrastructure state is unreadable: {0}. {1}" -f $statePath,$_.Exception.Message)
  }
}

function Get-ReproLegacyPaths {
  param([Parameter(Mandatory=$true)][string]$Root)

  return @(
    (Join-Path $Root 'data\final_freeze'),
    (Join-Path $Root 'data\analysis'),
    (Join-Path $Root 'data\reproduction'),
    (Join-Path $Root 'runs\final_freeze')
  )
}

function Get-ReproActiveCompatibilityViews {
  param([Parameter(Mandatory=$true)][string]$Root)

  $runsRoot = Join-Path $Root 'data\runs'
  if (-not (Test-Path -LiteralPath $runsRoot -PathType Container)) {
    return @()
  }

  return @(
    Get-ChildItem -LiteralPath $runsRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
      Where-Object {
        $_.Name -eq 'compat' -and
        $null -ne $_.Parent -and
        $_.Parent.Name -eq '.work'
      } |
      Select-Object -ExpandProperty FullName
  )
}

function Resolve-ReproAcknowledgedCompatibilityViews {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string[]]$PreservedRunIds=@()
  )

  $resolved = New-Object System.Collections.Generic.List[string]
  $seen = @{}
  foreach ($runId in @($PreservedRunIds)) {
    Assert-ReproId $runId 'Acknowledged preserved RunId'
    if ($seen.ContainsKey($runId)) {
      throw "Acknowledged preserved RunId is duplicated: $runId"
    }
    $seen[$runId] = $true

    $runRoot = Join-Path $Root "data\runs\frozen\$runId"
    $statusPath = Join-Path $runRoot 'status.json'
    $compatPath = Join-Path $runRoot '.work\compat'
    if (-not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
      throw "Acknowledged preserved run status is missing: $statusPath"
    }
    if (-not (Test-Path -LiteralPath $compatPath -PathType Container)) {
      throw "Acknowledged preserved run compatibility view is missing: $compatPath"
    }

    try {
      $status = Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
    }
    catch {
      throw "Acknowledged preserved run status is unreadable: $statusPath. $($_.Exception.Message)"
    }
    if ([string]$status.run_id -ne $runId) {
      throw (
        "Acknowledged preserved run ID mismatch: requested={0} status={1}" -f
        $runId,[string]$status.run_id
      )
    }
    if ([string]$status.lineage -ne 'FrozenReplay') {
      throw "Acknowledged preserved run is not FrozenReplay: $runId"
    }
    if ([string]$status.status -notin @('FAIL','INCOMPLETE')) {
      throw (
        "Only FAIL/INCOMPLETE preserved runs may be acknowledged; run={0} status={1}" -f
        $runId,[string]$status.status
      )
    }
    $resolved.Add([System.IO.Path]::GetFullPath($compatPath))
  }
  return @($resolved)
}

function Assert-ReproInfrastructureReady {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$ReferenceId='',
    [string]$InfrastructureRoot='',
    [string]$FrozenArtifactRoot='',
    [int]$LocalLegacyConsumerCount=0,
    [string[]]$AcknowledgedCompatibilityViews=@()
  )

  $state = Get-ReproInfrastructureState `
    -Root $Root `
    -InfrastructureRoot $InfrastructureRoot
  if ($null -eq $state) {
    throw (
      'Reproduction infrastructure is not initialized. Run: ' +
      '.\REPRODUCE.ps1 -Mode InitializeInfrastructure -ReferenceId thesis_v1 -MigrateLegacy'
    )
  }

  if ([string]$state.status -ne 'PASS') {
    throw (
      "Reproduction infrastructure is not ready. status={0}. " +
      "Run VerifyInfrastructure and resolve the reported blockers."
    ) -f [string]$state.status
  }

  if (
    -not [string]::IsNullOrWhiteSpace($ReferenceId) -and
    [string]$state.reference_id -ne $ReferenceId
  ) {
    throw (
      "Infrastructure reference mismatch. expected={0}, state={1}"
    ) -f $ReferenceId,[string]$state.reference_id
  }

  $legacy = @(
    Get-ReproLegacyPaths -Root $Root |
      Where-Object { Test-Path -LiteralPath $_ }
  )
  $localFinalFreeze = Join-Path $Root 'data\final_freeze'
  $nonArtifactLegacy = @($legacy | Where-Object {
    -not [string]::Equals($_,$localFinalFreeze,[System.StringComparison]::OrdinalIgnoreCase)
  })
  if ($nonArtifactLegacy.Count -gt 0) {
    throw (
      "Legacy active roots still exist and canonical runs are blocked: {0}"
    ) -f ($nonArtifactLegacy -join '; ')
  }
  if (Test-Path -LiteralPath $localFinalFreeze) {
    if ([string]::IsNullOrWhiteSpace($FrozenArtifactRoot) -or $LocalLegacyConsumerCount -gt 0) {
      throw (
        "Local legacy data/final_freeze exists and is active or no explicit FrozenArtifactRoot was supplied: {0}"
      ) -f $localFinalFreeze
    }
  }

  $compat = @(
    Get-ReproActiveCompatibilityViews -Root $Root |
      Where-Object { $AcknowledgedCompatibilityViews -notcontains $_ }
  )
  if ($compat.Count -gt 0) {
    throw (
      "Stale compatibility views exist. Inspect and remove them before a new run: {0}"
    ) -f ($compat -join '; ')
  }
}
