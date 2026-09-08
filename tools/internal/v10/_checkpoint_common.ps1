Set-StrictMode -Version Latest

function Get-ReproCheckpointContext {
  param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$ExecutionRunRoot
  )
  $required = @(
    'CREDIT_RECOURSE_CHECKPOINT_CATALOG_COUNT'
  )
  foreach ($name in $required) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($value)) {
      throw "Checkpoint context is incomplete: $name"
    }
  }
  $store = [Environment]::GetEnvironmentVariable('CREDIT_RECOURSE_CHECKPOINT_STORE')
  if ([string]::IsNullOrWhiteSpace($store)) {
    $store = Join-Path $ProjectRoot 'data\checkpoints\frozen_extensions'
  }
  $runId = Split-Path -Leaf $ExecutionRunRoot
  $repositoryRoot = [Environment]::GetEnvironmentVariable('CREDIT_RECOURSE_CHECKPOINT_REPOSITORY_ROOT')
  if ([string]::IsNullOrWhiteSpace($repositoryRoot)) { $repositoryRoot = $ProjectRoot }
  $catalogCount = [int]([Environment]::GetEnvironmentVariable('CREDIT_RECOURSE_CHECKPOINT_CATALOG_COUNT'))
  if ($catalogCount -ne 21) {
    throw "Checkpoint historical catalog count must be 21; got $catalogCount"
  }
  $analysisProfile = [Environment]::GetEnvironmentVariable('CREDIT_RECOURSE_CHECKPOINT_PROFILE')
  if ([string]::IsNullOrWhiteSpace($analysisProfile)) { $analysisProfile = 'historical_20260715' }
  if ($analysisProfile -ne 'historical_20260715') {
    throw "Checkpoint analysis profile must be historical_20260715; got $analysisProfile"
  }
  return [ordered]@{
    python_exe = $PythonExe
    repository_root = [System.IO.Path]::GetFullPath($repositoryRoot)
    store_root = [System.IO.Path]::GetFullPath($store)
    receipt_root = Join-Path $ExecutionRunRoot 'checkpoint_receipts'
    producer_run_id = $runId
    analysis_profile = $analysisProfile
    catalog_count = $catalogCount
    force_recompute = ([Environment]::GetEnvironmentVariable('CREDIT_RECOURSE_CHECKPOINT_FORCE_RECOMPUTE') -eq '1')
  }
}

function ConvertTo-ReproSafeFileName {
  param([Parameter(Mandatory=$true)][string]$Value)
  return ($Value -replace '[^A-Za-z0-9_.-]', '_')
}

function Get-ReproCheckpointProducerFiles {
  param(
    [Parameter(Mandatory=$true)][string]$TaskName
  )
  $analysis='src/credit_recourse/analysis'
  $axis=@(
    "$analysis/n5m_axis_swap_intervention.py",
    "$analysis/llm_action_budget_ablation.py",
    "$analysis/n5_7_10c_holm_inference.py"
  )
  if($TaskName -eq 'paper_repro.historical_20260715'){
    return @(
      'tools/internal/v10/run_postfreeze_analysis.ps1',
      "$analysis/paper_repro_analysis.py",
      "$analysis/paper_repro_assets.py",
      "$analysis/paper_output_layout.py",
      "$analysis/remaining_thesis_analyses.py",
      'src/credit_recourse/contracts/paper_reproduction.py',
      'src/credit_recourse/configs/paper_reproduction_profile.json',
      'src/credit_recourse/configs/paper_analysis_profiles.json'
    ) + $axis
  }
  if($TaskName -eq 'extensions.e2.c4r_matched'){
    return @("$analysis/c4r_matched_inference.py","$analysis/n5_7_10c_holm_inference.py")
  }
  if($TaskName -eq 'extensions.e3.c4r_journal' -or $TaskName -like 'extensions.e3.*'){
    return @(
      'tools/internal/v10/run_c4r_final_analysis.ps1',
      "$analysis/c4r_matched_inference_v3.py",
      "$analysis/c4r_axis_swap_summary.py",
      "$analysis/c4r_tost_equivalence.py",
      'src/credit_recourse/configs/c4r_journal_extension_prereg_v3.json'
    ) + $axis
  }
  if($TaskName -eq 'extensions.e4.common_cohort'){
    return @("$analysis/c4r_haiku_axis_common_cohort.py")
  }
  if($TaskName -like 'extensions.e4.axis.*'){
    return $axis
  }
  if($TaskName -eq 'extensions.e4.haiku_characterization'){
    return @(
      'tools/internal/original/run_haiku45_axis_compare.ps1',
      "$analysis/c4r_haiku_axis_common_cohort.py",
      "$analysis/c4r_haiku_axis_summary.py"
    ) + $axis
  }
  throw "No scientific producer registry entry for checkpoint task: $TaskName"
}

function Get-ReproScientificProducerIdentity {
  param(
    [Parameter(Mandatory=$true)][string]$RepositoryRoot,
    [Parameter(Mandatory=$true)][string[]]$ProducerFiles
  )
  $root=[System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\','/')
  $rows=@()
  foreach($relative in @($ProducerFiles|Sort-Object -Unique)){
    $normalized=$relative.Replace('\','/').TrimStart('/')
    if($normalized -match '(^|/)\.\.(/|$)'){throw "Producer path traversal is forbidden: $relative"}
    $full=Join-Path $root $normalized
    if(-not(Test-Path -LiteralPath $full -PathType Leaf)){throw "Scientific producer file missing: $normalized"}
    $rows += [ordered]@{
      path=$normalized
    }
  }
  return [ordered]@{
    schema_version='repro_scientific_producer_identity_v1'
    files=$rows
  }
}

function Invoke-ReproCheckpointCli {
  param(
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Context,
    [Parameter(Mandatory=$true)][string[]]$Arguments,
    [switch]$AllowFailure
  )
  & $Context.python_exe -B -m credit_recourse.repro.checkpoint @Arguments
  $exitCode = $LASTEXITCODE
  if (-not $AllowFailure -and $exitCode -ne 0) {
    throw "Checkpoint command failed exit=${exitCode}: $($Arguments -join ' ')"
  }
  return $exitCode
}

function New-ReproCheckpointContract {
  param(
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Context,
    [Parameter(Mandatory=$true)][string]$TaskName,
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Arguments,
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Inputs,
    [string[]]$ProducerFiles=@()
  )
  if($ProducerFiles.Count -eq 0){$ProducerFiles=@(Get-ReproCheckpointProducerFiles -TaskName $TaskName)}
  $producerIdentity=Get-ReproScientificProducerIdentity `
    -RepositoryRoot ([string]$Context.repository_root) `
    -ProducerFiles $ProducerFiles
  return [ordered]@{
    schema_version = 'repro_checkpoint_task_contract_v2'
    task = $TaskName
    arguments = $Arguments
    inputs = $Inputs
    source = [ordered]@{scientific_producer_identity = $producerIdentity}
    environment = [ordered]@{
    }
    profile = [ordered]@{
      analysis_profile = [string]$Context.analysis_profile
      historical_catalog_count = [int]$Context.catalog_count
    }
  }
}

function Invoke-ReproCheckpointedTask {
  param(
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Context,
    [Parameter(Mandatory=$true)][string]$TaskName,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Arguments,
    [Parameter(Mandatory=$true)][System.Collections.IDictionary]$Inputs,
    [string[]]$ProducerFiles=@(),
    [Parameter(Mandatory=$true)][scriptblock]$Execute,
    [Parameter(Mandatory=$true)][scriptblock]$Validate,
    [switch]$InPlaceExecution
  )
  if (Test-Path -LiteralPath $OutputRoot) {
    throw "Checkpoint task output must not exist before assembly: $OutputRoot"
  }
  $safeName = ConvertTo-ReproSafeFileName $TaskName
  $receiptRoot = [string]$Context.receipt_root
  New-Item -ItemType Directory -Path $receiptRoot -Force | Out-Null
  $contractPath = Join-Path $receiptRoot "$safeName.contract.json"
  $producerPath = Join-Path $receiptRoot "$safeName.producer.json"
  $receiptPath = Join-Path $receiptRoot "$safeName.reuse.json"
  $contract=New-ReproCheckpointContract -Context $Context -TaskName $TaskName `
    -Arguments $Arguments -Inputs $Inputs -ProducerFiles $ProducerFiles
  $producerIdentity=$contract.source.scientific_producer_identity
  $producer = [ordered]@{
    run_id = [string]$Context.producer_run_id
  }
  $contract | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $contractPath -Encoding UTF8
  $producer | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $producerPath -Encoding UTF8
  $checkpointRoot = Join-Path ([string]$Context.store_root) "$safeName\finalized"
  $canReuse = (Test-Path -LiteralPath $checkpointRoot -PathType Container) -and (-not [bool]$Context.force_recompute)
  if ($canReuse) {
    Write-Host "CHECKPOINT REUSE: $TaskName" -ForegroundColor Green
    Invoke-ReproCheckpointCli -Context $Context -Arguments @(
      'materialize','--checkpoint',$checkpointRoot,'--destination',$OutputRoot,
      '--contract',$contractPath,'--receipt',$receiptPath
    ) | Out-Null
    & $Validate $OutputRoot
    return [pscustomobject]@{ task=$TaskName; reused=$true; checkpoint=$checkpointRoot }
  }

  $temporaryOutput = if ($InPlaceExecution) {
    $OutputRoot
  } else {
    "$OutputRoot.incomplete.$([Guid]::NewGuid().ToString('N'))"
  }
  Write-Host "CHECKPOINT COMPUTE: $TaskName" -ForegroundColor Cyan
  try {
    & $Execute $temporaryOutput
    if (-not (Test-Path -LiteralPath $temporaryOutput -PathType Container)) {
      throw "Checkpoint task did not create its output: $temporaryOutput"
    }
    & $Validate $temporaryOutput
    if ([bool]$Context.force_recompute -and (Test-Path -LiteralPath $checkpointRoot -PathType Container)) {
      # Clean mode must use the freshly computed result without mutating or
      # replacing the existing immutable checkpoint for the same contract.
      Invoke-ReproCheckpointCli -Context $Context -Arguments @(
        'verify','--checkpoint',$checkpointRoot,'--contract',$contractPath
      ) | Out-Null
      if (-not $InPlaceExecution) {
        Move-Item -LiteralPath $temporaryOutput -Destination $OutputRoot
      }
      [ordered]@{
        schema_version='repro_checkpoint_reuse_receipt_v1';status='PASS'
        task=$TaskName;reuse=$false;force_recompute=$true
        existing_checkpoint=$checkpointRoot;producer=$producer
      } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
      & $Validate $OutputRoot
      return [pscustomobject]@{ task=$TaskName; reused=$false; checkpoint=$null; force_recompute=$true }
    }
    $finalized = (& $Context.python_exe -B -m credit_recourse.repro.checkpoint finalize `
      --store-root ([string]$Context.store_root) --task-name $TaskName `
      --contract $contractPath --producer $producerPath --output $temporaryOutput).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($finalized)) {
      throw "Unable to finalize checkpoint for $TaskName"
    }
    if (-not $InPlaceExecution) {
      Remove-Item -LiteralPath $temporaryOutput -Recurse -Force
      Invoke-ReproCheckpointCli -Context $Context -Arguments @(
        'materialize','--checkpoint',$finalized,'--destination',$OutputRoot,
        '--contract',$contractPath
      ) | Out-Null
    }
    $finalManifest = Get-Content -LiteralPath (Join-Path $finalized 'checkpoint_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    [ordered]@{
      schema_version='repro_checkpoint_reuse_receipt_v1';status='PASS'
      task=$TaskName;reuse=$false;force_recompute=[bool]$Context.force_recompute
      checkpoint=$finalized;producer=$producer
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
    & $Validate $OutputRoot
    return [pscustomobject]@{ task=$TaskName; reused=$false; checkpoint=$finalized }
  }
  catch {
    Write-Warning "Checkpoint task interrupted or failed; incomplete output is not reusable: $temporaryOutput"
    throw
  }
}
