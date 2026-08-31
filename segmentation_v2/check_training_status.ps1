$ErrorActionPreference = 'Stop'
$statePath = Join-Path $PSScriptRoot 'results\local_training_state.json'
if (-not (Test-Path -LiteralPath $statePath)) {
    Write-Output 'No local training state exists.'
    exit 1
}
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$start = [datetimeoffset]::Parse($state.start_time)
$elapsed = [datetimeoffset]::UtcNow - $start
$pidValue = $state.pid
$pidStatus = if ($pidValue -and (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)) { 'RUNNING' } else { 'NOT RUNNING' }
$macro = if ($null -eq $state.current_macro_dice) { 'not available until an epoch finishes' } else { '{0:N4}' -f $state.current_macro_dice }
$best = if ($null -eq $state.best_macro_dice) { 'not available' } else { '{0:N4}' -f $state.best_macro_dice }
$gpuLine = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits

Write-Output "Current experiment: $($state.experiment)"
Write-Output "Status: $($state.status)"
Write-Output "Current fold: $($state.fold)"
Write-Output "Current epoch: $($state.epoch) / $($state.total_epochs)"
Write-Output "Latest validation Macro Dice: $macro"
Write-Output "Best validation Macro Dice: $best"
Write-Output "Metric scope: $($state.metric_scope)"
Write-Output "GPU util, memory used/total MiB, temperature C: $gpuLine"
Write-Output "Training PID: $pidValue ($pidStatus)"
Write-Output "Supervisor PID: $($state.supervisor_pid)"
Write-Output "Worker PID: $($state.worker_pid)"
Write-Output ('Elapsed time: {0:dd\.hh\:mm\:ss}' -f $elapsed)
Write-Output "Latest checkpoint: $($state.checkpoint)"
Write-Output "Last update: $($state.last_update)"
foreach ($fold in $state.folds) {
    Write-Output "Fold $($fold.fold): $($fold.status), epoch $($fold.epoch), best macro $($fold.best_macro_dice)"
}
$logPath = Join-Path $PSScriptRoot ("logs\nnunet_fold{0}.log" -f $state.fold)
if (Test-Path -LiteralPath $logPath) {
    Write-Output "Log: $logPath"
    Write-Output 'Latest log lines:'
    Get-Content -LiteralPath $logPath -Tail 8
}
