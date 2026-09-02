param(
    [string]$RunRoot = (Join-Path $env:USERPROFILE 'CodexRuns\meniere_segmentation_v2'),
    [string]$PythonExe = '',
    [int]$EpochCap = 54
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $PSScriptRoot 'results\local_training_state.json'
$workerScript = Join-Path $PSScriptRoot 'run_local_training_worker.ps1'
if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $candidatePids = @($state.pid, $state.supervisor_pid, $state.worker_pid) | Where-Object { $_ }
    $live = @($candidatePids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    if ($state.status -eq 'RUNNING' -and $live.Count -gt 0) {
        Write-Output "Training is already running. PIDs: $($live -join ', ')"
        exit 0
    }
}

$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', ('"{0}"' -f $workerScript),
    '-RunRoot', ('"{0}"' -f $RunRoot),
    '-PythonExe', ('"{0}"' -f $PythonExe),
    '-EpochCap', $EpochCap
) -join ' '
$worker = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Normal `
    -PassThru
Start-Sleep -Seconds 5
if ($worker.HasExited) {
    throw "Detached training worker exited immediately with code $($worker.ExitCode)."
}
Write-Output "Detached local training worker started. Worker PID: $($worker.Id)"
Write-Output "State: $statePath"
Write-Output "Status command: .\segmentation_v2\check_training_status.ps1"
