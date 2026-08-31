$ErrorActionPreference = 'Stop'
$statePath = Join-Path $PSScriptRoot 'results\local_training_state.json'
$stopPath = Join-Path $PSScriptRoot 'results\stop_requested.json'
if (-not (Test-Path -LiteralPath $statePath)) {
    Write-Output 'No local training state exists.'
    exit 0
}
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.status -ne 'RUNNING') {
    Write-Output "Training status is $($state.status); no process was stopped."
    exit 0
}
@{ requested_utc = [datetimeoffset]::UtcNow.ToString('o'); requested_for_pid = $state.pid } |
    ConvertTo-Json | Set-Content -LiteralPath $stopPath -Encoding utf8
Write-Output "Graceful stop requested for this project's training PID $($state.pid)."
Write-Output 'The runner will send CTRL_BREAK and save checkpoint_latest.pth at the next monitor interval.'

$stopped = $false
for ($attempt = 0; $attempt -lt 12; $attempt++) {
    Start-Sleep -Seconds 5
    $current = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($current.status -ne 'RUNNING') {
        $stopped = $true
        Write-Output "Training stopped with state $($current.status). Checkpoint: $($current.checkpoint)"
        break
    }
}
if (-not $stopped) {
    $current = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    foreach ($exactPid in @($current.pid, $current.supervisor_pid, $current.worker_pid)) {
        if ($exactPid -and (Get-Process -Id $exactPid -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $exactPid -Force
        }
    }
    $current.status = 'INTERRUPTED'
    $current.last_update = [datetimeoffset]::UtcNow.ToString('o')
    $current.pid = $null
    $current | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
    Write-Warning 'Graceful stop timed out; only the exact recorded project PIDs were force-stopped.'
}
