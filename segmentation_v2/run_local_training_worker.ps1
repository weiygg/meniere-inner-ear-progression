param(
    [string]$RunRoot = (Join-Path $env:USERPROFILE 'CodexRuns\meniere_segmentation_v2'),
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'logs'
$resultDir = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Force -Path $logDir, $resultDir | Out-Null
$workerLog = Join-Path $logDir 'local_training_worker.log'
$environmentLog = Join-Path $logDir 'local_environment.txt'

Start-Transcript -LiteralPath $workerLog -Append | Out-Null
try {
    $environment = @()
    $environment += 'hostname'
    $environment += (& hostname | Out-String).Trim()
    $environment += 'whoami'
    $environment += (& whoami | Out-String).Trim()
    $environment += 'Get-Location'
    $environment += (Get-Location).Path
    $environment += 'nvidia-smi'
    $environment += (& nvidia-smi | Out-String).TrimEnd()
    $environment += 'python --version'
    $environment += (& $PythonExe --version 2>&1 | Out-String).Trim()
    $environment += 'where.exe python'
    $environment += (& where.exe python 2>&1 | Out-String).Trim()
    $environment += 'torch CUDA probe'
    $probe = "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('vram:', torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None, 'GB')"
    $environment += (& $PythonExe -c $probe 2>&1 | Out-String).Trim()
    $environment | Set-Content -LiteralPath $environmentLog -Encoding utf8

    Set-Location -LiteralPath $repoRoot
    & $PythonExe -u (Join-Path $PSScriptRoot 'scripts\run_local_5fold.py') `
        --run-root $RunRoot `
        --python $PythonExe `
        --experiment E1 `
        --plans nnUNetPlans `
        --poll-seconds 30 `
        --worker-pid $PID
    exit $LASTEXITCODE
}
finally {
    Stop-Transcript | Out-Null
}
