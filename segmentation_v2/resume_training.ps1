param(
    [string]$RunRoot = (Join-Path $env:USERPROFILE 'CodexRuns\meniere_segmentation_v2')
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_local_training.ps1') -RunRoot $RunRoot
exit $LASTEXITCODE
