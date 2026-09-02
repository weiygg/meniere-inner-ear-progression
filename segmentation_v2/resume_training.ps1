param(
    [string]$RunRoot = (Join-Path $env:USERPROFILE 'CodexRuns\meniere_segmentation_v2'),
    [int]$EpochCap = 54
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_local_training.ps1') -RunRoot $RunRoot -EpochCap $EpochCap
exit $LASTEXITCODE
