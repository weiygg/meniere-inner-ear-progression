param(
    [Parameter(Mandatory = $true)]
    [string]$InputDirectory,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$resolvedInput = (Resolve-Path -LiteralPath $InputDirectory).Path
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null

$archives = Get-ChildItem -LiteralPath $resolvedInput -File -Filter "浙二*.rar" | Sort-Object Name
if ($archives.Count -ne 5) {
    throw "Expected 5 Zhejiang Second Hospital archives, found $($archives.Count)."
}

foreach ($archive in $archives) {
    Write-Output "EXTRACT_START`t$($archive.Name)"
    & tar -xf $archive.FullName -C $resolvedOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Extraction failed for $($archive.FullName)"
    }
    Write-Output "EXTRACT_DONE`t$($archive.Name)"
}

$dicomCount = (Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File -Filter "*.dcm").Count
$studyCount = (Get-ChildItem -LiteralPath $resolvedOutput -Directory).Count
Write-Output "COMPLETE`tstudies=$studyCount`tdicom_files=$dicomCount`toutput=$resolvedOutput"
