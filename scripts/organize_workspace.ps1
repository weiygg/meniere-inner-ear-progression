[CmdletBinding()]
param(
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$dataRoot = Join-Path $workspaceRoot 'data'
$archiveRoot = Join-Path $workspaceRoot 'archive\legacy'
$manifestPath = Join-Path $dataRoot 'manifests\workspace_reorganization_20260824.json'

function Assert-Within([string]$Path, [string]$Base) {
    $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $resolvedBase = [System.IO.Path]::GetFullPath($Base).TrimEnd('\')
    $prefix = $resolvedBase + '\'
    if (-not $resolvedPath.Equals($resolvedBase, [System.StringComparison]::OrdinalIgnoreCase) -and
        -not $resolvedPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes approved root: $resolvedPath (base: $resolvedBase)"
    }
    return $resolvedPath
}

function Relative-Path([string]$Path) {
    return [System.IO.Path]::GetRelativePath($workspaceRoot, [System.IO.Path]::GetFullPath($Path)).Replace('\', '/')
}

function Get-ItemStats([string]$Path) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $item = Get-Item -LiteralPath $Path
        return @{ files = 1; bytes = [int64]$item.Length }
    }
    $measure = Get-ChildItem -LiteralPath $Path -Recurse -File -Force | Measure-Object -Property Length -Sum
    return @{ files = [int]$measure.Count; bytes = [int64]$measure.Sum }
}

function Ensure-Directory([string]$Path, [string]$Base) {
    $safePath = Assert-Within $Path $Base
    if ($Apply -and -not (Test-Path -LiteralPath $safePath)) {
        New-Item -ItemType Directory -Path $safePath -Force | Out-Null
    }
}

$actions = [System.Collections.Generic.List[object]]::new()

function Record-Action(
    [string]$Action,
    [string]$Source,
    [string]$Destination,
    [string]$Reason,
    [hashtable]$Stats,
    [string]$Sha256,
    [string]$Status,
    [string]$Recoverability
) {
    $actions.Add([ordered]@{
        action = $Action
        source = Relative-Path $Source
        destination = if ($Destination) { Relative-Path $Destination } else { $null }
        files = $Stats.files
        bytes = $Stats.bytes
        sha256 = if ($Sha256) { $Sha256.ToLowerInvariant() } else { $null }
        reason = $Reason
        status = $Status
        recoverability = $Recoverability
    })
}

# Exact duplicate root archives: retain the protected data/ copy and remove only the duplicate.
$duplicateArchives = @(
    @{ Name='中心2外部验证1.rar'; Final='data\centers\external_validation_1\reference_masks\中心2外部验证1.rar' },
    @{ Name='中心3外部验证2.rar'; Final='data\centers\external_validation_2\reference_masks\中心3外部验证2.rar' }
)
foreach ($entry in $duplicateArchives) {
    $name = $entry.Name
    $duplicate = Assert-Within (Join-Path $workspaceRoot $name) $workspaceRoot
    $retained = Assert-Within (Join-Path $dataRoot $name) $dataRoot
    $retainedFinal = Assert-Within (Join-Path $workspaceRoot $entry.Final) $dataRoot
    if ((Test-Path -LiteralPath $duplicate) -and (Test-Path -LiteralPath $retained)) {
        $duplicateHash = (Get-FileHash -LiteralPath $duplicate -Algorithm SHA256).Hash
        $retainedHash = (Get-FileHash -LiteralPath $retained -Algorithm SHA256).Hash
        if ($duplicateHash -ne $retainedHash) { throw "Archive hashes differ: $name" }
        $stats = Get-ItemStats $duplicate
        if ($Apply) { Remove-Item -LiteralPath $duplicate -Force }
        Record-Action 'delete_exact_duplicate' $duplicate $retainedFinal 'Byte-identical root duplicate; protected data copy retained.' $stats $duplicateHash $(if ($Apply) {'deleted'} else {'planned'}) 'fully_recoverable_from_retained_copy'
    }
}

$dataMoves = @(
    @{ Source='MD患者评估20260713.xlsx'; Destination='data\clinical\MD患者评估20260713.xlsx'; Reason='Protected mixed-center clinical source moved under the protected data boundary.' },
    @{ Source='data\丽水-xjj内耳分割4.rar'; Destination='data\centers\primary_lishui\raw_archives\丽水-xjj内耳分割4.rar'; Reason='Primary Lishui development archive.' },
    @{ Source='data\丽水-xjj内耳分割4'; Destination='data\centers\primary_lishui\extracted\丽水-xjj内耳分割4'; Reason='Primary Lishui extracted development data.' },
    @{ Source='data\浙二1-1.rar'; Destination='data\centers\external_validation_1\raw_archives\浙二1-1.rar'; Reason='Frozen external validation 1 source batch.' },
    @{ Source='data\浙二1-2.rar'; Destination='data\centers\external_validation_1\raw_archives\浙二1-2.rar'; Reason='Frozen external validation 1 source batch.' },
    @{ Source='data\中心2外部验证1.rar'; Destination='data\centers\external_validation_1\reference_masks\中心2外部验证1.rar'; Reason='Manual reference masks for external validation 1.' },
    @{ Source='data\浙二1-1'; Destination='data\centers\external_validation_1\extracted\浙二1-1'; Reason='Extracted external validation 1 imaging.' },
    @{ Source='data\浙二1-2'; Destination='data\centers\external_validation_1\extracted\浙二1-2'; Reason='Extracted external validation 1 imaging.' },
    @{ Source='data\浙二2-1.rar'; Destination='data\centers\external_validation_2\raw_archives\浙二2-1.rar'; Reason='Frozen external validation 2 source batch.' },
    @{ Source='data\浙二2-2.rar'; Destination='data\centers\external_validation_2\raw_archives\浙二2-2.rar'; Reason='Frozen external validation 2 source batch.' },
    @{ Source='data\浙二2例新.rar'; Destination='data\centers\external_validation_2\raw_archives\浙二2例新.rar'; Reason='Frozen external validation 2 addendum.' },
    @{ Source='data\中心3外部验证2.rar'; Destination='data\centers\external_validation_2\reference_masks\中心3外部验证2.rar'; Reason='Manual reference masks for external validation 2.' },
    @{ Source='data\浙二2-1'; Destination='data\centers\external_validation_2\extracted\浙二2-1'; Reason='Extracted external validation 2 imaging.' },
    @{ Source='data\浙二2-2'; Destination='data\centers\external_validation_2\extracted\浙二2-2'; Reason='Extracted external validation 2 imaging.' },
    @{ Source='data\浙二2例新'; Destination='data\centers\external_validation_2\extracted\浙二2例新'; Reason='Extracted external validation 2 imaging addendum.' }
)

foreach ($move in $dataMoves) {
    $source = Assert-Within (Join-Path $workspaceRoot $move.Source) $workspaceRoot
    $destination = Assert-Within (Join-Path $workspaceRoot $move.Destination) $dataRoot
    if (Test-Path -LiteralPath $source) {
        if (Test-Path -LiteralPath $destination) { throw "Both source and destination exist: $source -> $destination" }
        $stats = Get-ItemStats $source
        $hash = if (Test-Path -LiteralPath $source -PathType Leaf) { (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash } else { '' }
        Ensure-Directory (Split-Path -Parent $destination) $dataRoot
        if ($Apply) { Move-Item -LiteralPath $source -Destination $destination }
        Record-Action 'move_protected_data' $source $destination $move.Reason $stats $hash $(if ($Apply) {'moved'} else {'planned'}) 'retained_at_destination'
    } elseif (Test-Path -LiteralPath $destination) {
        $stats = Get-ItemStats $destination
        Record-Action 'move_protected_data' $source $destination $move.Reason $stats '' 'already_moved' 'retained_at_destination'
    } else {
        throw "Missing both source and destination: $source"
    }
}

$legacyMoves = @(
    @{ Source='xjj内耳分割'; Destination='archive\legacy\segmentation_versions\xjj内耳分割'; Reason='Older segmentation/geometry tree; authority unresolved.' },
    @{ Source='xjj内耳分割2'; Destination='archive\legacy\segmentation_versions\xjj内耳分割2'; Reason='Older segmentation/geometry tree; authority unresolved.' },
    @{ Source='seg3'; Destination='archive\legacy\segmentation_versions\seg3'; Reason='Superseded segmentation/geometry tree; seg4 remains active candidate.' }
)
foreach ($move in $legacyMoves) {
    $source = Assert-Within (Join-Path $workspaceRoot $move.Source) $workspaceRoot
    $destination = Assert-Within (Join-Path $workspaceRoot $move.Destination) $archiveRoot
    if (Test-Path -LiteralPath $source) {
        if (Test-Path -LiteralPath $destination) { throw "Both source and destination exist: $source -> $destination" }
        $stats = Get-ItemStats $source
        Ensure-Directory (Split-Path -Parent $destination) $archiveRoot
        if ($Apply) { Move-Item -LiteralPath $source -Destination $destination }
        Record-Action 'archive_legacy_tree' $source $destination $move.Reason $stats '' $(if ($Apply) {'moved'} else {'planned'}) 'retained_in_archive'
    } elseif (Test-Path -LiteralPath $destination) {
        $stats = Get-ItemStats $destination
        Record-Action 'archive_legacy_tree' $source $destination $move.Reason $stats '' 'already_moved' 'retained_in_archive'
    } else {
        throw "Missing both source and destination: $source"
    }
}

$obsoleteDirectories = @(
    @{ Path='results_md_progression\final\clinical_pebm_external_validation_20260731'; Reason='Obsolete affected-ear proxy; explicitly rejected for final inference.' },
    @{ Path='results_md_progression\final\clinical_pebm_z2_development_20260801'; Reason='Reversed-center clinical pilot with the same rejected affected-ear proxy.' },
    @{ Path='results_md_progression\final\patient_level_md_pebm_20260801'; Reason='Exploratory worse-ear/symptom-event P-EBM inconsistent with the current protocol.' },
    @{ Path='results_md_progression\intermediate\external_manual_validation_20260817_aborted_slow_run'; Reason='Aborted incomplete validation run.' },
    @{ Path='results_md_progression\intermediate\external_manual_validation_20260817_pre_override_qc_invalid'; Reason='Invalid pre-override QC run superseded by frozen final validation.' },
    @{ Path='results_md_progression\intermediate\external_manual_override_inference_20260817_failed_geometry'; Reason='Empty failed-geometry output directory.' }
)
foreach ($entry in $obsoleteDirectories) {
    $target = Assert-Within (Join-Path $workspaceRoot $entry.Path) (Join-Path $workspaceRoot 'results_md_progression')
    if (Test-Path -LiteralPath $target) {
        $stats = Get-ItemStats $target
        if ($Apply) { Remove-Item -LiteralPath $target -Recurse -Force }
        Record-Action 'delete_obsolete_analysis' $target '' $entry.Reason $stats '' $(if ($Apply) {'deleted'} else {'planned'}) 'not_locally_recoverable_regenerable_from_preserved_code_and_sources'
    }
}

$deletedActions = @($actions | Where-Object { $_['action'] -like 'delete_*' })
$movedActions = @($actions | Where-Object { $_['action'] -like 'move_*' -or $_['action'] -like 'archive_*' })
$deletedFiles = ($deletedActions | ForEach-Object { [int64]$_['files'] } | Measure-Object -Sum).Sum
$deletedBytes = ($deletedActions | ForEach-Object { [int64]$_['bytes'] } | Measure-Object -Sum).Sum
$movedFiles = ($movedActions | ForEach-Object { [int64]$_['files'] } | Measure-Object -Sum).Sum
$movedBytes = ($movedActions | ForEach-Object { [int64]$_['bytes'] } | Measure-Object -Sum).Sum

$payload = [ordered]@{
    schema_version = 1
    applied = [bool]$Apply
    generated_local = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    center_split_config = 'configs/center_split.yaml'
    safety = [ordered]@{
        raw_source_content_deleted = $false
        valid_frozen_external_validation_deleted = $false
        exact_duplicate_hash_required_before_deletion = $true
        all_paths_workspace_bounded = $true
    }
    totals = [ordered]@{
        actions = $actions.Count
        deleted_files = [int64]$deletedFiles
        deleted_bytes = [int64]$deletedBytes
        moved_files = [int64]$movedFiles
        moved_bytes = [int64]$movedBytes
    }
    actions = $actions
}

if ($Apply) {
    Ensure-Directory (Split-Path -Parent $manifestPath) $dataRoot
    [System.IO.File]::WriteAllText($manifestPath, ($payload | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
}
$payload | ConvertTo-Json -Depth 8
