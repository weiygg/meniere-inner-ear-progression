from __future__ import annotations

import argparse
import heapq
import itertools
import math
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import skeletonize

from mdp_utils import load_config, setup_logger, write_xlsx


MASK_RE = re.compile(r"^(?P<id>.+?)(?P<side>[LR])_(?P<structure>[^.]+)\.nii\.gz$", re.I)
CANAL_STRUCTURES = {"HSC", "PSC", "SSC"}
IMAGE_STRUCTURES = {"T2", "REAL"}
NEIGHBOUR_OFFSETS = [
    x for x in itertools.product((-1, 0, 1), repeat=3) if x != (0, 0, 0)
]


def _largest_component(binary: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        raise ValueError("empty mask")
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (int(np.argmax(sizes)) + 1)


def _surface(binary: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    padded = np.pad(binary.astype(np.uint8), 1)
    verts, faces, _, _ = marching_cubes(padded, 0.5, spacing=tuple(spacing))
    verts -= spacing
    return verts, faces, float(mesh_surface_area(verts, faces))


def basic_features(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    binary = np.asanyarray(img.dataobj) > 0
    if not binary.any():
        raise ValueError("empty mask")
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
    coords = np.argwhere(binary)
    physical = nib.affines.apply_affine(img.affine, coords)
    voxel_volume = abs(float(np.linalg.det(img.affine[:3, :3])))
    volume = float(len(coords) * voxel_volume)
    verts, faces, area = _surface(binary, spacing)
    centered = physical - physical.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    projections = centered @ eigvecs[:, order]
    axes = np.ptp(projections, axis=0)
    hull_points = verts
    if len(verts) >= 4:
        try:
            hull_points = verts[ConvexHull(verts).vertices]
        except Exception:
            hull_points = verts
    if len(hull_points) > 600:
        hull_points = hull_points[np.linspace(0, len(hull_points) - 1, 600, dtype=int)]
    diameter = float(pdist(hull_points).max()) if len(hull_points) > 1 else 0.0
    sphericity = math.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area if area else None
    compactness = 36 * math.pi * volume * volume / area**3 if area else None
    centroid = physical.mean(axis=0)
    result = {
        "volume_mm3": volume,
        "surface_area_mm2": area,
        "surface_to_volume_ratio": area / volume,
        "sphericity": sphericity,
        "compactness": compactness,
        "maximum_3d_diameter_mm": diameter,
        "principal_axis_lengths_mm": tuple(float(x) for x in axes),
        "elongation": float(axes[1] / axes[0]) if axes[0] else None,
        "flatness": float(axes[2] / axes[0]) if axes[0] else None,
        "centroid_mm": tuple(float(x) for x in centroid),
    }
    return result, binary, spacing, img.affine


def features(path: Path) -> dict:
    """Backward-compatible basic feature API used by unit tests and callers."""
    return basic_features(path)[0]


def _build_graph(points: np.ndarray, affine: np.ndarray):
    index = {tuple(p): i for i, p in enumerate(points)}
    physical = nib.affines.apply_affine(affine, points)
    graph: list[list[tuple[int, float]]] = [[] for _ in points]
    for i, p in enumerate(points):
        for off in NEIGHBOUR_OFFSETS:
            q = tuple((p + off).tolist())
            j = index.get(q)
            if j is not None and j > i:
                w = float(np.linalg.norm(physical[i] - physical[j]))
                graph[i].append((j, w))
                graph[j].append((i, w))
    return graph, physical


def _dijkstra(graph, start: int):
    dist = np.full(len(graph), np.inf)
    prev = np.full(len(graph), -1, dtype=int)
    dist[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        d, u = heapq.heappop(queue)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(queue, (nd, v))
    return dist, prev


def _longest_path(graph) -> tuple[list[int], bool]:
    if len(graph) < 4:
        raise ValueError("skeleton has fewer than four voxels")
    endpoints = [i for i, edges in enumerate(graph) if len(edges) == 1]
    closed_loop = len(endpoints) == 0
    if closed_loop and all(len(edges) == 2 for edges in graph):
        cycle = [0]
        previous = -1
        current = 0
        while True:
            candidates = [node for node, _ in graph[current] if node != previous]
            if not candidates:
                raise ValueError("closed skeleton traversal failed")
            following = candidates[0]
            if following == cycle[0]:
                cycle.append(following)
                break
            if following in cycle:
                raise ValueError("closed skeleton contains a premature cycle")
            cycle.append(following)
            previous, current = current, following
        return cycle, True
    candidates = endpoints if len(endpoints) >= 2 else list(range(len(graph)))
    first = int(candidates[0])
    distance_first, _ = _dijkstra(graph, first)
    far = int(max(candidates, key=lambda node: distance_first[node] if np.isfinite(distance_first[node]) else -1))
    distance_far, previous_far = _dijkstra(graph, far)
    target = int(max(candidates, key=lambda node: distance_far[node] if np.isfinite(distance_far[node]) else -1))
    best = (float(distance_far[target]), far, target, previous_far)
    _, start, target, prev = best
    if start is None or target is None or prev is None:
        raise ValueError("no connected skeleton path")
    path = [int(target)]
    while path[-1] != start:
        parent = int(prev[path[-1]])
        if parent < 0:
            raise ValueError("skeleton path is disconnected")
        path.append(parent)
    path.reverse()
    return path, closed_loop


def _smooth_path(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n < 7:
        return points.copy()
    window = min(11, n if n % 2 == 1 else n - 1)
    window = max(window, 5)
    return np.column_stack(
        [savgol_filter(points[:, j], window_length=window, polyorder=2, mode="interp") for j in range(3)]
    )


def centerline_features(binary: np.ndarray, spacing: np.ndarray, affine: np.ndarray) -> dict:
    clean = _largest_component(binary)
    skeleton = skeletonize(clean)
    voxels = np.argwhere(skeleton)
    graph, physical = _build_graph(voxels, affine)
    path_idx, closed_loop = _longest_path(graph)
    path_voxels = voxels[path_idx]
    path = _smooth_path(physical[path_idx])
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    length = float(steps.sum())
    if length <= 0:
        raise ValueError("zero-length centerline")
    radii = ndimage.distance_transform_edt(clean, sampling=tuple(spacing))[tuple(path_voxels.T)]
    s = np.r_[0.0, np.cumsum(steps)]
    unique = np.r_[True, np.diff(s) > 1e-8]
    path_u, s_u = path[unique], s[unique]
    if len(path_u) < 5:
        raise ValueError("centerline has insufficient unique points")
    d1 = np.gradient(path_u, s_u, axis=0)
    d2 = np.gradient(d1, s_u, axis=0)
    d3 = np.gradient(d2, s_u, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = np.cross(d1, d2)
    curvature = np.linalg.norm(cross, axis=1) / np.maximum(speed**3, 1e-12)
    torsion = np.einsum("ij,ij->i", cross, d3) / np.maximum(np.sum(cross**2, axis=1), 1e-12)
    centered = path_u - path_u.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    residuals = centered @ normal
    return {
        "centerline_length_mm": length,
        "mean_diameter_mm": float(2 * np.mean(radii)),
        "minimum_diameter_mm": float(2 * np.min(radii)),
        "mean_curvature_per_mm": float(np.nanmean(curvature)),
        "maximum_curvature_per_mm": float(np.nanmax(curvature)),
        "mean_abs_torsion_per_mm": float(np.nanmean(np.abs(torsion))),
        "plane_rms_residual_mm": float(np.sqrt(np.mean(residuals**2))),
        "plane_normal": tuple(float(x) for x in normal),
        "skeleton_voxel_n": int(len(voxels)),
        "main_path_point_n": int(len(path_idx)),
        "closed_loop_skeleton": bool(closed_loop),
    }


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        raise ValueError("plane normal has zero length")
    cosine = abs(float(np.dot(a, b))) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract inner-ear morphometry and canal geometry")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    _, paths = load_config(args.config)
    log = setup_logger("morph", paths.logs / "05_extract_inner_ear_morphometry.log")
    rows, centerline_rows, errors = [], [], []
    centroids: dict[tuple[str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
    normals: dict[tuple[str, str, str], dict[str, np.ndarray]] = defaultdict(dict)

    feature_keys = [
        "volume_mm3", "surface_area_mm2", "surface_to_volume_ratio", "sphericity",
        "compactness", "maximum_3d_diameter_mm", "principal_axis_lengths_mm",
        "elongation", "flatness", "centroid_mm",
    ]
    centerline_keys = [
        "centerline_length_mm", "mean_diameter_mm", "minimum_diameter_mm",
        "mean_curvature_per_mm", "maximum_curvature_per_mm", "mean_abs_torsion_per_mm",
        "plane_rms_residual_mm", "plane_normal", "skeleton_voxel_n", "main_path_point_n",
        "closed_loop_skeleton",
    ]

    for batch in paths.segmentation_batches:
        for subject in sorted((paths.segmentation_root / batch).glob("sub*")):
            if not subject.is_dir():
                continue
            for file_path in sorted(subject.glob("*.nii.gz")):
                match = MASK_RE.match(file_path.name)
                if not match or match.group("structure").upper() in IMAGE_STRUCTURES:
                    continue
                side = match.group("side").upper()
                structure = match.group("structure")
                structure_upper = structure.upper()
                rel = str(file_path.relative_to(paths.project_root))
                try:
                    feat, binary, spacing, affine = basic_features(file_path)
                    rows.append([
                        batch, subject.name, side, structure, rel,
                        *[feat[k] for k in feature_keys],
                    ])
                    centroids[(batch, subject.name, side)][structure_upper] = np.asarray(feat["centroid_mm"])
                except Exception as exc:
                    errors.append([batch, subject.name, side, structure, file_path.name, "basic_morphometry", type(exc).__name__, str(exc)])
                    continue
                if structure_upper in CANAL_STRUCTURES:
                    try:
                        canal = centerline_features(binary, spacing, affine)
                        centerline_rows.append([
                            batch, subject.name, side, structure, rel, "pass", "",
                            *[canal[k] for k in centerline_keys],
                        ])
                        normals[(batch, subject.name, side)][structure_upper] = np.asarray(canal["plane_normal"])
                    except Exception as exc:
                        centerline_rows.append([batch, subject.name, side, structure, rel, "failed", f"{type(exc).__name__}: {exc}", *([None] * len(centerline_keys))])
                        errors.append([batch, subject.name, side, structure, file_path.name, "centerline", type(exc).__name__, str(exc)])

    distance_rows = []
    for key, structure_centroids in sorted(centroids.items()):
        for a, b in itertools.combinations(sorted(structure_centroids), 2):
            distance_rows.append([*key, a, b, float(np.linalg.norm(structure_centroids[a] - structure_centroids[b]))])

    angle_rows = []
    for key, structure_normals in sorted(normals.items()):
        for a, b in itertools.combinations(sorted(structure_normals), 2):
            angle_rows.append([*key, a, b, _angle_degrees(structure_normals[a], structure_normals[b])])

    asymmetry_rows = []
    by_subject: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        batch, subject, side, structure = row[:4]
        values = dict(zip(feature_keys, row[5:]))
        by_subject[(batch, subject, structure.upper())][side] = values
    for key, sides in sorted(by_subject.items()):
        if not {"L", "R"}.issubset(sides):
            continue
        for metric in ("volume_mm3", "surface_area_mm2", "maximum_3d_diameter_mm"):
            left, right = float(sides["L"][metric]), float(sides["R"][metric])
            mean = (abs(left) + abs(right)) / 2
            asymmetry_rows.append([*key, metric, left, right, abs(left - right) / mean if mean else None])

    feature_headers = ["batch", "seg_subject_id", "ear_side", "structure", "relative_path", *feature_keys]
    centerline_headers = ["batch", "seg_subject_id", "ear_side", "structure", "relative_path", "centerline_status", "failure_reason", *centerline_keys]
    error_headers = ["batch", "seg_subject_id", "ear_side", "structure", "file", "analysis_step", "error_type", "message"]
    output = paths.output_root / "03_morphometry"
    write_xlsx(
        output / "morphometry_features.xlsx",
        {
            "features": (feature_headers, rows),
            "canal_centerlines": (centerline_headers, centerline_rows),
            "interstructure_distances": (["batch", "seg_subject_id", "ear_side", "structure_a", "structure_b", "centroid_distance_mm"], distance_rows),
            "canal_plane_angles": (["batch", "seg_subject_id", "ear_side", "canal_a", "canal_b", "acute_plane_angle_degrees"], angle_rows),
            "bilateral_asymmetry": (["batch", "seg_subject_id", "structure", "metric", "left_value", "right_value", "relative_absolute_difference"], asymmetry_rows),
            "errors": (error_headers, errors),
        },
    )
    centerline_pass = sum(row[5] == "pass" for row in centerline_rows)
    log.info(
        "features=%d basic_errors=%d centerlines=%d centerline_pass=%d total_error_records=%d",
        len(rows), sum(row[5] == "basic_morphometry" for row in errors), len(centerline_rows), centerline_pass, len(errors),
    )
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
