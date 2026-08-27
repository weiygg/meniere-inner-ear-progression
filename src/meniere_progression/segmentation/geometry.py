from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


GEOMETRY_ALGORITHM_VERSION = "protocol-v2-geometry-2"
NEIGHBOUR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _sparse_components(coordinates: np.ndarray) -> list[set[tuple[int, int, int]]]:
    remaining = {tuple(int(value) for value in row) for row in coordinates}
    components: list[set[tuple[int, int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            coordinate = stack.pop()
            for offset in NEIGHBOUR_OFFSETS:
                neighbour = tuple(coordinate[index] + offset[index] for index in range(3))
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _skeleton_graph_length_mm(
    skeleton: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    coordinates = {tuple(int(value) for value in row) for row in np.argwhere(skeleton)}
    offsets = [offset for offset in NEIGHBOUR_OFFSETS if offset > (0, 0, 0)]
    spacing_array = np.asarray(spacing, dtype=float)
    length = 0.0
    for coordinate in coordinates:
        for offset in offsets:
            neighbour = tuple(coordinate[index] + offset[index] for index in range(3))
            if neighbour in coordinates:
                length += float(np.linalg.norm(np.asarray(offset) * spacing_array))
    return length


def mask_geometry(mask: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    binary = np.asarray(mask, dtype=bool)
    voxel_volume = float(np.prod(spacing))
    coordinates = np.argwhere(binary)
    if not len(coordinates):
        return {
            "volume_mm3": 0.0,
            "component_count": 0.0,
            "centerline_voxel_count": 0.0,
            "centerline_length_mm": 0.0,
        }
    components = _sparse_components(coordinates)
    centerline_voxel_count = 0
    centerline_length_mm = 0.0
    for component in components:
        component_coordinates = np.asarray(sorted(component), dtype=int)
        lower = component_coordinates.min(axis=0)
        local = component_coordinates - lower
        shape = tuple(int(value) for value in local.max(axis=0) + 1)
        cropped = np.zeros(shape, dtype=bool)
        cropped[tuple(local.T)] = True
        skeleton = skeletonize(cropped)
        centerline_voxel_count += int(skeleton.sum())
        centerline_length_mm += _skeleton_graph_length_mm(skeleton, spacing)
    return {
        "volume_mm3": float(binary.sum()) * voxel_volume,
        "component_count": float(len(components)),
        "centerline_voxel_count": float(centerline_voxel_count),
        "centerline_length_mm": centerline_length_mm,
    }


def plane_normal(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Return the unoriented least-variance PCA normal in NIfTI physical coordinates."""
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if len(coordinates) < 3:
        return np.full(3, np.nan)
    homogeneous = np.column_stack([coordinates, np.ones(len(coordinates))])
    physical = (np.asarray(affine, dtype=float) @ homogeneous.T).T[:, :3]
    covariance = np.cov(physical - physical.mean(axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, int(np.argmin(values))]
    norm = np.linalg.norm(normal)
    return normal / norm if norm else np.full(3, np.nan)


def inter_canal_angle_degrees(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    """Unsigned plane angle (0-90 degrees), invariant to normal sign and reflection."""
    normal_a = np.asarray(normal_a, dtype=float)
    normal_b = np.asarray(normal_b, dtype=float)
    if not np.isfinite(normal_a).all() or not np.isfinite(normal_b).all():
        return float("nan")
    cosine = float(np.clip(abs(np.dot(normal_a, normal_b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
