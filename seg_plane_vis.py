# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

from skimage.morphology import skeletonize_3d, ball, binary_closing
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt


# ============================================================
# 0. 配置区（你只需要改 DATA_DIR）
# ============================================================
DATA_DIR = Path(os.environ.get("INNER_EAR_DATA_DIR", "data/segmentations"))
OUT_DIR = DATA_DIR / "analysis_out"
OUT_DIR.mkdir(exist_ok=True)

# 半规管（用于平面拟合与夹角）
STRUCTS_CANALS = ["SSC", "HSC", "PSC"]

# 体积统计结构（你有啥就写啥；缺了就自动跳过）
STRUCTS_VOLUME = ["Cochlear", "Vestibular", "SSC", "HSC", "PSC", "TV", "ELS"]

# 文件名匹配：例如 068L_SSC.nii.gz / 068R_Cochlear.nii.gz
# 注意：只用文件名，不依赖父目录名（sub068 不影响）
FNAME_PATTERN = re.compile(r"(?P<pid>\d+)(?P<side>[LR])_(?P<struct>[A-Za-z]+)\.nii(\.gz)?$", re.IGNORECASE)

# mask 清理参数（按你数据质量可调）
MIN_COMPONENT_VOXELS = 200     # 太小的连通域视为噪声（半规管体素不多时可再调小）
CLOSING_RADIUS = 1             # 形态学闭运算半径（1~2常用）
SKELETON_MIN_POINTS = 30       # 骨架点少于该值则跳过（避免失败的拟合）

# 可视化参数
PLANE_SIZE_MM = 20.0           # 平面网格大小（mm）
SUBSAMPLE_POINTS = 2500        # 骨架点云下采样（避免太密）
RANDOM_SEED = 42


# ============================================================
# 1. 基础函数：读 NIfTI / 清理 mask / 体积
# ============================================================
def load_mask_nii(fp: Path):
    """
    读取 NIfTI，转为接近标准方向，并输出：
    mask: uint8 0/1
    zooms: (sx, sy, sz) mm
    affine: 4x4
    """
    img = nib.load(str(fp))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    mask = (data > 0.5).astype(np.uint8)
    zooms = img.header.get_zooms()[:3]
    return mask, zooms, img.affine


def keep_largest_component(mask: np.ndarray, min_voxels: int = MIN_COMPONENT_VOXELS):
    """保留最大连通域；最大域太小则不强删（避免误删掉细结构）。"""
    lab = label(mask)
    if lab.max() == 0:
        return mask
    props = regionprops(lab)
    props = sorted(props, key=lambda r: r.area, reverse=True)
    largest = props[0]
    if largest.area < min_voxels:
        return mask
    out = (lab == largest.label).astype(np.uint8)
    return out


def clean_mask(mask: np.ndarray, closing_radius: int = CLOSING_RADIUS, min_voxels: int = MIN_COMPONENT_VOXELS):
    """
    对半规管这类细长结构常见处理：
    - 闭运算：填补小孔洞/断裂（过大可能会糊掉细管）
    - 保留最大连通域：去掉小噪声
    """
    if mask.sum() == 0:
        return mask
    selem = ball(closing_radius)
    mask2 = binary_closing(mask.astype(bool), selem).astype(np.uint8)
    mask3 = keep_largest_component(mask2, min_voxels=min_voxels)
    return mask3


def volume_mm3(mask: np.ndarray, zooms):
    """体积 mm^3"""
    voxel_vol = float(zooms[0] * zooms[1] * zooms[2])
    return float(mask.sum() * voxel_vol)


# ============================================================
# 2. 骨架、平面拟合、角度
# ============================================================
def skeleton_points_mm(mask: np.ndarray, zooms):
    """
    3D skeletonize -> 骨架点（物理坐标 mm）
    返回 (N,3) float
    """
    if mask.sum() == 0:
        return None
    sk = skeletonize_3d(mask.astype(bool))
    pts = np.argwhere(sk > 0)
    if pts.shape[0] < SKELETON_MIN_POINTS:
        return None
    pts_mm = pts.astype(float) * np.array(zooms)[None, :]
    return pts_mm


def fit_plane_pca(points_mm: np.ndarray):
    """
    PCA/SVD 拟合平面：
    center c = mean(points)
    normal n = 最小方差方向（第三主成分）
    """
    c = points_mm.mean(axis=0)
    X = points_mm - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n


def angle_between_normals(n1, n2):
    """
    平面法向量夹角，输出 0~90°（法向量正负等价，取 abs(dot)）
    """
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    cos = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ============================================================
# 3. 3D 绘图辅助：等比例坐标轴
# ============================================================
def set_axes_equal(ax):
    """让 3D 坐标轴等比例，避免视觉误导。"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    r = 0.5 * max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - r, x_middle + r])
    ax.set_ylim3d([y_middle - r, y_middle + r])
    ax.set_zlim3d([z_middle - r, z_middle + r])


# ============================================================
# 4. 可视化：单半规管（骨架+平面+法向量）
# ============================================================
def plot_skeleton_and_plane(points_mm, center, normal, title, save_path: Path, plane_size=PLANE_SIZE_MM):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points_mm[:, 0], points_mm[:, 1], points_mm[:, 2], s=2, alpha=0.8)

    n = normal / (np.linalg.norm(normal) + 1e-12)
    a = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v = v / (np.linalg.norm(v) + 1e-12)

    grid = np.linspace(-plane_size, plane_size, 10)
    U, V = np.meshgrid(grid, grid)
    plane_pts = center[None, None, :] + U[..., None] * u[None, None, :] + V[..., None] * v[None, None, :]
    Xp, Yp, Zp = plane_pts[..., 0], plane_pts[..., 1], plane_pts[..., 2]
    ax.plot_surface(Xp, Yp, Zp, alpha=0.25, linewidth=0)

    ax.quiver(center[0], center[1], center[2], n[0], n[1], n[2],
              length=plane_size * 0.9, normalize=True)

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.view_init(elev=20, azim=35)
    set_axes_equal(ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================================================
# 5. 可视化：每个耳朵一张（SSC/HSC/PSC 三平面+三法向量）
# ============================================================
def plot_three_canals_planes(canal_data: dict, title: str, save_path: Path,
                             plane_size=PLANE_SIZE_MM, subsample_points=SUBSAMPLE_POINTS):
    """
    canal_data:
      { "SSC": {"pts": pts_mm, "c": c, "n": n}, "HSC": {...}, "PSC": {...} }
    """
    rng = np.random.default_rng(RANDOM_SEED)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for struct, d in canal_data.items():
        pts = d["pts"]
        if pts is None or pts.shape[0] == 0:
            continue

        # 下采样点云避免太密
        if pts.shape[0] > subsample_points:
            idx = rng.choice(pts.shape[0], subsample_points, replace=False)
            pts_plot = pts[idx]
        else:
            pts_plot = pts

        ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], s=2, alpha=0.75, label=struct)

        c = d["c"]
        n = d["n"]
        n = n / (np.linalg.norm(n) + 1e-12)

        # 构造平面内正交基 u,v
        a = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(a, n)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a); u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u); v = v / (np.linalg.norm(v) + 1e-12)

        grid = np.linspace(-plane_size, plane_size, 10)
        U, V = np.meshgrid(grid, grid)
        plane_pts = c[None, None, :] + U[..., None] * u[None, None, :] + V[..., None] * v[None, None, :]
        Xp, Yp, Zp = plane_pts[..., 0], plane_pts[..., 1], plane_pts[..., 2]
        ax.plot_surface(Xp, Yp, Zp, alpha=0.18, linewidth=0)

        ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2],
                  length=plane_size * 0.9, normalize=True)

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend(loc="upper right")
    ax.view_init(elev=20, azim=35)
    set_axes_equal(ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================================================
# 6. 汇总图：角度分布 / 体积分布
# ============================================================
def plot_angle_distributions(df_angles: pd.DataFrame, save_path: Path):
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(df_angles["angle_deg"].dropna().values, bins=20)
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Count")
    ax1.set_title("Angle distribution")

    ax2 = fig.add_subplot(1, 2, 2)
    groups, labels_ = [], []
    for t, sub in df_angles.groupby("angle_type"):
        groups.append(sub["angle_deg"].dropna().values)
        labels_.append(t)
    ax2.boxplot(groups, labels=labels_, vert=True)
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_title("By angle type")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_volume_distributions(df_vol: pd.DataFrame, save_path: Path):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    groups, labels_ = [], []
    for s, sub in df_vol.groupby("struct"):
        groups.append(sub["volume_mm3"].dropna().values)
        labels_.append(s)
    ax.boxplot(groups, labels=labels_, vert=True)
    ax.set_ylabel("Volume (mm³)")
    ax.set_title("Volume distributions by structure")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================================================
# 7. 扫描文件（递归 rglob）并解析 pid/side/struct
# ============================================================
def parse_files(data_dir: Path):
    items = []
    for fp in data_dir.rglob("*.nii*"):  # 关键：递归子文件夹（sub068/...）
        m = FNAME_PATTERN.match(fp.name)
        if not m:
            continue
        pid = m.group("pid")
        side = m.group("side").upper()
        struct = m.group("struct")
        items.append({
            "pid": pid,
            "side": side,
            "struct": struct,
            "path": fp
        })
    return pd.DataFrame(items)


# ============================================================
# 8. 主流程
# ============================================================
def main():
    df_files = parse_files(DATA_DIR)
    if df_files.empty:
        raise RuntimeError("未匹配到任何符合命名规则的nii文件，请检查文件名是否形如 068L_SSC.nii.gz，并确认目录正确。")

    # 调试：看看解析是否正确
    print("Matched files:", len(df_files))
    print(df_files.head(10))

    vol_rows = []
    plane_rows = []

    # 按每个耳朵(pid, side)处理
    for (pid, side), sub in df_files.groupby(["pid", "side"]):
        # -------------------------
        # 8.1 体积统计
        # -------------------------
        for struct in STRUCTS_VOLUME:
            rec = sub[sub["struct"].str.lower() == struct.lower()]
            if rec.empty:
                continue
            fp = Path(rec.iloc[0]["path"])
            mask, zooms, _ = load_mask_nii(fp)
            mask = clean_mask(mask)
            vol_rows.append({
                "pid": pid, "side": side, "struct": struct,
                "volume_mm3": volume_mm3(mask, zooms),
                "file": str(fp)
            })

        # -------------------------
        # 8.2 半规管骨架 -> 平面拟合 -> 单结构图 + 汇总图
        # -------------------------
        canal_data = {}  # 存 SSC/HSC/PSC 的 pts/c/n，用于每耳一张同图

        for struct in STRUCTS_CANALS:
            rec = sub[sub["struct"].str.lower() == struct.lower()]
            if rec.empty:
                continue

            fp = Path(rec.iloc[0]["path"])
            mask, zooms, _ = load_mask_nii(fp)
            mask = clean_mask(mask)
            pts_mm = skeleton_points_mm(mask, zooms)
            if pts_mm is None:
                continue

            c, n = fit_plane_pca(pts_mm)
            canal_data[struct] = {"pts": pts_mm, "c": c, "n": n}

            plane_rows.append({
                "pid": pid, "side": side, "struct": struct,
                "nx": float(n[0]), "ny": float(n[1]), "nz": float(n[2]),
                "cx": float(c[0]), "cy": float(c[1]), "cz": float(c[2]),
                "n_points": int(pts_mm.shape[0]),
                "file": str(fp)
            })

            # 单半规管图：骨架+平面+法向量
            fig_name = f"{pid}{side}_{struct}_skeleton_plane.png"
            plot_skeleton_and_plane(
                points_mm=pts_mm, center=c, normal=n,
                title=f"{pid}{side} {struct}: skeleton + plane + normal",
                save_path=OUT_DIR / fig_name
            )

        # 每耳一张：三半规管同图（至少2个存在就画；最好3个齐全）
        if len(canal_data) >= 2:
            fig_name = f"{pid}{side}_3canals_planes_normals.png"
            plot_three_canals_planes(
                canal_data=canal_data,
                title=f"{pid}{side} | SSC/HSC/PSC planes + normals",
                save_path=OUT_DIR / fig_name
            )

    # -------------------------
    # 8.3 保存表格
    # -------------------------
    df_vol = pd.DataFrame(vol_rows)
    df_plane = pd.DataFrame(plane_rows)

    df_vol.to_csv(OUT_DIR / "volumes_mm3.csv", index=False, encoding="utf-8-sig")
    df_plane.to_csv(OUT_DIR / "plane_normals.csv", index=False, encoding="utf-8-sig")

    # -------------------------
    # 8.4 计算三半规管平面夹角
    # -------------------------
    angle_rows = []
    for (pid, side), subp in df_plane.groupby(["pid", "side"]):
        nmap = {r["struct"].upper(): np.array([r["nx"], r["ny"], r["nz"]], dtype=float) for _, r in subp.iterrows()}
        if all(k in nmap for k in ["SSC", "HSC", "PSC"]):
            angle_rows.append({"pid": pid, "side": side, "angle_type": "SSC-HSC",
                               "angle_deg": angle_between_normals(nmap["SSC"], nmap["HSC"])})
            angle_rows.append({"pid": pid, "side": side, "angle_type": "SSC-PSC",
                               "angle_deg": angle_between_normals(nmap["SSC"], nmap["PSC"])})
            angle_rows.append({"pid": pid, "side": side, "angle_type": "HSC-PSC",
                               "angle_deg": angle_between_normals(nmap["HSC"], nmap["PSC"])})

    df_angles = pd.DataFrame(angle_rows)
    df_angles.to_csv(OUT_DIR / "canal_plane_angles_deg.csv", index=False, encoding="utf-8-sig")

    # -------------------------
    # 8.5 汇总图
    # -------------------------
    if not df_angles.empty:
        plot_angle_distributions(df_angles, OUT_DIR / "angles_summary.png")

    if not df_vol.empty:
        plot_volume_distributions(df_vol, OUT_DIR / "volumes_summary.png")

    print("\nDone.")
    print("All outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
