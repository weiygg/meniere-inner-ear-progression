
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.morphology import skeletonize_3d, ball, binary_closing
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# =========================
# 1. 配置区：按你文件名风格改这里即可
# =========================
DATA_DIR = Path(os.environ.get("INNER_EAR_DATA_DIR", "data/segmentations"))
OUT_DIR = DATA_DIR / "analysis_out"
OUT_DIR.mkdir(exist_ok=True)

STRUCTS_CANALS = ["SSC", "HSC", "PSC"]          # 半规管（用于角度）
STRUCTS_VOLUME = ["Cochlear", "Vestibular", "SSC", "HSC", "PSC", "TV", "ELS"]  # 体积统计
# 你的命名类似：068L_SSC.nii.gz / 068R_HSC.nii.gz
FNAME_PATTERN = re.compile(r"(?P<pid>\d+)(?P<side>[LR])_(?P<struct>[A-Za-z]+)\.nii(\.gz)?$")

# 清理参数（可根据 mask 质量微调）
MIN_COMPONENT_VOXELS = 500   # 去掉非常小的连通域噪声（按体素）
CLOSING_RADIUS = 1           # 形态学闭运算半径（1~2 常用）


# =========================
# 2. I/O 与预处理
# =========================
def load_mask_nii(fp: Path):
    img = nib.load(str(fp))
    # 统一到接近标准的轴向方向（减少 orientation 带来的混乱）
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    mask = (data > 0.5).astype(np.uint8)
    # 体素间距（mm）
    zooms = img.header.get_zooms()[:3]
    return mask, zooms, img.affine

def keep_largest_component(mask: np.ndarray, min_voxels=MIN_COMPONENT_VOXELS):
    """保留最大连通域，同时也允许“最大域太小”时保留原mask（避免误删）。"""
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

def clean_mask(mask: np.ndarray, closing_radius=CLOSING_RADIUS, min_voxels=MIN_COMPONENT_VOXELS):
    """闭运算平滑 + 保留最大连通域，适合半规管这种细长结构。"""
    if mask.sum() == 0:
        return mask
    selem = ball(closing_radius)
    mask2 = binary_closing(mask.astype(bool), selem).astype(np.uint8)
    mask3 = keep_largest_component(mask2, min_voxels=min_voxels)
    return mask3


# =========================
# 3. 骨架与平面拟合（PCA）
# =========================
def skeleton_points(mask: np.ndarray):
    """3D骨架化并返回骨架点坐标（体素坐标系）。"""
    if mask.sum() == 0:
        return None
    sk = skeletonize_3d(mask.astype(bool))
    pts = np.argwhere(sk > 0)
    if pts.shape[0] < 20:
        return None
    return pts

def fit_plane_pca(points: np.ndarray):
    """
    PCA拟合平面：
    - 点云中心 c
    - 法向量 n：最小方差方向（第三主成分）
    """
    c = points.mean(axis=0)
    X = points - c
    # SVD: X = U S Vt，Vt 的最后一行是最小方差方向
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n

def angle_between_normals(n1, n2):
    """返回 0~90 度夹角（因为平面法向量正负号等价）。"""
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    cos = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return float(ang)

def volume_mm3(mask: np.ndarray, zooms):
    voxel_vol = float(zooms[0] * zooms[1] * zooms[2])
    return float(mask.sum() * voxel_vol)


# =========================
# 4. 可视化：骨架点云+平面+法向量
# =========================
def plot_skeleton_and_plane(points, center, normal, title, save_path: Path, plane_size=30):
    """
    3D图：骨架点（散点）+ 拟合平面（半透明网格）+ 法向量箭头
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 骨架点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.8)

    # 构造平面网格：在平面内找两个正交方向 u,v
    # 取一个与normal不共线的向量 a，然后 u = n×a，v = n×u
    n = normal
    a = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    grid = np.linspace(-plane_size, plane_size, 10)
    U, V = np.meshgrid(grid, grid)
    plane_pts = center[None, None, :] + U[..., None] * u[None, None, :] + V[..., None] * v[None, None, :]
    Xp, Yp, Zp = plane_pts[..., 0], plane_pts[..., 1], plane_pts[..., 2]
    ax.plot_surface(Xp, Yp, Zp, alpha=0.25, linewidth=0)

    # 法向量箭头
    ax.quiver(center[0], center[1], center[2], n[0], n[1], n[2],
              length=plane_size * 0.8, normalize=True)

    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("k")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_angle_distributions(df_angles: pd.DataFrame, save_path: Path):
    """角度分布图：直方+箱线（简单但发表友好）。"""
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(df_angles["angle_deg"].dropna().values, bins=20)
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Count")
    ax1.set_title("Angle distribution")

    ax2 = fig.add_subplot(1, 2, 2)
    # 以角度类型分组的箱线图
    groups = []
    labels = []
    for t, sub in df_angles.groupby("angle_type"):
        groups.append(sub["angle_deg"].dropna().values)
        labels.append(t)
    ax2.boxplot(groups, labels=labels, vert=True)
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_title("By angle type")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_volume_distributions(df_vol: pd.DataFrame, save_path: Path):
    """体积分布图：按结构分组箱线图。"""
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    groups = []
    labels = []
    for s, sub in df_vol.groupby("struct"):
        groups.append(sub["volume_mm3"].dropna().values)
        labels.append(s)
    ax.boxplot(groups, labels=labels, vert=True)
    ax.set_ylabel("Volume (mm³)")
    ax.set_title("Volume distributions by structure")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =========================
# 5. 批处理主流程
# =========================
def parse_files(data_dir: Path):
    items = []
    for fp in data_dir.rglob("*.nii*"):   # ← 关键修改
        m = FNAME_PATTERN.match(fp.name)
        if not m:
            continue
        items.append({
            "pid": m.group("pid"),
            "side": m.group("side"),
            "struct": m.group("struct"),
            "path": fp
        })
    return pd.DataFrame(items)


def main():
    df_files = parse_files(DATA_DIR)
    if df_files.empty:
        raise RuntimeError("未匹配到任何符合命名规则的nii文件，请检查FNAME_PATTERN与文件名。")

    # --- 体积结果表 ---
    vol_rows = []
    # --- 平面法向量与拟合质量（可扩展） ---
    plane_rows = []

    # --- 每个(pid, side, struct)处理 ---
    for (pid, side), sub in df_files.groupby(["pid", "side"]):
        # 先做体积统计
        for struct in STRUCTS_VOLUME:
            rec = sub[sub["struct"].str.lower() == struct.lower()]
            if rec.empty:
                continue
            fp = rec.iloc[0]["path"]
            mask, zooms, _ = load_mask_nii(fp)
            mask = clean_mask(mask)
            vol_rows.append({
                "pid": pid, "side": side, "struct": struct,
                "volume_mm3": volume_mm3(mask, zooms)
            })

        # 再做半规管：骨架->平面
        normals = {}
        centers = {}
        for struct in STRUCTS_CANALS:
            rec = sub[sub["struct"].str.lower() == struct.lower()]
            if rec.empty:
                continue
            fp = rec.iloc[0]["path"]
            mask, zooms, _ = load_mask_nii(fp)
            mask = clean_mask(mask)
            pts = skeleton_points(mask)
            if pts is None:
                continue

            # 将体素坐标换成“毫米坐标”可更物理（推荐）
            pts_mm = pts.astype(float) * np.array(zooms)[None, :]
            c, n = fit_plane_pca(pts_mm)
            normals[struct] = n
            centers[struct] = c

            plane_rows.append({
                "pid": pid, "side": side, "struct": struct,
                "nx": n[0], "ny": n[1], "nz": n[2],
                "cx": c[0], "cy": c[1], "cz": c[2],
                "n_points": pts_mm.shape[0]
            })

            # 3D图：每个半规管一张
            fig_name = f"{pid}{side}_{struct}_skeleton_plane.png"
            plot_skeleton_and_plane(
                points=pts_mm, center=c, normal=n,
                title=f"{pid}{side} {struct}: skeleton + plane",
                save_path=OUT_DIR / fig_name,
                plane_size=20
            )

        # 也可以为每个耳朵额外画一张：三个法向量一起画（先略，避免太长）

    df_vol = pd.DataFrame(vol_rows)
    df_plane = pd.DataFrame(plane_rows)
    df_vol.to_csv(OUT_DIR / "volumes_mm3.csv", index=False)
    df_plane.to_csv(OUT_DIR / "plane_normals.csv", index=False)

    # --- 计算三半规管平面夹角 ---
    # 需要同一(pid, side)同时具备 SSC/HSC/PSC 的 normal
    angle_rows = []
    for (pid, side), sub in df_plane.groupby(["pid", "side"]):
        nmap = {r["struct"]: np.array([r["nx"], r["ny"], r["nz"]]) for _, r in sub.iterrows()}
        if all(k in nmap for k in ["SSC", "HSC", "PSC"]):
            angle_rows.append({"pid": pid, "side": side, "angle_type": "SSC-HSC",
                               "angle_deg": angle_between_normals(nmap["SSC"], nmap["HSC"])})
            angle_rows.append({"pid": pid, "side": side, "angle_type": "SSC-PSC",
                               "angle_deg": angle_between_normals(nmap["SSC"], nmap["PSC"])})
            angle_rows.append({"pid": pid, "side": side, "angle_type": "HSC-PSC",
                               "angle_deg": angle_between_normals(nmap["HSC"], nmap["PSC"])})

    df_angles = pd.DataFrame(angle_rows)
    df_angles.to_csv(OUT_DIR / "canal_plane_angles_deg.csv", index=False)

    # --- 汇总图 ---
    if not df_angles.empty:
        plot_angle_distributions(df_angles, OUT_DIR / "angles_summary.png")
    if not df_vol.empty:
        plot_volume_distributions(df_vol, OUT_DIR / "volumes_summary.png")

    print("Done. Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
