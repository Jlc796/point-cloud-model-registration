import numpy as np
import open3d as o3d
from typing import Tuple


def estimate_voxel_size(
    pcd: o3d.geometry.PointCloud,
    sample_n: int = 8000,
    scale: float = 2.5,
    seed: int = 0,
    min_voxel: float = 0.05
) -> Tuple[float, float]:
    """
    Estimate voxel size from median nearest-neighbor distance.

    Returns:
        voxel (float): suggested voxel size in meters
        median_nn (float): median nearest-neighbor distance in meters
    """
    if pcd.is_empty():
        raise ValueError("Point cloud is empty.")

    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < 50:
        raise ValueError("Point cloud too small for voxel estimation.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_n, n), replace=False)

    kd = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in idx:
        _, _, nn_d2 = kd.search_knn_vector_3d(pts[i], 2)  # itself + nearest
        if len(nn_d2) == 2:
            dists.append(float(np.sqrt(nn_d2[1])))

    if not dists:
        raise RuntimeError("Failed to compute nearest-neighbor distances.")

    median_nn = float(np.median(dists))
    voxel = max(min_voxel, median_nn * scale)
    return voxel, median_nn


if __name__ == "__main__":
    # quick test
    p = r"F:\data\model_3_walls.pcd"
    pc = o3d.io.read_point_cloud(p)
    v, med = estimate_voxel_size(pc)
    print(f"median_nn={med:.4f} m -> voxel={v:.3f} m")
