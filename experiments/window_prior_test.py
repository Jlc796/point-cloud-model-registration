import copy
import numpy as np
import open3d as o3d

# ============================================================
# 0) Paths
# ============================================================
MODEL_PATH = r"F:\data\building_lod3.pcd"  # CityGML wall point cloud (UTM/global)
MLS_PATH = r"F:\data\pointCloud_surroundings.pcd"  # TLS scan

# ============================================================
# 1) CloudCompare-like global shift (KNOWN from your CC screenshot)
#    CC "Global shift": (-691000.00; -5336000.00; 0.00)
#    We apply it to the *target only* (CityGML) to bring it into local frame.
# ============================================================
CC_SHIFT = np.array([-691000.0, -5336000.0, 0.0], dtype=float)

# ============================================================
# 2) Determinism
# ============================================================
SEED = 42

# ============================================================
# 3) Downsample / normals
# ============================================================
VOXEL = 0.50
NORMAL_RADIUS = 3.0 * VOXEL
NORMAL_MAX_NN = 60

# ============================================================
# 4) Plane gating via normal clustering
#      - estimate normals for target_ds
#      - keep vertical-ish points (|nz| < VERTICAL_NZ_MAX)
#      - cluster normals into 2 groups (two major wall directions)
# ============================================================
VERTICAL_NZ_MAX = 0.35  # wall-like
K_CLUSTERS = 2

# ============================================================
# 4.5) Window-edge / feature-rich target weighting
# ============================================================
USE_FEATURE_RICH_TARGET = False

FEATURE_RADIUS = 1.2       # meters, local neighborhood for edge-ness
FEATURE_MAX_NN = 40
FEATURE_TOP_RATIO = 0.25   # keep top 25% most non-planar wall points

# ============================================================
# 5) Weighted P2L ICP parameters (multi-stage)
# ============================================================
STAGES = [
    (8.0, 40),
    (5.0, 40),
    (3.0, 40),
    (1.5, 40),
    (0.9, 50),
    (0.6, 60),
]

# robust kernel (Huber)
HUBER_K = 0.30  # meters; residual beyond k gets down-weighted

# density balancing on source
DENSITY_VOXEL = 0.60  # for counting points per cell
DENSITY_POWER = 1.0  # weight ~ 1 / (count^power)

# correspondence normal compatibility
MAX_NORMAL_ANGLE_DEG = 30.0

# initial alignment helper
USE_CENTER_INIT_XY = True
USE_GROUND_INIT_Z = True
GROUND_Q = 0.05  # 5% lowest points as ground proxy (robust)

# CROP parameters
CROP_PAD = 2.0  # meters padding around source bbox


# ============================================================
# 6) Spatial clustering inside each normal cluster
# ============================================================
DBSCAN_EPS = 2.5
DBSCAN_MIN_POINTS = 30
MIN_FACADE_POINTS = 150

# ============================================================
# 7) Source main façade extraction (for candidate-specific init)
# ============================================================
SRC_VERTICAL_NZ_MAX = 0.35
SRC_DBSCAN_EPS = 2.0
SRC_DBSCAN_MIN_POINTS = 80
MIN_SOURCE_FACADE_POINTS = 300

# ============================================================
# 8) Facing prior (soft geometric prior)
# ============================================================
USE_FACING_PRIOR = True
FACING_PRIOR_WEIGHT = 0.08

# ============================================================
# Utils
# ============================================================
def set_seed(seed: int):
    np.random.seed(seed)
    try:
        o3d.utility.random.seed(seed)
    except Exception:
        pass


def T_translate(txyz):
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(txyz, dtype=float)
    return T


def draw_two(src, tgt, T=np.eye(4), title="view", recenter=True):
    s = copy.deepcopy(src)
    t = copy.deepcopy(tgt)
    s.paint_uniform_color([1, 0, 0])
    t.paint_uniform_color([0, 0, 1])
    s.transform(T)
    if recenter:
        c = 0.5 * (s.get_center() + t.get_center())
        s.translate(-c)
        t.translate(-c)
    o3d.visualization.draw([s, t], title=title)


def stats(pcd, name):
    pts = np.asarray(pcd.points)
    bb = pcd.get_axis_aligned_bounding_box()
    print(f"{name}: n={len(pts):,}")
    print(f"  center={pcd.get_center()}")
    print(f"  bbox min={bb.min_bound}")
    print(f"  bbox max={bb.max_bound}")


# NN distance diagnostic
def nn_distance_stats(source_pts, T, kd_tree, sample_n=5000, seed=42, name=""):
    """
    Sample some transformed source points, compute NN distances to target kd-tree,
    report quantiles + % within typical thresholds.
    """
    rng = np.random.default_rng(seed)
    Ns = source_pts.shape[0]
    if Ns == 0:
        print(f"[NN] {name}: empty source")
        return

    idx = rng.choice(Ns, size=min(sample_n, Ns), replace=False)
    P0 = source_pts[idx]

    R = T[:3, :3]
    t = T[:3, 3]
    P = (P0 @ R.T) + t

    dists = []
    for p in P:
        k, _, dist2 = kd_tree.search_knn_vector_3d(p, 1)
        if k < 1:
            continue
        dists.append(np.sqrt(dist2[0]))
    dists = np.asarray(dists, dtype=float)

    if dists.size == 0:
        print(f"[NN] {name}: no NN found")
        return

    qs = np.quantile(dists, [0.05, 0.25, 0.5, 0.75, 0.90, 0.95])
    print(f"[NN] {name}: n={dists.size}, "
          f"q05={qs[0]:.3f}, q25={qs[1]:.3f}, q50={qs[2]:.3f}, "
          f"q75={qs[3]:.3f}, q90={qs[4]:.3f}, q95={qs[5]:.3f}")

    for th in [0.45, 0.6, 0.9, 1.5, 3.0, 5.0, 8.0]:
        frac = np.mean(dists <= th) * 100
        print(f"      <= {th:.2f} m : {frac:.1f}%")


def voxel_counts(points, voxel):
    """Return per-point occupancy count in voxel cell."""
    ijk = np.floor(points / voxel).astype(np.int64)
    key = ijk[:, 0].astype(np.int64) * 73856093 ^ ijk[:, 1].astype(np.int64) * 19349663 ^ ijk[:, 2].astype(
        np.int64) * 83492791
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv].astype(np.float64)


def huber_weight(r, k):
    """Huber weight for scalar residual r."""
    a = np.abs(r)
    if a <= k:
        return 1.0
    return k / a


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=float)


def Rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def delta_T_4dof(dyaw, dtx, dty, dtz):
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rz(dyaw)
    T[:3, 3] = [dtx, dty, dtz]
    return T


def exp_so3(w):
    """Rodrigues exp for small rotation vector w."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def se3_update(T, xi):
    """Apply twist xi=[wx,wy,wz, tx,ty,tz] to T (left-multiply)."""
    w = xi[:3]
    t = xi[3:]
    dR = exp_so3(w)
    dT = np.eye(4)
    dT[:3, :3] = dR
    dT[:3, 3] = t
    return dT @ T


def angle_between_normals(n1, n2):
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    c = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def estimate_ground_z(points, q=0.05):
    """Robust ground proxy: quantile of z."""
    return float(np.quantile(points[:, 2], q))


def compute_local_nonplanarity(points, radius=1.2, max_nn=40):
    """
    For each point, compute a simple non-planarity score from neighborhood PCA.
    Larger score -> more edge / hole-boundary like.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kd = o3d.geometry.KDTreeFlann(pcd)

    scores = np.zeros(len(points), dtype=float)

    for i, p in enumerate(points):
        k, idx, _ = kd.search_radius_vector_3d(p, radius)
        if k < 6:
            scores[i] = 0.0
            continue

        if k > max_nn:
            idx = idx[:max_nn]

        nbrs = points[np.asarray(idx)]
        c = nbrs.mean(axis=0)
        X = nbrs - c

        cov = (X.T @ X) / max(1, len(nbrs) - 1)
        eigvals = np.linalg.eigvalsh(cov)   # ascending
        eigvals = np.maximum(eigvals, 1e-12)

        # small eigenvalue / total variance -> local non-planarity / edge-ness
        scores[i] = eigvals[0] / eigvals.sum()

    return scores


# ============================================================
# Spatial candidate splitting via DBSCAN
# ============================================================
def split_into_spatial_candidates(pts, nrms, eps=1.5, min_points=80, min_facade_points=300):
    """
    Cluster points spatially using DBSCAN (XY only) and return each
    sufficiently large cluster as a candidate dict with keys:
        "points", "normals", "size", "normal_cluster", "label"
    """
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts)

    labels = np.array(pcd_tmp.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    candidates = []
    for lbl in np.unique(labels):
        if lbl < 0:
            continue  # noise
        mask = labels == lbl
        if mask.sum() < min_facade_points:
            continue
        candidates.append({
            "points": pts[mask],
            "normals": nrms[mask],
            "size": int(mask.sum()),
            "normal_cluster": -1,   # filled in by caller
            "label": int(lbl),
        })
    return candidates


# ============================================================
# Source main façade extraction (for candidate-specific init)
# ============================================================
def extract_main_source_facade(src_ds, vertical_nz_max=0.35, eps=1.5, min_points=80, min_facade_points=300):
    """
    Extract the largest wall-like connected component from source cloud.
    Returns:
        facade_pts, facade_normals
    """
    src_tmp = copy.deepcopy(src_ds)
    src_tmp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN)
    )
    src_tmp.orient_normals_towards_camera_location(src_ds.get_center())

    src_pts = np.asarray(src_tmp.points)
    src_nrm = np.asarray(src_tmp.normals)

    # keep vertical-ish points
    wall_mask = np.abs(src_nrm[:, 2]) < vertical_nz_max
    src_wall_pts = src_pts[wall_mask]
    src_wall_nrm = src_nrm[wall_mask]

    if len(src_wall_pts) < min_facade_points:
        print("[WARNING] Too few vertical-ish source points; fallback to full source cloud.")
        return src_pts, src_nrm

    # cluster normals into 2 major directions
    V = src_wall_nrm.copy()
    flip = V[:, 2] < 0
    V[flip] *= -1.0
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    labels, centers = kmeans2_unit_vectors(V, iters=30)

    pts0 = src_wall_pts[labels == 0]
    nrm0 = src_wall_nrm[labels == 0]
    pts1 = src_wall_pts[labels == 1]
    nrm1 = src_wall_nrm[labels == 1]

    cands0 = split_into_spatial_candidates(
        pts0, nrm0,
        eps=eps,
        min_points=min_points,
        min_facade_points=min_facade_points
    )
    cands1 = split_into_spatial_candidates(
        pts1, nrm1,
        eps=eps,
        min_points=min_points,
        min_facade_points=min_facade_points
    )

    all_cands = cands0 + cands1
    all_cands.sort(key=lambda c: c["size"], reverse=True)

    if len(all_cands) == 0:
        print("[WARNING] Source façade clustering failed; fallback to all vertical-ish source points.")
        return src_wall_pts, src_wall_nrm

    best = all_cands[0]
    print(f"[SOURCE FACADE] selected main facade with {best['size']} pts")
    return best["points"], best["normals"]


# ============================================================
# Facing prior (soft geometric prior)
# ============================================================
def compute_facing_prior(src_facade_center, cand_pts, cand_normals):

    if len(cand_pts) < 10 or len(cand_normals) < 10:
        return 0.0

    cand_center = cand_pts.mean(axis=0)

    n = cand_normals.mean(axis=0).astype(float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return 0.0
    n = n / n_norm

    # direction from candidate wall center toward source façade center
    v = src_facade_center - cand_center
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return 0.0
    v = v / v_norm

    # If wall normal points toward source, dot is larger
    d = float(np.dot(n, v))

    # map [-1,1] -> [0,1]
    score = 0.5 * (d + 1.0)
    return score


# ============================================================
# Normal clustering (deterministic)
# ============================================================
def kmeans2_unit_vectors(V, iters=20):
    """
    Very small deterministic k-means for unit vectors (k=2).
    V: (N,3) unit vectors
    return labels (N,), centers (2,3)
    """
    assert V.shape[0] >= 10

    c0 = V[np.argmax(V[:, 0])]
    c1 = V[np.argmin(V[:, 0])]
    C = np.stack([c0, c1], axis=0)

    for _ in range(iters):
        dots = V @ C.T  # (N,2)
        lbl = np.argmax(dots, axis=1)
        C_new = []
        for k in [0, 1]:
            Vk = V[lbl == k]
            if Vk.shape[0] == 0:
                C_new.append(C[k])
            else:
                m = Vk.mean(axis=0)
                m = m / (np.linalg.norm(m) + 1e-12)
                C_new.append(m)
        C_new = np.stack(C_new, axis=0)
        if np.linalg.norm(C_new - C) < 1e-6:
            C = C_new
            break
        C = C_new

    return lbl, C


# ============================================================
# RANSAC Coarse Alignment (yaw + xy only)
# ============================================================
def restrict_to_yaw_xy(T):
    R = T[:3, :3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)

    T_new = np.eye(4, dtype=float)
    T_new[:3, :3] = R_yaw
    T_new[0, 3] = T[0, 3]
    T_new[1, 3] = T[1, 3]
    return T_new


def ransac_coarse_xy(src_ds, tgt_ds, voxel):
    """
    Lightweight RANSAC using FPFH features.
    """
    radius_normal = voxel * 2.0
    src_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    src_ds.normalize_normals()
    tgt_ds.normalize_normals()

    radius_feat = voxel * 5.0

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat, max_nn=100)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat, max_nn=100)
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_ds, tgt_ds,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel * 4.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 4.0)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 0.999)
    )

    T_xy = restrict_to_yaw_xy(result.transformation)
    return T_xy, result


# ============================================================
# Weighted Point-to-Plane ICP
# ============================================================
def solve_weighted_p2l_step_4dof(src_pts_transformed, tgt_pts_corr, tgt_nrm_corr, weights):

    H = np.zeros((4, 4), dtype=float)
    g = np.zeros(4, dtype=float)

    used = 0
    residuals = []

    for p, q, n, w in zip(src_pts_transformed, tgt_pts_corr, tgt_nrm_corr, weights):
        nx, ny, nz = n
        x, y, z = p

        r = float(np.dot(n, (p - q)))

        J = np.array([
            -nx * y + ny * x,   # d/d(yaw)
            nx,                 # d/d(tx)
            ny,                 # d/d(ty)
            nz                  # d/d(tz)
        ], dtype=float)

        H += w * np.outer(J, J)
        g += w * J * r
        residuals.append(r * r)
        used += 1

    if used < 4:
        return np.eye(4), used, 0.0

    try:
        delta = -np.linalg.solve(H + 1e-8 * np.eye(4), g)
    except np.linalg.LinAlgError:
        return np.eye(4), used, 0.0

    dyaw, dtx, dty, dtz = delta
    dT = delta_T_4dof(dyaw, dtx, dty, dtz)

    rms = float(np.sqrt(np.mean(residuals))) if residuals else 0.0
    return dT, used, rms


def weighted_p2l_icp(
        source_pts,  # (Ns,3) in current frame
        target_pts, target_normals,  # (Nt,3), (Nt,3)
        kd_tree,  # o3d KDTree built on target pcd
        T_init=np.eye(4),
        max_corr_dist=0.6,
        iters=20,
        src_density_weights=None,  # (Ns,)
        max_normal_angle_deg=30.0,
):

    T = T_init.copy()
    Ns = source_pts.shape[0]

    used = 0
    rms = 0.0

    for it_idx in range(iters):
        R = T[:3, :3]
        t = T[:3, 3]
        P = (source_pts @ R.T) + t  # (Ns,3) transformed source

        # --- build correspondences ---
        corr_src = []
        corr_tgt = []
        corr_nrm = []
        corr_w   = []

        for i in range(Ns):
            p = P[i]

            k, idx, dist2 = kd_tree.search_knn_vector_3d(p, 1)
            if k < 1:
                continue
            j = idx[0]
            d = float(np.sqrt(dist2[0]))
            if d > max_corr_dist:
                continue

            q = target_pts[j]
            n = target_normals[j]
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn

            r = float(n.dot(p - q))
            w = huber_weight(r, HUBER_K)
            if src_density_weights is not None:
                w *= float(src_density_weights[i])

            corr_src.append(p)
            corr_tgt.append(q)
            corr_nrm.append(n)
            corr_w.append(w)

        used = len(corr_src)

        if used < 20:
            print(f"    [ICP] break at iter {it_idx}: used={used} < 20 (th={max_corr_dist:.2f}m)")
            break

        # --- 4DOF solve ---
        dT, used, rms = solve_weighted_p2l_step_4dof(
            corr_src, corr_tgt, corr_nrm, corr_w
        )

        T = dT @ T

        # convergence check on translation + yaw magnitude
        dyaw = float(np.arctan2(dT[1, 0], dT[0, 0]))
        dt   = float(np.linalg.norm(dT[:3, 3]))
        if abs(dyaw) < 1e-6 and dt < 1e-6:
            break

    metrics = {
        "used_corr": used,
        "rms_p2l": rms,
    }
    return T, metrics


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    # --- load ---
    src_g = o3d.io.read_point_cloud(MLS_PATH)  # MLS (local)
    tgt_g = o3d.io.read_point_cloud(MODEL_PATH)  # CityGML (UTM/global)

    if src_g.is_empty() or tgt_g.is_empty():
        raise RuntimeError("Empty point cloud. Check paths.")

    print("=== RAW (Open3D read) ===")
    stats(src_g, "src (TLS)")
    stats(tgt_g, "tgt (CityGML UTM)")

    # --- build local target (CC-like shift) ---
    S = T_translate(CC_SHIFT)   # global -> local shift for target
    Si = T_translate(-CC_SHIFT) # local -> global

    tgt_l = copy.deepcopy(tgt_g)
    tgt_l.transform(S)

    src_l = copy.deepcopy(src_g)

    print("\n=== After shifting target to local (CC-like) ===")
    stats(src_l, "src_l")
    stats(tgt_l, "tgt_l (shifted)")

    # --- downsample ---
    src_ds = src_l.voxel_down_sample(VOXEL)
    tgt_ds = tgt_l.voxel_down_sample(VOXEL)

    print(f"\nDownsample VOXEL={VOXEL:.2f}m")
    print(f"src_ds: {len(src_ds.points):,}")
    print(f"tgt_ds: {len(tgt_ds.points):,}")

    # ============================================================
    # Extract main façade from source for candidate-specific init
    # ============================================================
    src_facade_pts, src_facade_nrm = extract_main_source_facade(
        src_ds,
        vertical_nz_max=SRC_VERTICAL_NZ_MAX,
        eps=SRC_DBSCAN_EPS,
        min_points=SRC_DBSCAN_MIN_POINTS,
        min_facade_points=MIN_SOURCE_FACADE_POINTS
    )
    src_facade_center = src_facade_pts.mean(axis=0)
    print(f"[SOURCE FACADE] center = {src_facade_center}")

    # ============================================================
    # RANSAC COARSE ALIGNMENT
    # ============================================================
    print("\n" + "=" * 60)
    print("=== RANSAC Coarse Alignment (yaw + xy only) ===")
    print("=" * 60)

    T_ransac, ransac_result = ransac_coarse_xy(src_ds, tgt_ds, VOXEL)

    print(f"RANSAC fitness: {ransac_result.fitness:.4f}")
    print(f"RANSAC inlier_rmse: {ransac_result.inlier_rmse:.4f}")
    print("T_ransac_local (3-DOF: yaw+xy):")
    print(T_ransac)

    draw_two(src_ds, tgt_ds, T_ransac,
             title="After RANSAC (LOCAL) - coarse yaw+xy",
             recenter=True)

    # --- initial transform in local ---
    T0_heuristic = np.eye(4)

    dz = 0.0  # fallback if USE_GROUND_INIT_Z is False

    if USE_CENTER_INIT_XY:
        cs = src_ds.get_center()
        ct = tgt_ds.get_center()
        t_xy = np.array([ct[0] - cs[0], ct[1] - cs[1], 0.0])
        T0_heuristic = T_translate(t_xy) @ T0_heuristic

    if USE_GROUND_INIT_Z:
        src_z0 = estimate_ground_z(np.asarray(src_ds.points), q=GROUND_Q)
        tgt_z0 = estimate_ground_z(np.asarray(tgt_ds.points), q=GROUND_Q)
        dz = (tgt_z0 - src_z0)
        T0_heuristic = T_translate([0.0, 0.0, dz]) @ T0_heuristic

    T0 = T0_heuristic @ T_ransac

    print("\n=== Initial local transform ===")
    print("T0_heuristic (center+ground):")
    print(T0_heuristic)
    print("\nT0 (final = heuristic ∘ RANSAC):")
    print(T0)

    draw_two(src_ds, tgt_ds, T0, title="Overlay (LOCAL): init (T0 = heuristic+RANSAC)", recenter=True)

    # ============================================================
    # CROP TARGET AROUND TRANSFORMED SOURCE
    # ============================================================
    print("\n" + "=" * 60)
    print("=== CROP target around transformed source ===")
    print("=" * 60)

    src_T0 = copy.deepcopy(src_ds)
    src_T0.transform(T0)

    bb = src_T0.get_axis_aligned_bounding_box()
    pad = CROP_PAD
    bb = o3d.geometry.AxisAlignedBoundingBox(bb.min_bound - pad, bb.max_bound + pad)

    tgt_crop = tgt_ds.crop(bb)
    print(f"CROP target_ds -> tgt_crop: {len(tgt_crop.points):,} points (pad={pad}m)")

    if len(tgt_crop.points) < 500:
        print(f"WARNING: Very few points after crop. Consider increasing CROP_PAD (current={pad}m)")

    # --- estimate normals on tgt_crop ---
    tgt_crop.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN))
    tgt_crop.orient_normals_towards_camera_location(src_ds.get_center())

    tgt_pts = np.asarray(tgt_crop.points)
    tgt_nrm = np.asarray(tgt_crop.normals)

    # --- choose vertical-ish target points (wall-like) ---
    nz = np.abs(tgt_nrm[:, 2])
    wall_mask = nz < VERTICAL_NZ_MAX
    if wall_mask.sum() < 200:
        raise RuntimeError("Too few vertical-ish points in target after normal filtering. Check normals/VOXEL.")

    tgt_wall_pts = tgt_pts[wall_mask]
    tgt_wall_nrm = tgt_nrm[wall_mask]

    # --- feature-rich wall point selection (window/door edge prioritization) ---
    if USE_FEATURE_RICH_TARGET:
        feat_scores = compute_local_nonplanarity(
            tgt_wall_pts,
            radius=FEATURE_RADIUS,
            max_nn=FEATURE_MAX_NN
        )

        th_feat = np.quantile(feat_scores, 1.0 - FEATURE_TOP_RATIO)
        feat_mask = feat_scores >= th_feat

        print(f"feature-rich wall pts: {feat_mask.sum():,} / {len(tgt_wall_pts):,} "
              f"(top {FEATURE_TOP_RATIO*100:.0f}% by non-planarity)")

        tgt_wall_pts = tgt_wall_pts[feat_mask]
        tgt_wall_nrm = tgt_wall_nrm[feat_mask]

    V = tgt_wall_nrm.copy()
    flip = V[:, 2] < 0
    V[flip] *= -1.0
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    labels, centers = kmeans2_unit_vectors(V, iters=30)

    pts0 = tgt_wall_pts[labels == 0]
    nrm0 = tgt_wall_nrm[labels == 0]
    pts1 = tgt_wall_pts[labels == 1]
    nrm1 = tgt_wall_nrm[labels == 1]

    tgt0 = o3d.geometry.PointCloud()
    tgt0.points = o3d.utility.Vector3dVector(pts0)
    tgt0.normals = o3d.utility.Vector3dVector(nrm0)

    tgt1 = o3d.geometry.PointCloud()
    tgt1.points = o3d.utility.Vector3dVector(pts1)
    tgt1.normals = o3d.utility.Vector3dVector(nrm1)

    if len(tgt0.points) < 200 or len(tgt1.points) < 200:
        print("Warning: one feature-rich normal cluster too small; results may be unstable.")

    print("\n=== Target wall clustering ===")
    print(f"vertical-ish points: {wall_mask.sum():,} / {len(tgt_pts):,}")
    print(f"cluster0 pts: {len(tgt0.points):,}  center_normal={centers[0]}")
    print(f"cluster1 pts: {len(tgt1.points):,}  center_normal={centers[1]}")

    # --- build source density weights on src_ds ---
    src_pts = np.asarray(src_ds.points)
    counts = voxel_counts(src_pts, DENSITY_VOXEL)
    w_density = 1.0 / np.power(counts, DENSITY_POWER)
    w_density = w_density / (np.mean(w_density) + 1e-12)

    # ============================================================
    # run_on_target_subset:  WITH NN DISTANCE DIAGNOSTICS
    # ============================================================
    def run_on_target_subset(tgt_subset, name, T_init):
        tgt_subset.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN))
        tgt_subset.orient_normals_towards_camera_location(src_ds.get_center())

        t_pts = np.asarray(tgt_subset.points)
        t_nrm = np.asarray(tgt_subset.normals)

        kd = o3d.geometry.KDTreeFlann(tgt_subset)

        T = T_init.copy()
        print(f"\n--- Weighted P2L ICP on {name} ---")

        # ★★★ Distance check at T0 ★★★
        nn_distance_stats(src_pts, T, kd, sample_n=5000, seed=SEED, name=f"{name} @ T0")

        for si, (th, it) in enumerate(STAGES, 1):
            T, met = weighted_p2l_icp(
                source_pts=src_pts,
                target_pts=t_pts,
                target_normals=t_nrm,
                kd_tree=kd,
                T_init=T,
                max_corr_dist=th,
                iters=it,
                src_density_weights=w_density,
                max_normal_angle_deg=MAX_NORMAL_ANGLE_DEG,
            )
            print(f"  stage {si}: th={th:.2f} it={it} -> used={met['used_corr']} rms_p2l={met['rms_p2l']:.4f}")

            # Distance check after each stage
            nn_distance_stats(src_pts, T, kd, sample_n=5000, seed=SEED,
                              name=f"{name} @ stage{si} (th={th:.2f})")

        ev = o3d.pipelines.registration.evaluate_registration(src_ds, tgt_subset, 0.60, T)
        score = ev.fitness - 0.5 * ev.inlier_rmse
        print(f"  [score] fitness={ev.fitness:.4f} rmse={ev.inlier_rmse:.4f} score={score:.4f}")
        return T, score, ev

    # ============================================================
    # Build facade candidates from both normal clusters and run
    # candidate-specific ICP with per-candidate T0 init
    # ============================================================
    cands0 = split_into_spatial_candidates(
        pts0, nrm0,
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        min_facade_points=MIN_FACADE_POINTS
    )
    for c in cands0:
        c["normal_cluster"] = 0

    cands1 = split_into_spatial_candidates(
        pts1, nrm1,
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        min_facade_points=MIN_FACADE_POINTS
    )
    for c in cands1:
        c["normal_cluster"] = 1

    facade_candidates = cands0 + cands1
    if len(facade_candidates) == 0:
        print("[WARNING] DBSCAN produced 0 facade candidates; fallback to full normal clusters.")
        facade_candidates = [
            {"points": pts0, "normals": nrm0, "size": len(pts0), "normal_cluster": 0, "label": 0},
            {"points": pts1, "normals": nrm1, "size": len(pts1), "normal_cluster": 1, "label": 0},
        ]
    facade_candidates.sort(key=lambda c: c["size"], reverse=True)

    print(f"\n[FACADE CANDIDATES] total={len(facade_candidates)}")
    for i, c in enumerate(facade_candidates):
        print(f"  cand{i}: normal_cluster={c['normal_cluster']}, label={c['label']}, size={c['size']}")

    best_score = -np.inf
    best_T = T0.copy()
    best_ev = None
    best_name = "none"

    for i, c in enumerate(facade_candidates):
        tgt_subset = o3d.geometry.PointCloud()
        tgt_subset.points = o3d.utility.Vector3dVector(c["points"])
        tgt_subset.normals = o3d.utility.Vector3dVector(c["normals"])

        cand_center = c["points"].mean(axis=0)

        # candidate-specific XY init: source main façade center -> candidate façade center
        dx = cand_center[0] - src_facade_center[0]
        dy = cand_center[1] - src_facade_center[1]

        T0_i = T_translate([dx, dy, dz]) @ T_ransac

        name = f"facade cand{i} (normal_cluster={c['normal_cluster']}, label={c['label']}, size={c['size']})"

        print(f"\n[INIT] {name}")
        print(f"  cand_center = {cand_center}")
        print(f"  src_facade_center = {src_facade_center}")
        print(f"  candidate-specific init dx,dy,dz = ({dx:.3f}, {dy:.3f}, {dz:.3f})")

        T_i, score_geom_i, ev_i = run_on_target_subset(tgt_subset, name, T0_i)

        facing_prior = 0.0
        if USE_FACING_PRIOR:
            facing_prior = compute_facing_prior(
                src_facade_center=src_facade_center,
                cand_pts=c["points"],
                cand_normals=c["normals"]
            )

        score_i = score_geom_i + FACING_PRIOR_WEIGHT * facing_prior

        print(f"  [prior] facing={facing_prior:.4f}  "
              f"geom={score_geom_i:.4f}  final={score_i:.4f}")

        if score_i > best_score:
            best_score = score_i
            best_T = T_i
            best_ev = ev_i
            best_name = name

    T_ref_l = best_T

    print("\n==================== BEST (LOCAL) ====================")
    print(f"best = {best_name}")
    print("T_ref_local:\n", T_ref_l)
    print(f"eval@0.60m: fitness={best_ev.fitness:.4f} rmse={best_ev.inlier_rmse:.4f}")

    draw_two(src_l, tgt_l, T_ref_l, title="After weighted P2L (LOCAL)", recenter=True)
    draw_two(src_l, tgt_crop, T_ref_l, title="After weighted P2L (LOCAL) - cropped target", recenter=True)

    # --- convert back to global (UTM) ---
    T_ref_g = Si @ T_ref_l

    print("\n==================== RESULT (GLOBAL / UTM) ====================")
    print("T_ref_global:\n", T_ref_g)

    draw_two(src_g, tgt_g, T_ref_g, title="After weighted P2L (GLOBAL/UTM)", recenter=True)

    src_aligned_local = copy.deepcopy(src_l)
    src_aligned_local.transform(T_ref_l)
    o3d.io.write_point_cloud("src_aligned_local.pcd", src_aligned_local)

    src_aligned_global = copy.deepcopy(src_g)
    src_aligned_global.transform(T_ref_g)
    o3d.io.write_point_cloud("src_aligned_global_utm.pcd", src_aligned_global)

    print("\nSaved:")
    print("  src_aligned_local.pcd")
    print("  src_aligned_global_utm.pcd")


if __name__ == "__main__":
    main()