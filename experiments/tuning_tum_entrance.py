import copy
import numpy as np
import open3d as o3d

# ============================================================
# 0) Paths
# ============================================================
MODEL_PATH = r"F:\data\building_TUM_entrance.pcd"  # CityGML wall point cloud (UTM/global)
MLS_PATH = r"F:\data\pointCloud_TUM_entrance_global_coordinates_offset.pcd"  #  global TLS

# ============================================================
# 2) Determinism
# ============================================================
SEED = 42

# ============================================================
# 3) Downsample / normals
# ============================================================
VOXEL = 0.3  # fast tuning mode
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
USE_CENTER_INIT_XY = False
USE_GROUND_INIT_Z = False
GROUND_Q = 0.05  # 5% lowest points as ground proxy (robust)

# CROP parameters
CROP_PAD = 40.0  # larger pad to avoid cropping correct faces


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


#  NEW: NN distance diagnostic
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


# ============================================================
# Normal clustering (deterministic)
# ============================================================
def kmeans2_unit_vectors(V, iters=20):
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
    """Keep only yaw rotation + xy translation, zero out pitch/roll/z"""
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
# Weighted Point-to-Plane ICP (custom)
# ============================================================
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

    cos_max = np.cos(np.radians(max_normal_angle_deg))
    Ns = source_pts.shape[0]

    used = 0
    rms = None

    for it_idx in range(iters):
        R = T[:3, :3]
        t = T[:3, 3]
        P = (source_pts @ R.T) + t  # (Ns,3)

        A_list = []
        b_list = []
        w_list = []

        res2_sum = 0.0
        used = 0

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

            r = float(n.dot(p - q))  # point-to-plane residual

            J_rot = -n @ skew(p)
            J = np.hstack([J_rot, n])  # (6,)

            w = 1.0
            w *= huber_weight(r, HUBER_K)
            if src_density_weights is not None:
                w *= float(src_density_weights[i])

            A_list.append(J)
            b_list.append(-r)
            w_list.append(w)

            res2_sum += (r * r)
            used += 1

        # MODIFIED: revert to last stable pose if correspondence collapses
        if used < 20:
            print(f"    [ICP] break at iter {it_idx}: used={used} < 20 (th={max_corr_dist:.2f}m)")
            # DO NOT update T further; revert to previous stable pose
            return T_init.copy(), {
                "used_corr": used,
                "rms_p2l": 0.0,
            }

        A = np.vstack(A_list)  # (M,6)
        b = np.array(b_list).reshape(-1, 1)  # (M,1)
        w = np.array(w_list).reshape(-1, 1)  # (M,1)

        Aw = A * w
        bw = b * w
        H = Aw.T @ A
        g = Aw.T @ b

        lam = 1e-6
        H = H + lam * np.eye(6)

        try:
            xi = np.linalg.solve(H, g).flatten()
        except np.linalg.LinAlgError:
            break

        T = se3_update(T, xi)

        rms = float(np.sqrt(res2_sum / max(1, used)))

        if np.linalg.norm(xi) < 1e-6:
            break

    metrics = {
        "used_corr": used,
        "rms_p2l": rms if rms is not None else 0.0,
    }
    return T, metrics


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    # --- load ---
    src_g = o3d.io.read_point_cloud(MLS_PATH)  # MLS (global)
    tgt_g = o3d.io.read_point_cloud(MODEL_PATH)  # CityGML (UTM/global)

    if src_g.is_empty() or tgt_g.is_empty():
        raise RuntimeError("Empty point cloud. Check paths.")

    print("=== RAW (Open3D read) ===")
    stats(src_g, "src (TLS)")
    stats(tgt_g, "tgt (CityGML UTM)")

    src_l = copy.deepcopy(src_g)
    tgt_l = copy.deepcopy(tgt_g)

    print("\n=== Clouds in global frame (no shift applied) ===")
    stats(src_l, "src_l")
    stats(tgt_l, "tgt_l")

    # --- downsample ---
    src_ds = src_l.voxel_down_sample(VOXEL)
    tgt_ds = tgt_l.voxel_down_sample(VOXEL)

    print(f"\nDownsample VOXEL={VOXEL:.2f}m")
    print(f"src_ds: {len(src_ds.points):,}")
    print(f"tgt_ds: {len(tgt_ds.points):,}")

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
    # --- GLOBAL MODE INIT ---
    T0 = T_ransac.copy()

    # only align ground in Z
    src_z0 = estimate_ground_z(np.asarray(src_ds.points), q=GROUND_Q)
    tgt_z0 = estimate_ground_z(np.asarray(tgt_ds.points), q=GROUND_Q)
    dz = (tgt_z0 - src_z0)

    T0 = T_translate([0.0, 0.0, dz]) @ T0

    print("\n=== Initial local transform ===")
    print("T0 (= Z-ground ∘ RANSAC):")
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

    V = tgt_wall_nrm.copy()
    flip = V[:, 2] < 0
    V[flip] *= -1.0
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    labels, centers = kmeans2_unit_vectors(V, iters=30)

    idx_wall = np.where(wall_mask)[0]
    idx0 = idx_wall[labels == 0]
    idx1 = idx_wall[labels == 1]

    if len(idx0) < 200 or len(idx1) < 200:
        print("Warning: one normal cluster too small; will still run but results may be unstable.")

    tgt0 = tgt_crop.select_by_index(idx0.tolist())
    tgt1 = tgt_crop.select_by_index(idx1.tolist())

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
    def run_on_target_subset(tgt_subset, name):
        tgt_subset.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN))
        tgt_subset.orient_normals_towards_camera_location(src_ds.get_center())

        t_pts = np.asarray(tgt_subset.points)
        t_nrm = np.asarray(tgt_subset.normals)

        kd = o3d.geometry.KDTreeFlann(tgt_subset)

        T = T0.copy()
        print(f"\n--- Weighted P2L ICP on {name} ---")

        # Distance check at T0
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

            if met["used_corr"] < 20:
                print("  stage failed — stop refinement")
                break

            # Distance check after each stage 
            nn_distance_stats(src_pts, T, kd, sample_n=5000, seed=SEED,
                              name=f"{name} @ stage{si} (th={th:.2f})")

        ev = o3d.pipelines.registration.evaluate_registration(src_ds, tgt_subset, 0.60, T)
        score = ev.fitness - 0.5 * ev.inlier_rmse
        print(f"  [score] fitness={ev.fitness:.4f} rmse={ev.inlier_rmse:.4f} score={score:.4f}")
        return T, score, ev

    T_a, score_a, ev_a = run_on_target_subset(tgt0, "target wall cluster 0")
    T_b, score_b, ev_b = run_on_target_subset(tgt1, "target wall cluster 1")

    if score_b > score_a:
        T_ref_l = T_b
        best_name = "cluster 1"
        best_ev = ev_b
    else:
        T_ref_l = T_a
        best_name = "cluster 0"
        best_ev = ev_a

    print("\n==================== BEST (LOCAL) ====================")
    print(f"best = {best_name}")
    print("T_ref_local:\n", T_ref_l)
    print(f"eval@0.60m: fitness={best_ev.fitness:.4f} rmse={best_ev.inlier_rmse:.4f}")

    draw_two(src_l, tgt_l, T_ref_l, title="After weighted P2L (LOCAL)", recenter=True)
    draw_two(src_l, tgt_crop, T_ref_l, title="After weighted P2L (LOCAL) - cropped target", recenter=True)

    # Already in global frame; no inverse shift needed
    T_ref_g = T_ref_l.copy()

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