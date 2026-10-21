# crop_utils.py
import copy
import numpy as np
import open3d as o3d


def T_translate(txyz):
    """4x4 translation matrix."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(txyz, dtype=float)
    return T


def make_local(src_g: o3d.geometry.PointCloud,
               tgt_g: o3d.geometry.PointCloud,
               shift_mode: str = "center"):

    if shift_mode == "center":
        shift = np.asarray(tgt_g.get_center(), dtype=float)
    elif shift_mode == "min":
        shift = np.asarray(tgt_g.get_min_bound(), dtype=float)
    else:
        raise ValueError("shift_mode must be 'center' or 'min'")

    S = T_translate(-shift)   # global -> local
    Si = T_translate(shift)   # local -> global

    src_l = copy.deepcopy(src_g); src_l.transform(S)
    tgt_l = copy.deepcopy(tgt_g); tgt_l.transform(S)
    return src_l, tgt_l, S, Si, shift


def crop_by_aabb_of_target(source: o3d.geometry.PointCloud,
                           target: o3d.geometry.PointCloud,
                           margin: float = 2.0) -> o3d.geometry.PointCloud:

    bbox = target.get_axis_aligned_bounding_box()
    minb = bbox.min_bound - margin
    maxb = bbox.max_bound + margin
    bbox2 = o3d.geometry.AxisAlignedBoundingBox(minb, maxb)
    return source.crop(bbox2)


def crop_by_aabb_with_retry(source: o3d.geometry.PointCloud,
                            target: o3d.geometry.PointCloud,
                            margin: float = 2.0,
                            retry_margins=(5.0, 10.0, 20.0, 50.0, 100.0),
                            min_points: int = 200):

    # try initial
    cropped = crop_by_aabb_of_target(source, target, margin=margin)
    if len(cropped.points) >= min_points:
        return cropped, margin

    # retry larger margins
    for m in retry_margins:
        cropped2 = crop_by_aabb_of_target(source, target, margin=float(m))
        if len(cropped2.points) >= min_points:
            return cropped2, float(m)

    # still too few: return the best we got (maybe empty) + last margin tried
    return cropped, margin
