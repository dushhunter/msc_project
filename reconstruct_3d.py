#!/usr/bin/env python3
"""3D reconstruction of a stone from rotated depth maps (.npy).

Inputs
------
A directory containing N .npy depth maps (float32, shape (H, W),
camera-space Z in metres, e.g. Blender's `Depth` pass). Each view is the
same scene with the stone rotated about a vertical axis on a flat surface.

Conventions
-----------
World frame:
  Z  = "up", the ground-plane normal pointing toward the camera.
  X, Y lie in the ground plane.
  Origin = the turntable axis intersection with the plane.
  Rotation between successive views is therefore a rotation about Z.

Pipeline
--------
For each view:
  1. Pinhole back-projection (intrinsics from --focal_mm/--sensor_mm or from
     a user-supplied horizontal FOV when no camera info is available).
  2. RANSAC ground-plane fit, refit on inliers via SVD. A single shared
     ground plane is then refit by pooling inliers across all views (the
     camera is fixed, so the plane is the same in every view).
  3. Stone segmentation: pixels with signed distance above the plane greater
     than --stone_offset, kept as the largest 4-connected component.

Across views:
  4. Estimate the turntable axis from data: the stone centroid traces a
     circle on the plane as the stone rotates, so with known angles theta_i
     the axis-on-plane is the linear least-squares solution of
         c_i = O + R(theta_i) v.
     This handles asymmetric stones whose centroid is offset from the axis.
  5. Build a world frame at (axis-on-plane, plane-normal toward camera) with
     Z = up, then de-rotate each view by -theta_i around Z.
  6. Optional pairwise ICP refinement (Besl & McKay 1992) of every view
     against view 0, using scipy.spatial.cKDTree.

Outputs
-------
  view_<idx>_<deg>.ply : per-view stone cloud, one colour per view (helps
                        verify alignment visually).
  merged.ply           : concatenation, colour-tagged.
  merged_white.ply     : single-colour version (good input for Poisson
                        surface reconstruction).
  transforms.json      : intrinsics, plane equation, axis location, all
                        per-view 4x4 poses, ICP residuals.

Configuration
-------------
Default options live in `<project_root>/configs/reconstruct_3d.txt`
(argparse-style: one `--option value` per line). CLI arguments override
the config values. Use --config to point at a different file.

References
----------
  Besl & McKay, "A Method for Registration of 3-D Shapes",
    IEEE TPAMI 1992. (ICP)
  Curless & Levoy, "A Volumetric Method for Building Complex Models from
    Range Images", SIGGRAPH 1996.
  Kazhdan & Hoppe, "Screened Poisson Surface Reconstruction", TOG 2013.
"""

from __future__ import print_function

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    from scipy.ndimage import label
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "reconstruct_3d.txt")


# ---------------------------------------------------------------------------
# Argparse with config-file support (argparse-style, one --opt value per line)
# ---------------------------------------------------------------------------

class CfgFileParser(argparse.ArgumentParser):
    """Argparse parser that reads `@filename` config files where each line is
    `--option value` (or `# comment`)."""

    def convert_arg_line_to_args(self, arg_line):
        s = arg_line.strip()
        if not s or s.startswith("#"):
            return []
        return s.split()


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def blender_intrinsics(width, height, focal_mm=50.0, sensor_mm=36.0):
    """Pinhole intrinsics for Blender's default sensor_fit=AUTO (sensor_mm
    refers to the longer image axis). Returns fx, fy, cx, cy."""
    long_side = max(width, height)
    f_pix = focal_mm / sensor_mm * long_side
    fx = fy = f_pix
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return fx, fy, cx, cy


def hfov_intrinsics(width, height, hfov_deg):
    """Pinhole intrinsics from a horizontal field of view (degrees) only.
    Used as a fallback when no camera settings are available. Assumes square
    pixels and principal point at the image centre."""
    f = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    fx = fy = f
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def backproject(depth, fx, fy, cx, cy):
    """Back-project (H, W) depth (camera-space Z, m) into 3D points in the
    camera frame. Returns (H, W, 3) and a (H, W) bool validity mask."""
    H, W = depth.shape
    u = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    v = np.arange(H, dtype=np.float64)[:, None].repeat(W, axis=1)
    Z = depth.astype(np.float64)
    valid = np.isfinite(Z) & (Z > 1e-6)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack((X, Y, Z), axis=-1)
    return pts, valid


def ransac_plane(points, n_iter=2000, threshold=5e-4, rng=None):
    """RANSAC plane fit: returns (n, d, inlier_mask) with ||n|| = 1 and
    n . p + d = 0 on the plane. Refits on inliers via SVD."""
    if rng is None:
        rng = np.random.RandomState(0)
    N = points.shape[0]
    best_count = -1
    best_n = None
    best_d = 0.0
    best_inliers = None
    for _ in range(n_iter):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        nl = np.linalg.norm(nrm)
        if nl < 1e-9:
            continue
        nrm = nrm / nl
        d = -float(nrm @ p0)
        dist = np.abs(points @ nrm + d)
        inliers = dist < threshold
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_n = nrm
            best_d = d
            best_inliers = inliers
    if best_inliers is None or best_count < 3:
        raise RuntimeError("RANSAC plane fit failed")
    pin = points[best_inliers]
    centroid = pin.mean(axis=0)
    cov = (pin - centroid).T @ (pin - centroid)
    _, _, Vh = np.linalg.svd(cov)
    n = Vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ centroid)
    dist = np.abs(points @ n + d)
    inliers = dist < threshold
    return n, d, inliers


def orient_normal_toward_camera(n, d):
    """Flip plane normal so the camera origin (0, 0, 0) lies on the positive
    side, i.e. so the normal points from the plane back to the camera. The
    plane equation is n.p + d = 0, so the camera signed distance is d. Make
    it positive."""
    if d < 0:
        return -n, -d
    return n, d


def largest_connected_component(mask):
    """Keep only the largest 4-connected component of a 2D boolean mask."""
    if not HAVE_SCIPY:
        return mask
    lab, n = label(mask.astype(np.uint8))
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = int(counts.argmax())
    return lab == keep


# ---------------------------------------------------------------------------
# Turntable axis (centroid-circle fit)
# ---------------------------------------------------------------------------

def project_to_plane_basis(points, n, d):
    """Express 3D points in a 2D orthonormal basis (e1, e2) on the plane.
    Returns 2D coords (M, 2) and (e1, e2)."""
    if abs(n[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    base = -d * n  # arbitrary point on the plane
    proj = points - (points @ n + d)[:, None] * n[None, :] - base[None, :]
    return np.stack((proj @ e1, proj @ e2), axis=-1), (e1, e2)


def axis_from_centroids(centroids_2d, angles_rad):
    """Solve for axis location O and stone-centroid offset v in plane coords:
        c_i = O + R(theta_i) v
    via linear least-squares.

    Components (with v = (a, b)):
        c_i.x = O.x + a cos(theta_i) - b sin(theta_i)
        c_i.y = O.y + a sin(theta_i) + b cos(theta_i)
    """
    N = len(angles_rad)
    A = np.zeros((2 * N, 4))
    rhs = np.zeros(2 * N)
    for i, t in enumerate(angles_rad):
        c, s = np.cos(t), np.sin(t)
        A[2 * i + 0] = [1.0, 0.0,  c, -s]
        A[2 * i + 1] = [0.0, 1.0,  s,  c]
        rhs[2 * i + 0] = centroids_2d[i, 0]
        rhs[2 * i + 1] = centroids_2d[i, 1]
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return sol[:2], sol[2:]


# ---------------------------------------------------------------------------
# World frame and rotations (Z up)
# ---------------------------------------------------------------------------

def make_world_frame(n_up_cam, axis_point_cam):
    """Build T_world_from_cam such that:
        world Z = plane normal toward camera
        world X = horizontal projection of (camera->axis) direction
        world Y = world Z x world X
    Origin is at axis_point_cam (turntable axis on the ground plane)."""
    z = n_up_cam / np.linalg.norm(n_up_cam)
    horiz = axis_point_cam - (axis_point_cam @ z) * z
    if np.linalg.norm(horiz) < 1e-9:
        # Camera is right above the axis: pick any horizontal direction.
        helper = np.array([1.0, 0.0, 0.0])
        horiz = helper - (helper @ z) * z
    x = horiz / np.linalg.norm(horiz)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.stack((x, y, z), axis=0)
    t = -R @ axis_point_cam
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def transform_points(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# ICP (point-to-point, Besl & McKay 1992)
# ---------------------------------------------------------------------------

def icp_p2p(src, dst, max_iter=40, max_corr_dist=5e-3, tol=1e-6):
    """Align src -> dst rigidly. Returns (R, t, rmse)."""
    if not HAVE_SCIPY:
        return np.eye(3), np.zeros(3), float("nan")
    R = np.eye(3)
    t = np.zeros(3)
    src_c = src.copy()
    tree = cKDTree(dst)
    prev_rmse = np.inf
    rmse = np.inf
    for _ in range(max_iter):
        dist, idx = tree.query(src_c, k=1)
        m = dist < max_corr_dist
        if int(m.sum()) < 50:
            break
        s = src_c[m]
        d = dst[idx[m]]
        sc = s.mean(axis=0)
        dc = d.mean(axis=0)
        H = (s - sc).T @ (d - dc)
        U, _, Vh = np.linalg.svd(H)
        Rd = Vh.T @ U.T
        if np.linalg.det(Rd) < 0:
            Vh[-1] *= -1
            Rd = Vh.T @ U.T
        td = dc - Rd @ sc
        R = Rd @ R
        t = Rd @ t + td
        src_c = src @ R.T + t
        rmse = float(np.sqrt(((src_c[m] - dst[idx[m]]) ** 2).sum(axis=1).mean()))
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse
    return R, t, rmse


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def write_ply_binary(path, points, colors=None):
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    has_color = colors is not None
    with open(path, "wb") as f:
        header = ["ply",
                  "format binary_little_endian 1.0",
                  "element vertex %d" % n,
                  "property float x",
                  "property float y",
                  "property float z"]
        if has_color:
            header += ["property uchar red",
                       "property uchar green",
                       "property uchar blue"]
        header.append("end_header")
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if has_color:
            colors = np.asarray(colors, dtype=np.uint8)
            assert colors.shape == (n, 3)
            buf = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
            buf["xyz"] = points
            buf["rgb"] = colors
            buf.tofile(f)
        else:
            points.astype("<f4").tofile(f)


def turbo_palette(n):
    base = np.array([
        [0.18, 0.07, 0.55],
        [0.27, 0.41, 0.97],
        [0.20, 0.72, 0.92],
        [0.15, 0.94, 0.50],
        [0.74, 1.00, 0.27],
        [1.00, 0.81, 0.20],
        [0.99, 0.46, 0.13],
        [0.80, 0.10, 0.05],
    ])
    if n <= len(base):
        cols = base[:n]
    else:
        cols = np.vstack([base] + [base] * ((n + len(base) - 1) // len(base)))[:n]
    return (cols * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-view processing
# ---------------------------------------------------------------------------

def process_view(depth, fx, fy, cx, cy, plane_threshold, stone_offset,
                  ransac_iter, rng_seed=0):
    pts, valid = backproject(depth, fx, fy, cx, cy)
    H, W, _ = pts.shape
    flat = pts.reshape(-1, 3)
    valid_flat = valid.reshape(-1)
    sub = flat[valid_flat]
    if sub.shape[0] > 60000:
        rng = np.random.RandomState(rng_seed)
        sel = rng.choice(sub.shape[0], 60000, replace=False)
        sub = sub[sel]
    n, d, _ = ransac_plane(sub, n_iter=ransac_iter,
                           threshold=plane_threshold,
                           rng=np.random.RandomState(rng_seed))
    n, d = orient_normal_toward_camera(n, d)
    sd = (flat @ n + d).reshape(H, W)
    plane_inlier_mask = (np.abs(sd) < plane_threshold) & valid
    stone_mask = (sd > stone_offset) & valid
    stone_mask = largest_connected_component(stone_mask)
    return {
        "points_cam": pts,
        "valid": valid,
        "plane_n_cam": n,
        "plane_d_cam": d,
        "plane_inliers": plane_inlier_mask,
        "stone_mask": stone_mask,
        "signed_dist": sd,
    }


def shared_plane(per_view, plane_threshold):
    """Refit a single ground plane using inliers pooled across all views."""
    accum = []
    for v in per_view:
        m = v["plane_inliers"]
        accum.append(v["points_cam"][m])
    P = np.concatenate(accum, axis=0)
    if P.shape[0] > 200000:
        rng = np.random.RandomState(0)
        P = P[rng.choice(P.shape[0], 200000, replace=False)]
    centroid = P.mean(axis=0)
    cov = (P - centroid).T @ (P - centroid)
    _, _, Vh = np.linalg.svd(cov)
    n = Vh[-1]
    n /= np.linalg.norm(n)
    d = -float(n @ centroid)
    n, d = orient_normal_toward_camera(n, d)
    return n, d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    ap = CfgFileParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        fromfile_prefix_chars="@")

    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="Path to a .txt config file (argparse-style, one "
                         "`--option value` per line). CLI args override it. "
                         "Default: %s" % DEFAULT_CONFIG)

    # Inputs / outputs
    ap.add_argument("--input_dir", required=False, default=None,
                    help="Directory containing the .npy depth maps.")
    ap.add_argument("--output_dir", required=False, default=None,
                    help="Directory to write per-view PLYs and merged.ply.")

    # Acquisition geometry
    ap.add_argument("--angles_deg", default="0,15,30,45,60,75,90,105",
                    help="Comma-separated rotation angles, one per file.")
    ap.add_argument("--rotation_sign", type=float, default=+1.0,
                    help="+1 or -1; flip if reconstructed views are mirror-rotated.")

    # Camera settings
    ap.add_argument("--use_camera_settings", default="true",
                    help="true: build pinhole intrinsics from --focal_mm "
                         "and --sensor_mm. false: rely only on the depth "
                         "values and a horizontal FOV (--horizontal_fov_deg).")
    ap.add_argument("--focal_mm", type=float, default=50.0)
    ap.add_argument("--sensor_mm", type=float, default=36.0,
                    help="Sensor size along the longer image axis.")
    ap.add_argument("--horizontal_fov_deg", type=float, default=50.0,
                    help="Used when --use_camera_settings=false. Common "
                         "values: phone main camera ~65, action cam ~120, "
                         "Blender 50mm/36mm ~39.6 deg.")
    ap.add_argument("--cx", type=float, default=None,
                    help="Override principal point u (default: image centre).")
    ap.add_argument("--cy", type=float, default=None,
                    help="Override principal point v (default: image centre).")

    # Reconstruction tuning
    ap.add_argument("--plane_threshold", type=float, default=5e-4,
                    help="RANSAC plane inlier distance, m.")
    ap.add_argument("--stone_offset", type=float, default=1e-3,
                    help="Min height above the plane (m) to count as stone.")
    ap.add_argument("--ransac_iter", type=int, default=2000)
    ap.add_argument("--decimate", type=int, default=2,
                    help="Stride to thin saved per-view stone clouds (1 = full).")

    # ICP
    ap.add_argument("--icp", default="true",
                    help="true to run ICP refinement of every view against view 0.")
    ap.add_argument("--icp_max_corr", type=float, default=5e-3)

    return ap


def parse_args_with_config():
    """Parse with --config support: config file is loaded first (as if its
    lines were prepended to argv), then CLI flags override it."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    cfg_args, _ = pre.parse_known_args()
    cfg_path = cfg_args.config

    ap = build_parser()
    argv_chain = []
    if cfg_path and os.path.isfile(cfg_path):
        argv_chain.append("@" + cfg_path)
    argv_chain.extend(sys.argv[1:])
    return ap.parse_args(argv_chain), cfg_path


def main():
    args, cfg_path = parse_args_with_config()

    if args.input_dir is None:
        sys.exit("--input_dir is required (set it in the config file or pass it on CLI).")
    if args.output_dir is None:
        sys.exit("--output_dir is required (set it in the config file or pass it on CLI).")

    print("Config file:", cfg_path if (cfg_path and os.path.isfile(cfg_path))
          else "(none)")
    print("Input dir :", args.input_dir)
    print("Output dir:", args.output_dir)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not files:
        sys.exit("No .npy files found in %s" % args.input_dir)
    angles_deg = [float(x) for x in args.angles_deg.split(",")]
    if len(angles_deg) != len(files):
        sys.exit("Got %d files but %d angles." % (len(files), len(angles_deg)))
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64)) * args.rotation_sign

    os.makedirs(args.output_dir, exist_ok=True)

    depths = [np.load(f) for f in files]
    H, W = depths[0].shape
    for d in depths:
        if d.shape != (H, W):
            sys.exit("Depth maps must have the same shape.")

    use_cam = _truthy(args.use_camera_settings)
    if use_cam:
        fx, fy, cx_def, cy_def = blender_intrinsics(W, H, args.focal_mm, args.sensor_mm)
        intr_src = "camera (%.1fmm / %.1fmm sensor)" % (args.focal_mm, args.sensor_mm)
    else:
        fx, fy, cx_def, cy_def = hfov_intrinsics(W, H, args.horizontal_fov_deg)
        intr_src = "no-camera fallback (HFOV=%.2f deg)" % args.horizontal_fov_deg
    cx = args.cx if args.cx is not None else cx_def
    cy = args.cy if args.cy is not None else cy_def
    print("Intrinsics from %s: fx=%.3f fy=%.3f cx=%.3f cy=%.3f (image %dx%d)"
          % (intr_src, fx, fy, cx, cy, W, H))

    per_view = []
    for i, (f, dep) in enumerate(zip(files, depths)):
        v = process_view(dep, fx, fy, cx, cy, args.plane_threshold,
                          args.stone_offset, args.ransac_iter, rng_seed=i)
        v["filename"] = os.path.basename(f)
        v["angle_deg"] = angles_deg[i]
        per_view.append(v)
        print("[%d] %s  stone_pixels=%d  plane_inliers=%d  depth_min=%.3f max=%.3f"
              % (i, v["filename"], int(v["stone_mask"].sum()),
                 int(v["plane_inliers"].sum()),
                 float(dep.min()), float(dep.max())))

    n_cam, d_cam = shared_plane(per_view, args.plane_threshold)
    print("Shared plane (camera frame): n=%s d=%.4f" % (n_cam, d_cam))

    centroids_cam = []
    for v in per_view:
        m = v["stone_mask"]
        if m.sum() < 100:
            sys.exit("View %s: too few stone pixels (%d). Tune --stone_offset."
                     % (v["filename"], int(m.sum())))
        sp = v["points_cam"][m]
        c = sp.mean(axis=0)
        c_proj = c - (float(c @ n_cam) + d_cam) * n_cam
        centroids_cam.append(c_proj)
    centroids_cam = np.asarray(centroids_cam)

    centroids_2d, (e1, e2) = project_to_plane_basis(centroids_cam, n_cam, d_cam)
    spread = float(np.linalg.norm(centroids_2d - centroids_2d.mean(axis=0), axis=1).max())
    print("Centroid spread on plane = %.4f m" % spread)

    if spread > 2e-3:
        O_2d, v_2d = axis_from_centroids(centroids_2d, angles_rad)
        axis_point_cam = (-d_cam) * n_cam + O_2d[0] * e1 + O_2d[1] * e2
        circle_radius = float(np.linalg.norm(v_2d))
        print("Turntable axis fit: O_plane=%s  radius=%.4f m" % (O_2d, circle_radius))
    else:
        # Stone is rotationally symmetric: centroids coincide. Use mean.
        axis_point_cam = centroids_cam.mean(axis=0)
        print("Centroids near-coincident -> using mean centroid as axis point.")

    T_world_cam = make_world_frame(n_cam, axis_point_cam)
    print("T_world_from_cam =\n", T_world_cam)

    palette = turbo_palette(len(files))
    aligned = []
    transforms = []
    for i, v in enumerate(per_view):
        m = v["stone_mask"]
        sp_cam = v["points_cam"][m]
        if args.decimate > 1:
            sp_cam = sp_cam[::args.decimate]
        Tc = np.eye(4)
        Tc[:3, :3] = rotation_z(-angles_rad[i])
        T_full = Tc @ T_world_cam
        sp_world = transform_points(T_full, sp_cam)
        aligned.append(sp_world)
        transforms.append({"file": v["filename"], "angle_deg": v["angle_deg"],
                            "T_coarse": T_full.tolist()})

    do_icp = _truthy(args.icp)
    if do_icp and HAVE_SCIPY:
        ref = aligned[0]
        for i in range(1, len(aligned)):
            R, t, rmse = icp_p2p(aligned[i], ref,
                                  max_iter=50,
                                  max_corr_dist=args.icp_max_corr)
            aligned[i] = aligned[i] @ R.T + t
            transforms[i]["T_icp"] = np.block([[R, t.reshape(3, 1)],
                                                [np.zeros((1, 3)), np.ones((1, 1))]]).tolist()
            transforms[i]["icp_rmse"] = rmse
            print("[ICP] view %d -> view 0   rmse=%.5f m" % (i, rmse))
    elif do_icp and not HAVE_SCIPY:
        print("--icp requested but scipy is missing; skipping ICP.")

    for i, (sp_world, v) in enumerate(zip(aligned, per_view)):
        col = np.tile(palette[i % len(palette)], (sp_world.shape[0], 1))
        path = os.path.join(args.output_dir,
                             "view_%02d_%03dDeg.ply" % (i, int(round(angles_deg[i]))))
        write_ply_binary(path, sp_world, col)
        print("wrote", path, "points=", sp_world.shape[0])

    merged = np.concatenate(aligned, axis=0)
    merged_col = np.concatenate([np.tile(palette[i % len(palette)], (a.shape[0], 1))
                                  for i, a in enumerate(aligned)], axis=0)
    p_merge = os.path.join(args.output_dir, "merged.ply")
    write_ply_binary(p_merge, merged, merged_col)
    print("wrote", p_merge, "points=", merged.shape[0])

    p_white = os.path.join(args.output_dir, "merged_white.ply")
    write_ply_binary(p_white, merged,
                      np.full((merged.shape[0], 3), 230, dtype=np.uint8))
    print("wrote", p_white, "points=", merged.shape[0])

    info = {
        "config_file": cfg_path if (cfg_path and os.path.isfile(cfg_path)) else None,
        "intrinsics_source": intr_src,
        "use_camera_settings": use_cam,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                        "width": W, "height": H},
        "world_axes": {"Z_up_is_plane_normal": True,
                        "X_horiz_along_camera_to_axis": True,
                        "Y": "Z x X"},
        "plane_camera_frame": {"n": n_cam.tolist(), "d": float(d_cam)},
        "axis_point_camera_frame": axis_point_cam.tolist(),
        "rotation_axis": "Z",
        "rotation_sign": args.rotation_sign,
        "angles_deg": angles_deg,
        "transforms": transforms,
        "centroid_spread_on_plane_m": spread,
        "icp_used": bool(do_icp and HAVE_SCIPY),
    }
    with open(os.path.join(args.output_dir, "transforms.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("wrote", os.path.join(args.output_dir, "transforms.json"))


if __name__ == "__main__":
    main()
