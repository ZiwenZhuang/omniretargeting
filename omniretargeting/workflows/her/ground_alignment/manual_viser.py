"""
Interactive ground plane alignment for scene + SMPL-X reconstruction outputs.

This tool loads a scene point cloud (either from a fused PLY or reconstructed RGB-D frames),
optionally overlays the reconstructed SMPL-X human mesh, and provides an interactive Viser UI
to:
  - pick points on the ground plane,
  - fit a plane with local RANSAC,
  - compute a rigid transform that aligns the plane normal to +Z and moves the plane to z=0,
  - optionally fine-adjust height with a slider,
  - export the final 4x4 transform to a stable JSON path for downstream pipeline steps.

Interface: call `ground_alignment.main(config_dict)` from a pipeline. The function blocks until
the user clicks "Save transform matrix", then returns the 4x4 numpy transform.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import open3d as o3d
import smplx
import torch
import viser
import tyro
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# -----------------------------
# Coordinate helpers
# -----------------------------
def opencv_to_z_up(xyz: np.ndarray) -> np.ndarray:
    """OpenCV (x right, y down, z forward) -> Z-up (x forward, y left, z up)."""
    out = xyz.copy()
    out[..., 0] = xyz[..., 0]
    out[..., 1] = xyz[..., 2]
    out[..., 2] = -xyz[..., 1]
    return out


def apply_transform_matrix(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    pts = np.asarray(points)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)
    return (pts_h @ T.T)[:, :3]


def create_pointcloud_from_rgbd(rgb_image, depth_map, camera_intrinsics, camera_pose, mask=None):
    """
    Create a point cloud (world frame) from an RGB image and depth map.

    Args:
        rgb_image: (H, W, 3) RGB image (uint8, 0-255)
        depth_map: (H, W) depth map in meters
        camera_intrinsics: (3, 3) intrinsic matrix
        camera_pose: (4, 4) camera-to-world matrix
        mask: Optional (H, W) binary mask: mask>0 indicates pixels to KEEP.

    Returns:
        points_world: (N, 3) points in world frame
        colors_uint8: (N, 3) colors uint8
    """
    H, W = depth_map.shape
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    valid_mask = depth_map > 0
    if mask is not None:
        valid_mask = valid_mask & (mask > 0)

    z_cam = depth_map[valid_mask]
    x_cam = (u[valid_mask] - cx) * z_cam / fx
    y_cam = (v[valid_mask] - cy) * z_cam / fy
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    R_cw = camera_pose[:3, :3]
    t_cw = camera_pose[:3, 3]
    points_world = (R_cw @ points_cam.T).T + t_cw

    colors_uint8 = rgb_image[valid_mask].astype(np.uint8)
    return points_world, colors_uint8


def voxel_downsample(points, colors=None, voxel_size=0.05):
    """Voxel grid downsampling (pure numpy)."""
    points = np.asarray(points)
    if points.size == 0:
        return points, colors

    voxel_coords = np.floor(points / float(voxel_size)).astype(int)
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    downsampled_points = points[unique_indices]
    downsampled_colors = colors[unique_indices] if colors is not None else None
    return downsampled_points, downsampled_colors


# -----------------------------
# Plane fitting + alignment
# -----------------------------
def fit_plane_with_local_cloud(
    points: np.ndarray,
    kd: cKDTree,
    picked_indices: list[int],
    neighbor_radius: float,
    distance_threshold: float,
    ransac_n: int = 3,
    num_iterations: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use picked points to crop a local subset, then RANSAC-fit a plane.

    Returns:
        plane_model: (a,b,c,d) for ax+by+cz+d=0
        inlier_indices_global: indices into `points`
    """
    if len(picked_indices) < 3:
        raise ValueError("Need at least 3 picked points to fit a plane.")

    candidate: set[int] = set()
    for idx in picked_indices:
        nbrs = kd.query_ball_point(points[idx], r=float(neighbor_radius))
        candidate.update(nbrs)

    candidate_idx = np.array(sorted(candidate), dtype=np.int64)
    if candidate_idx.size < 50:
        candidate_idx = np.arange(points.shape[0], dtype=np.int64)

    pcd_local = o3d.geometry.PointCloud()
    pcd_local.points = o3d.utility.Vector3dVector(points[candidate_idx].astype(np.float64))
    plane_model, inliers_local = pcd_local.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iterations),
    )

    inliers_local = np.asarray(inliers_local, dtype=np.int64)
    inliers_global = candidate_idx[inliers_local]
    return np.asarray(plane_model, dtype=np.float64), inliers_global


def compute_align_transform_from_plane(
    points: np.ndarray, plane_model: np.ndarray, inliers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute T such that:
      1) plane normal aligns to +Z
      2) plane inliers mean height becomes z=0

    Returns:
      T (4,4), aligned_points (N,3)
    """
    normal = plane_model[:3].astype(np.float64)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-12:
        raise ValueError("Invalid plane normal.")
    normal = normal / n_norm

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if float(np.dot(normal, z_axis)) < 0.0:
        normal = -normal

    R_obj, _ = Rotation.align_vectors([z_axis], [normal])
    Rm = R_obj.as_matrix().astype(np.float64)

    rotated = (Rm @ points.T).T
    ground_z_mean = float(np.mean(rotated[inliers, 2]))

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[2, 3] = -ground_z_mean

    aligned = apply_transform_matrix(points, T)
    return T, aligned


# -----------------------------
# Picking via ray
# -----------------------------
def ray_aabb_intersection(origin: np.ndarray, direction: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray):
    """Return (tmin,tmax) for ray-AABB intersection; None if no hit."""
    inv = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    t0 = (bb_min - origin) * inv
    t1 = (bb_max - origin) * inv
    tmin = np.max(np.minimum(t0, t1))
    tmax = np.min(np.maximum(t0, t1))
    if tmax < max(tmin, 0.0):
        return None
    return float(max(tmin, 0.0)), float(tmax)


def pick_point_from_ray(
    points: np.ndarray,
    kd: cKDTree,
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    bb_min: np.ndarray,
    bb_max: np.ndarray,
    samples: int = 64,
) -> Optional[int]:
    """
    Approximate picking on point clouds:
      1) intersect ray with AABB to get [t0,t1]
      2) sample along the ray, query nearest neighbor, collect candidates
      3) choose candidate with smallest distance-to-ray
    """
    hit = ray_aabb_intersection(ray_origin, ray_dir, bb_min, bb_max)
    if hit is None:
        return None
    t0, t1 = hit
    if t1 <= t0:
        return None

    ts = np.linspace(t0, t1, samples, dtype=np.float64)
    candidate: set[int] = set()
    for t in ts:
        p = ray_origin + t * ray_dir
        _, idx = kd.query(p, k=1)
        candidate.add(int(idx))

    cand = np.fromiter(candidate, dtype=np.int64)
    if cand.size == 0:
        return None

    v = points[cand] - ray_origin[None, :]
    proj = (v @ ray_dir)[:, None] * ray_dir[None, :]
    perp = v - proj
    dist2 = np.sum(perp * perp, axis=1)
    return int(cand[int(np.argmin(dist2))])


# -----------------------------
# SMPL-X helpers
# -----------------------------
def transform_smplx_to_world(vertices_cam_m: np.ndarray, camera_pose_c2w: np.ndarray, scale_factor: float) -> np.ndarray:
    """Camera coords -> world coords. Scale translation by scale_factor (do NOT mutate input pose)."""
    R_cw = camera_pose_c2w[:3, :3]
    t_cw = camera_pose_c2w[:3, 3] * float(scale_factor)
    return (R_cw @ vertices_cam_m.T).T + t_cw


def load_scale_factor(scale_factors_path: str, smplx_results_path: Optional[str] = None) -> float:
    """
    Load and compute average scale factor across frames.

    The directory should contain files: `{frame:04d}_scale_factor.txt`.
    If `smplx_results_path` is provided and readable, only frames that exist in SMPL-X
    results are used.
    """
    scale_path = Path(scale_factors_path)

    if smplx_results_path is not None:
        smplx_path = Path(smplx_results_path)
        if smplx_path.exists():
            try:
                results = torch.load(smplx_path, map_location="cpu", weights_only=False)
                valid_frames = []
                for frame_idx, frame_data in enumerate(results):
                    if frame_data.get("vertices") is not None and frame_data.get("camera_pose") is not None:
                        valid_frames.append(frame_idx)
            except Exception as e:
                print(f"[WARNING] Could not load SMPL-X results for frame validation: {e}")
                valid_frames = None
        else:
            valid_frames = None
    else:
        valid_frames = None

    frames_to_check = valid_frames if valid_frames is not None else range(10000)
    values: list[float] = []
    for frame_idx in frames_to_check:
        p = scale_path / f"{frame_idx:04d}_scale_factor.txt"
        if not p.exists():
            continue
        try:
            with open(p, "r") as f:
                values.append(float(f.read().strip()))
        except Exception:
            continue

    if not values:
        raise FileNotFoundError(f"No valid scale factors found in {scale_path}")

    avg = float(np.mean(values))
    print(f"Computed average scale factor: {avg:.6f} (from {len(values)} frames)")
    return avg


def load_human_mesh_vertices_base(pt_path: str, scale_factors_path: str) -> Optional[np.ndarray]:
    """Load per-frame SMPL-X vertices in a consistent Z-up world-like frame (not ground-aligned)."""
    results_path = Path(pt_path)
    if not results_path.exists():
        print(f"[ground_alignment] Human mesh file not found: {results_path}")
        return None

    results = torch.load(results_path, map_location="cpu", weights_only=False)
    avg_scale_factor = load_scale_factor(scale_factors_path, pt_path)
    print("Note: Human mesh vertices keep 1.7m prior; scene scale is recovered separately.")

    vertices_seq: list[np.ndarray] = []
    for frame_data in results:
        frame_vertices = frame_data.get("vertices")
        camera_pose = frame_data.get("camera_pose")
        if frame_vertices is None or camera_pose is None:
            continue

        frame_vertices = np.asarray(frame_vertices, dtype=np.float64)
        camera_pose = np.asarray(camera_pose, dtype=np.float64)
        v_world = transform_smplx_to_world(frame_vertices, camera_pose, scale_factor=avg_scale_factor)
        v_world = opencv_to_z_up(v_world)
        vertices_seq.append(v_world.astype(np.float32))

    if not vertices_seq:
        return None
    return np.stack(vertices_seq, axis=0)


def load_smplx_faces(model_path: str) -> np.ndarray:
    smplx_model = smplx.create(model_path, model_type="smplx", gender="neutral", use_pca=False)
    return np.asarray(smplx_model.faces, dtype=np.int32)


def load_scene_pointcloud_from_rgbd(
    *,
    data_dir: str,
    depth_dir: str,
    smplx_results_dir: str,
    frame_indices: Optional[list[int]] = None,
    avg_scale_factor: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a fused scene point cloud from RGB, recovered depth, and camera poses.

    Expected files under `data_dir`:
      - `{frame:04d}_color.png`
      - `{frame:04d}_depth.png` (uint16, mm)
      - `{frame:04d}_mask.png`  (optional; mask>0 indicates human region)
      - `{frame:04d}_pose.txt` or `{frame:04d}_pose.pi3.txt` (4x4)

    Expected files under `smplx_results_dir`:
      - `{frame:04d}/smplx_params.pt` containing `cam_int`
    """
    data_dir_p = Path(data_dir)
    depth_dir_p = Path(depth_dir)
    smplx_results_dir_p = Path(smplx_results_dir)

    if frame_indices is None:
        frame_dirs = sorted(smplx_results_dir_p.glob("[0-9][0-9][0-9][0-9]"))
        frame_indices = [int(d.name) for d in frame_dirs]

    if avg_scale_factor is None:
        all_results = smplx_results_dir_p / "all_results_video.pt"
        avg_scale_factor = load_scale_factor(str(depth_dir_p), str(all_results) if all_results.exists() else None)

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for frame_idx in frame_indices:
        frame_id = f"{frame_idx:04d}"

        rgb_file = data_dir_p / f"{frame_id}_color.png"
        depth_file = data_dir_p / f"{frame_id}_depth.png"
        if not rgb_file.exists() or not depth_file.exists():
            continue

        rgb_image = cv2.imread(str(rgb_file))[:, :, ::-1]
        depth_map_m = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        # Mask convention (legacy): mask>0 indicates human region. We want to keep SCENE pixels.
        mask_file = data_dir_p / f"{frame_id}_mask.png"
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            scene_keep_mask = mask > 0
        else:
            scene_keep_mask = None

        pose_file = data_dir_p / f"{frame_id}_pose.txt"
        if not pose_file.exists():
            pose_file = data_dir_p / f"{frame_id}_pose.pi3.txt"
        if not pose_file.exists():
            continue

        try:
            camera_pose = np.loadtxt(str(pose_file), delimiter=",")
        except ValueError:
            camera_pose = np.loadtxt(str(pose_file), delimiter=" ")

        smplx_file = smplx_results_dir_p / frame_id / "smplx_params.pt"
        if not smplx_file.exists():
            continue
        smplx_data = torch.load(smplx_file, map_location="cpu", weights_only=False)
        cam_int = smplx_data["cam_int"].numpy()

        depth_map_scaled = depth_map_m * float(avg_scale_factor)
        camera_pose_scaled = camera_pose.copy()
        camera_pose_scaled[:3, 3] = camera_pose[:3, 3] * float(avg_scale_factor)

        pts, cols = create_pointcloud_from_rgbd(rgb_image, depth_map_scaled, cam_int, camera_pose_scaled, scene_keep_mask)
        all_points.append(pts)
        all_colors.append(cols)

    if not all_points:
        raise ValueError("No valid frames found for scene point cloud generation")

    points = np.vstack(all_points)
    colors_uint8 = np.vstack(all_colors)
    colors_float01 = colors_uint8.astype(np.float32) / 255.0

    # Only for debugging, delete later
    # points, colors_float01 = voxel_downsample(points, colors_float01, voxel_size=0.05)
    return points, colors_float01


def _write_transform_json(path: Path, T: np.ndarray, *, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transform_matrix": np.asarray(T, dtype=np.float64).round(8).tolist(),
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    T = np.asarray(data["transform_matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"transform_matrix in {path} must be 4x4, got {T.shape}")
    return T


def _downsample(points: np.ndarray, colors_float01: np.ndarray | None, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors_float01
    idx = np.random.choice(points.shape[0], size=int(max_points), replace=False)
    pts = points[idx]
    cols = colors_float01[idx] if colors_float01 is not None and colors_float01.size else None
    return pts, cols


def _z_stats(z: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size == 0:
        return {}
    ps = [0.0, 0.1, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0]
    vals = np.percentile(z, ps)
    return {f"p{p:g}": float(v) for p, v in zip(ps, vals)}


def _print_alignment_stats(points_zup: np.ndarray, points_aligned: np.ndarray) -> None:
    z0 = _z_stats(points_zup[:, 2] if points_zup.size else np.asarray([]))
    z1 = _z_stats(points_aligned[:, 2] if points_aligned.size else np.asarray([]))
    if z0:
        print(f"[ground_alignment] Z (z-up, pre-align) percentiles: {z0}")
    if z1:
        print(f"[ground_alignment] Z (post-align) percentiles: {z1}")
        if "p0.5" in z1:
            print(f"[ground_alignment] ground_est_z (p0.5, post-align): {z1['p0.5']:.6f}")


@dataclass(frozen=True)
class GroundAlignmentPreviewArgs:
    """Standalone preview for debugging whether ground is at z=0 after alignment."""

    seq: str
    robot: str = "g1"

    runs_root: Path = Path("/home/juyiang/data/holosoma_runs")
    """Must match your pipeline runs_root."""

    predicted_dir: Optional[Path] = None
    depth_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    """Override input dirs (otherwise inferred from standard HER layout)."""

    transform_json: Optional[Path] = None
    """Override transform.json path (otherwise uses runs_root/<seq>/<robot>/artifacts/ground_alignment/transform.json)."""

    max_points: int = 150_000
    point_size: float = 0.02
    show_original_zup: bool = True
    show_aligned: bool = True
    show_reference_plane: bool = True
    reference_plane_size: float = 4.0
    print_stats: bool = True


def preview(cfg: GroundAlignmentPreviewArgs) -> None:
    """
    Visualize original Z-up scene points vs aligned points using an existing transform.json.

    - 'original_zup' is OpenCV->Z-up only.
    - 'aligned' applies T_align to the Z-up points.
    - reference plane is z=0 in the aligned frame.
    """
    from omniretargeting.workflows.her.config import ManualPaths
    from omniretargeting.workflows.her.paths import get_sequence_paths

    manual = ManualPaths(runs_root=cfg.runs_root)
    paths = get_sequence_paths(seq=cfg.seq, robot=cfg.robot, manual=manual)

    predicted_dir = cfg.predicted_dir or paths.predicted_dir
    depth_dir = cfg.depth_dir or paths.depth_recovered
    results_dir = cfg.results_dir or paths.results_dir

    transform_json = cfg.transform_json or paths.ground_transform_json
    if not transform_json.exists():
        raise FileNotFoundError(f"transform_json not found: {transform_json}")
    T_align = _load_transform_json(transform_json)

    points_cv, colors_float01 = load_scene_pointcloud_from_rgbd(
        data_dir=str(predicted_dir),
        depth_dir=str(depth_dir),
        smplx_results_dir=str(results_dir),
    )
    points_zup = opencv_to_z_up(points_cv)
    points_aligned = apply_transform_matrix(points_zup, T_align)

    if cfg.print_stats:
        _print_alignment_stats(points_zup, points_aligned)

    pts0, cols0 = _downsample(points_zup, colors_float01, cfg.max_points)
    pts1, cols1 = _downsample(points_aligned, colors_float01, cfg.max_points)

    def _to_uint8(cols: np.ndarray | None):
        if cols is None or cols.size == 0:
            return None
        return (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)

    server = viser.ViserServer()
    server.scene.add_frame("/world_axes", show_axes=True)

    h0 = server.scene.add_point_cloud(
        "/scene/original_zup",
        points=pts0.astype(np.float32),
        colors=_to_uint8(cols0),
        point_size=float(cfg.point_size),
        visible=bool(cfg.show_original_zup),
    )
    h1 = server.scene.add_point_cloud(
        "/scene/aligned",
        points=pts1.astype(np.float32),
        colors=_to_uint8(cols1),
        point_size=float(cfg.point_size),
        visible=bool(cfg.show_aligned),
    )

    plane = None
    if cfg.show_reference_plane:
        size = float(cfg.reference_plane_size)
        V = np.array(
            [
                [-size, -size, 0.0],
                [size, -size, 0.0],
                [size, size, 0.0],
                [-size, size, 0.0],
            ],
            dtype=np.float32,
        )
        F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        plane = server.scene.add_mesh_simple(
            "/scene/z0_plane",
            vertices=V,
            faces=F,
            color=(200, 200, 255),
            opacity=0.35,
            visible=True,
        )

    gui = server.gui
    gui.add_markdown(f"**Transform**: `{transform_json}`")
    gui.add_markdown("Toggle clouds to verify that the aligned ground sits at z=0.")
    t0 = gui.add_checkbox("Show original (Z-up)", initial_value=bool(cfg.show_original_zup))
    t1 = gui.add_checkbox("Show aligned (T_align)", initial_value=bool(cfg.show_aligned))
    tp = gui.add_checkbox("Show z=0 plane", initial_value=bool(cfg.show_reference_plane))

    @t0.on_update
    def _(_evt):
        h0.visible = bool(t0.value)

    @t1.on_update
    def _(_evt):
        h1.visible = bool(t1.value)

    @tp.on_update
    def _(_evt):
        if plane is not None:
            plane.visible = bool(tp.value)

    while True:
        time.sleep(0.2)


@dataclass(frozen=True)
class GroundAlignmentAlignArgs:
    """Run the interactive alignment UI with standard HER paths (no pipeline wrapper needed)."""

    seq: str
    robot: str = "g1"
    runs_root: Path = Path("/home/juyiang/data/holosoma_runs")

    use_rgbd_scene: bool = True
    max_points: int = 200_000
    exit_process_on_save: bool = False


def align(cfg: GroundAlignmentAlignArgs) -> np.ndarray:
    """Launch the interactive picker+plane-fit tool and write outputs to runs_root."""
    from omniretargeting.workflows.her.config import ManualPaths
    from omniretargeting.workflows.her.paths import get_sequence_paths

    manual = ManualPaths(runs_root=cfg.runs_root)
    paths = get_sequence_paths(seq=cfg.seq, robot=cfg.robot, manual=manual)

    config = {
        "sequence": cfg.seq,
        "robot": cfg.robot,
        "smpl_model_path": str(manual.smpl_model_path),
        "human_mesh_path": str(paths.all_results_video),
        "scale_factors_path": str(paths.depth_recovered),
        "scene_ply_path": str(paths.fused_scene_ply),
        "default_save_path": str(paths.aligned_scene_ply),
        "data_dir": str(paths.predicted_dir),
        "depth_dir": str(paths.depth_recovered),
        "smplx_results_dir": str(paths.results_dir),
        "use_rgbd_scene": bool(cfg.use_rgbd_scene),
        "transform_output_path": str(paths.ground_transform_json),
        "exit_process_on_save": bool(cfg.exit_process_on_save),
        "max_points": int(cfg.max_points),
    }
    return main(config)


# -----------------------------
# State
# -----------------------------
@dataclass
class PlanePickState:
    picking: bool = False
    picked_indices: list[int] = None
    pick_markers: list = None
    plane_inliers: Optional[np.ndarray] = None
    T_align: Optional[np.ndarray] = None
    aligned_points: Optional[np.ndarray] = None
    adjusting_height: bool = False
    initial_T_align: Optional[np.ndarray] = None
    height_offset: float = 0.0

    def __post_init__(self):
        self.picked_indices = []
        self.pick_markers = []


# -----------------------------
# Main
# -----------------------------
def main(config: dict) -> np.ndarray:
    """
    Run the interactive ground alignment tool.

    Required keys (either-or):
      - use_rgbd_scene=True:
          data_dir, depth_dir, smplx_results_dir
        OR
      - use_rgbd_scene=False:
          scene_ply_path

    Optional:
      - human_mesh_path, scale_factors_path, smpl_model_path (for SMPL-X overlay)
      - default_save_path (for saving aligned point cloud)
      - transform_output_path (JSON output)
      - exit_process_on_save (bool)

    Returns:
      4x4 numpy transform matrix.
    """
    if not config:
        raise ValueError("ground_alignment.main(config) requires a config dict.")

    use_rgbd_scene = bool(config.get("use_rgbd_scene", False))
    scene_ply_path = config.get("scene_ply_path")
    default_save_path = config.get("default_save_path")

    data_dir = config.get("data_dir")
    depth_dir = config.get("depth_dir")
    smplx_results_dir = config.get("smplx_results_dir")

    human_mesh_path = config.get("human_mesh_path")
    scale_factors_path = config.get("scale_factors_path")
    smpl_model_path = config.get("smpl_model_path")

    transform_output_path = config.get("transform_output_path")
    exit_process_on_save = bool(config.get("exit_process_on_save", False))
    max_points = int(config.get("max_points", 200_000))

    # Load scene point cloud
    if use_rgbd_scene:
        if not (data_dir and depth_dir and smplx_results_dir):
            raise ValueError("use_rgbd_scene=True requires data_dir, depth_dir, smplx_results_dir")
        points, colors_float01 = load_scene_pointcloud_from_rgbd(
            data_dir=str(data_dir),
            depth_dir=str(depth_dir),
            smplx_results_dir=str(smplx_results_dir),
        )
        points = opencv_to_z_up(points)
    else:
        if not scene_ply_path:
            raise ValueError("scene_ply_path is required when use_rgbd_scene=False")
        pcd = o3d.io.read_point_cloud(str(scene_ply_path))
        points = np.asarray(pcd.points, dtype=np.float64)
        colors_float01 = np.asarray(pcd.colors, dtype=np.float64)
        points = opencv_to_z_up(points)

    if max_points > 0 and points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors_float01 = colors_float01[idx] if colors_float01 is not None and colors_float01.size else colors_float01
        print(f"[ground_alignment] Downsampled scene points to {max_points}")

    colors_uint8 = (np.clip(colors_float01, 0.0, 1.0) * 255.0).astype(np.uint8)
    colors_uint8_backup = colors_uint8.copy()

    kd = cKDTree(points)
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)

    # Optional SMPL-X overlay
    human_vertices_base = None
    human_faces = None
    if human_mesh_path and scale_factors_path:
        if not smpl_model_path:
            raise ValueError("smpl_model_path is required when loading SMPL-X overlay")
        human_vertices_base = load_human_mesh_vertices_base(str(human_mesh_path), str(scale_factors_path))
        if human_vertices_base is not None:
            human_faces = load_smplx_faces(str(smpl_model_path))

    server = viser.ViserServer()
    server.scene.add_frame("/world_axes", show_axes=True, position=(0.0, 0.0, 0.0))

    pc_original = server.scene.add_point_cloud(
        name="/cloud/original",
        points=points.astype(np.float32),
        colors=colors_uint8,
        point_size=0.02,
        visible=True,
    )
    pc_aligned = server.scene.add_point_cloud(
        name="/cloud/aligned",
        points=points.astype(np.float32),
        colors=colors_uint8_backup,
        point_size=0.02,
        visible=False,
    )

    human_handle = {"mesh": None}
    current_human_frame = {"idx": 0}
    animation_playing = False
    animation_thread = None
    animation_stop_event = threading.Event()
    fps = 30
    updating_slider_programmatically = False
    reference_plane = None

    if human_vertices_base is not None and human_faces is not None:
        server.scene.add_frame("/human", show_axes=False)
        human_handle["mesh"] = server.scene.add_mesh_simple(
            "/human/mesh",
            vertices=human_vertices_base[0],
            faces=human_faces,
            color=(200, 200, 200),
            opacity=0.8,
        )

    state = PlanePickState()
    saved_transform: dict[str, np.ndarray | None] = {"T": None}

    def restore_original_colors() -> None:
        nonlocal colors_uint8, pc_original
        colors_uint8 = colors_uint8_backup.copy()
        try:
            pc_original.colors = colors_uint8
        except Exception:
            pc_original.remove()
            pc_original = server.scene.add_point_cloud(
                name="/cloud/original",
                points=points.astype(np.float32),
                colors=colors_uint8,
                point_size=0.02,
                visible=True,
            )

    def clear_pick_markers() -> None:
        for h in state.pick_markers:
            try:
                h.remove()
            except Exception:
                pass
        state.pick_markers.clear()

    def create_reference_plane() -> None:
        nonlocal reference_plane
        if reference_plane is not None:
            try:
                reference_plane.remove()
            except Exception:
                pass

        plane_size = 20.0
        vertices = np.array(
            [
                [-plane_size, -plane_size, 0],
                [plane_size, -plane_size, 0],
                [plane_size, plane_size, 0],
                [-plane_size, plane_size, 0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        reference_plane = server.scene.add_mesh_simple(
            "/reference_plane",
            vertices=vertices,
            faces=faces,
            color=(200, 200, 200),
            opacity=1.0,
            wireframe=False,
        )

    def show_reference_plane(show: bool) -> None:
        if reference_plane is not None:
            reference_plane.visible = show

    def apply_height_adjustment(height_offset: float) -> np.ndarray:
        if state.initial_T_align is None:
            return state.T_align
        T_adjusted = state.initial_T_align.copy()
        T_adjusted[2, 3] += float(height_offset)
        return T_adjusted

    def recenter_transform_xy(T: np.ndarray) -> np.ndarray:
        aligned = apply_transform_matrix(points, T)
        center_xy = aligned[:, :2].mean(axis=0)
        T_new = T.copy()
        T_new[0, 3] -= float(center_xy[0])
        T_new[1, 3] -= float(center_xy[1])
        return T_new

    def update_aligned_cloud() -> None:
        if state.T_align is None:
            return
        aligned = apply_transform_matrix(points, state.T_align)
        state.aligned_points = aligned
        pc_aligned.points = aligned.astype(np.float32)
        pc_aligned.colors = colors_uint8_backup
        pc_aligned.visible = True
        pc_original.visible = False

        if human_handle["mesh"] is not None and human_vertices_base is not None:
            v = human_vertices_base[current_human_frame["idx"]].astype(np.float64)
            v = apply_transform_matrix(v, state.T_align).astype(np.float32)
            human_handle["mesh"].vertices = v

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        nonlocal animation_playing, animation_thread, fps, updating_slider_programmatically

        with client.gui.add_folder("Plane Alignment"):
            show_original = client.gui.add_checkbox("Show original cloud", initial_value=True)
            show_aligned = client.gui.add_checkbox("Show aligned cloud", initial_value=False)

            neighbor_radius = client.gui.add_slider(
                "Neighbor radius (m)", min=0.1, max=5.0, step=0.1, initial_value=1.0
            )
            dist_thresh = client.gui.add_slider(
                "Plane dist threshold (m)", min=0.005, max=0.10, step=0.005, initial_value=0.02
            )

            pick_btn = client.gui.add_button("Start picking plane (max 5)")
            clear_btn = client.gui.add_button("Clear picked points")
            fit_btn = client.gui.add_button("Fit plane + highlight + align")

            height_slider = client.gui.add_slider(
                "Height adjustment (m)", min=-2.0, max=2.0, step=0.01, initial_value=0.0
            )
            confirm_height_btn = client.gui.add_button("Confirm height")
            height_slider.visible = False
            confirm_height_btn.visible = False

            save_btn = client.gui.add_button("Save transform matrix")
            save_aligned_btn = client.gui.add_button("Save aligned point cloud (PLY)")

            pick_status = client.gui.add_markdown("Picked: **0 / 5**")

        if human_vertices_base is not None and human_faces is not None:
            with client.gui.add_folder("Human Mesh Animation"):
                play_pause_btn = client.gui.add_button("▶ Play")
                frame_slider = client.gui.add_slider(
                    "Frame", min=0, max=len(human_vertices_base) - 1, step=1, initial_value=0
                )
                fps_slider = client.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=30)
                frame_info = client.gui.add_markdown(f"Frame: 0/{len(human_vertices_base)-1}")
        else:
            play_pause_btn = None
            frame_slider = None
            fps_slider = None
            frame_info = None

        @show_original.on_update
        def _(_evt):
            pc_original.visible = bool(show_original.value)

        @show_aligned.on_update
        def _(_evt):
            pc_aligned.visible = bool(show_aligned.value)

        def update_pick_status() -> None:
            pick_status.content = f"Picked: **{len(state.picked_indices)} / 5**"

        if human_vertices_base is not None and human_faces is not None:

            def update_human_mesh_frame(frame_idx: int, update_slider: bool = True) -> None:
                nonlocal updating_slider_programmatically
                frame_idx = max(0, min(int(frame_idx), len(human_vertices_base) - 1))
                current_human_frame["idx"] = frame_idx

                if human_handle["mesh"] is not None:
                    v = human_vertices_base[frame_idx].astype(np.float64)
                    if state.T_align is not None:
                        v = apply_transform_matrix(v, state.T_align)
                    human_handle["mesh"].vertices = v.astype(np.float32)

                if frame_info is not None:
                    frame_info.content = f"Frame: {frame_idx}/{len(human_vertices_base)-1}"

                if update_slider and frame_slider is not None and not updating_slider_programmatically:
                    updating_slider_programmatically = True
                    try:
                        frame_slider.value = frame_idx
                    finally:
                        updating_slider_programmatically = False

            def animation_loop() -> None:
                while not animation_stop_event.is_set():
                    next_frame = (current_human_frame["idx"] + 1) % len(human_vertices_base)
                    update_human_mesh_frame(next_frame, update_slider=True)
                    dt = 1.0 / float(max(fps, 1))
                    # Allow responsive pause via event.
                    animation_stop_event.wait(dt)

            @play_pause_btn.on_click  # type: ignore[union-attr]
            def _(_evt):
                nonlocal animation_playing, animation_thread, fps
                fps = int(fps_slider.value) if fps_slider is not None else fps  # type: ignore[union-attr]

                if not animation_playing:
                    animation_playing = True
                    animation_stop_event.clear()
                    play_pause_btn.name = "⏸ Pause"  # type: ignore[union-attr]
                    # Avoid spawning multiple threads if the previous one hasn't fully exited.
                    if animation_thread is None or (not animation_thread.is_alive()):
                        animation_thread = threading.Thread(target=animation_loop, daemon=True)
                        animation_thread.start()
                else:
                    animation_playing = False
                    animation_stop_event.set()
                    play_pause_btn.name = "▶ Play"  # type: ignore[union-attr]
                    # Do not join here (can block UI thread); the worker exits via the event.

            @frame_slider.on_update  # type: ignore[union-attr]
            def _(_evt):
                if not animation_playing and not updating_slider_programmatically:
                    update_human_mesh_frame(int(frame_slider.value), update_slider=False)  # type: ignore[union-attr]

            @fps_slider.on_update  # type: ignore[union-attr]
            def _(_evt):
                nonlocal fps
                fps = int(fps_slider.value)  # type: ignore[union-attr]

        @clear_btn.on_click
        def _(_evt):
            state.picking = False
            state.picked_indices.clear()
            state.adjusting_height = False
            state.initial_T_align = None
            state.height_offset = 0.0
            clear_pick_markers()
            restore_original_colors()
            client.scene.remove_pointer_callback()
            update_pick_status()

            height_slider.visible = False
            confirm_height_btn.visible = False
            show_reference_plane(False)

            client.add_notification(title="Cleared", body="Picked points cleared; colors restored.")

        @pick_btn.on_click
        def _(_evt):
            state.picking = True
            state.picked_indices.clear()
            clear_pick_markers()
            restore_original_colors()
            state.plane_inliers = None
            state.T_align = None
            state.aligned_points = None
            pc_aligned.visible = False
            update_pick_status()

            client.add_notification(title="Picking enabled", body="Left-click points on the desired plane (up to 5).")

            @client.scene.on_pointer_event(event_type="click")
            def _(event: viser.ScenePointerEvent) -> None:
                if not state.picking:
                    return

                ray_origin = np.array(event.ray_origin, dtype=np.float64)
                ray_dir = np.array(event.ray_direction, dtype=np.float64)
                ray_dir_norm = np.linalg.norm(ray_dir)
                if ray_dir_norm < 1e-12:
                    return
                ray_dir = ray_dir / ray_dir_norm

                idx = pick_point_from_ray(points, kd, ray_origin, ray_dir, bb_min, bb_max)
                if idx is None or idx in state.picked_indices:
                    return

                state.picked_indices.append(idx)
                update_pick_status()

                marker = server.scene.add_icosphere(
                    name=f"/picks/p{len(state.picked_indices)}",
                    radius=0.03,
                    position=tuple(points[idx].astype(float)),
                    color=(255, 255, 0),
                )
                state.pick_markers.append(marker)

                if len(state.picked_indices) >= 5:
                    state.picking = False
                    client.scene.remove_pointer_callback()
                    client.add_notification(title="Picking done", body="Reached 5 points; now fit the plane.")

        @fit_btn.on_click
        def _(_evt):
            state.picking = False
            client.scene.remove_pointer_callback()

            if len(state.picked_indices) < 3:
                client.add_notification(title="Not enough points", body="Pick at least 3 points.", color="red")
                return

            restore_original_colors()

            try:
                plane_model, inliers = fit_plane_with_local_cloud(
                    points=points,
                    kd=kd,
                    picked_indices=state.picked_indices,
                    neighbor_radius=float(neighbor_radius.value),
                    distance_threshold=float(dist_thresh.value),
                )
            except Exception as e:
                client.add_notification(title="Fit failed", body=str(e), color="red")
                return

            state.plane_inliers = inliers

            # Highlight plane inliers in ORIGINAL view.
            colors_hl = colors_uint8_backup.copy()
            colors_hl[inliers] = np.array([255, 0, 0], dtype=np.uint8)
            colors_uint8[:] = colors_hl
            pc_original.colors = colors_uint8

            try:
                T, _aligned = compute_align_transform_from_plane(points, plane_model, inliers)
            except Exception as e:
                client.add_notification(title="Align failed", body=str(e), color="red")
                return

            T = recenter_transform_xy(T)
            state.initial_T_align = T
            state.T_align = T
            state.height_offset = 0.0
            state.adjusting_height = True

            height_slider.visible = True
            confirm_height_btn.visible = True
            height_slider.value = 0.0

            create_reference_plane()
            show_reference_plane(True)
            update_aligned_cloud()

            # Keep the legacy behavior: alignment switches to the aligned cloud view.
            show_original.value = False
            show_aligned.value = True

            client.add_notification(
                title="Initial alignment done",
                body="Adjust height with the slider, then confirm.",
                color="green",
            )

        @height_slider.on_update
        def _(_evt):
            if not state.adjusting_height:
                return
            state.height_offset = float(height_slider.value)
            state.T_align = apply_height_adjustment(state.height_offset)
            update_aligned_cloud()

        @confirm_height_btn.on_click
        def _(_evt):
            if not state.adjusting_height:
                return
            state.adjusting_height = False
            height_slider.visible = False
            confirm_height_btn.visible = False
            show_reference_plane(False)
            update_aligned_cloud()
            client.add_notification(title="Height confirmed", body="Final alignment applied.", color="green")

        @save_aligned_btn.on_click
        def _(_evt):
            if not default_save_path:
                client.add_notification(
                    title="No output path",
                    body="default_save_path not provided in config.",
                    color="red",
                )
                return
            if state.T_align is None:
                client.add_notification(title="Nothing to save", body="Fit a plane first.", color="red")
                return

            aligned = state.aligned_points
            if aligned is None:
                aligned = apply_transform_matrix(points, state.T_align)

            pcd_out = o3d.geometry.PointCloud()
            pcd_out.points = o3d.utility.Vector3dVector(aligned.astype(np.float64))
            pcd_out.colors = o3d.utility.Vector3dVector((colors_uint8_backup.astype(np.float64) / 255.0))
            out_path = Path(default_save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = o3d.io.write_point_cloud(str(out_path), pcd_out)
            if not ok:
                client.add_notification(title="Save failed", body=f"Failed writing: {out_path}", color="red")
                return
            client.add_notification(title="Saved", body=f"Aligned point cloud saved: {out_path}", color="green")

        @save_btn.on_click
        def _(_evt):
            if state.T_align is None:
                client.add_notification(title="Nothing to save", body="Fit a plane first.", color="red")
                return

            saved_transform["T"] = np.asarray(state.T_align, dtype=np.float64)

            if transform_output_path:
                out_path = Path(transform_output_path)
                _write_transform_json(
                    out_path,
                    saved_transform["T"],
                    extra={"sequence": config.get("sequence"), "robot": config.get("robot")},
                )
                print(f"Transform matrix saved to: {out_path}")
                client.add_notification(title="Saved", body=f"Transform saved: {out_path}", color="green")
            else:
                print("[WARNING] transform_output_path not provided; transform not written to disk.")
                client.add_notification(title="Saved", body="Transform saved (not written to disk).", color="green")

            if exit_process_on_save:
                import os

                def delayed_exit():
                    time.sleep(1)
                    os._exit(0)

                threading.Thread(target=delayed_exit, daemon=True).start()

    while True:
        if saved_transform["T"] is not None:
            return saved_transform["T"]
        time.sleep(0.2)


if __name__ == "__main__":
    # Two usage patterns:
    # 1) Pipeline: import and call `main(config_dict)` (interactive alignment).
    # 2) Standalone debug:
    #      python -m omniretargeting.workflows.her.ground_alignment.manual_viser preview --seq smooth --robot g1
    #    to visualize current transform.json and whether ground is at z=0.
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "preview":
        args = tyro.cli(GroundAlignmentPreviewArgs, args=sys.argv[2:])
        preview(args)
    elif len(sys.argv) >= 2 and sys.argv[1] == "align":
        args = tyro.cli(GroundAlignmentAlignArgs, args=sys.argv[2:])
        align(args)
    else:
        raise SystemExit(
            "Use `ground_alignment.main(config_dict)` from your pipeline, or run:\n"
            "  python -m omniretargeting.workflows.her.ground_alignment.manual_viser preview --seq smooth --robot g1\n"
            "  python -m omniretargeting.workflows.her.ground_alignment.manual_viser align --seq smooth --robot g1\n"
        )
