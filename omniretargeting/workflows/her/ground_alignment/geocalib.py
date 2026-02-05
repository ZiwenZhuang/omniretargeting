from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from . import manual_viser as ground_alignment


def _parse_frame_indices(spec: Optional[str]) -> Optional[list[int]]:
    if spec is None:
        return None
    items: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            items.extend(list(range(start, end + 1)))
        else:
            try:
                items.append(int(part))
            except ValueError:
                continue
    return sorted(set(items)) if items else None


def _discover_frame_indices(predicted_dir: Path) -> list[int]:
    frames: list[int] = []
    for p in sorted(predicted_dir.glob("*_color.png")):
        stem = p.stem
        if "_" not in stem:
            continue
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            frames.append(int(prefix))
    return sorted(set(frames))


def _load_pose(predicted_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    frame_id = f"{frame_idx:04d}"
    pose_file = predicted_dir / f"{frame_id}_pose.txt"
    if not pose_file.exists():
        pose_file = predicted_dir / f"{frame_id}_pose.pi3.txt"
    if not pose_file.exists():
        return None
    try:
        pose = np.loadtxt(str(pose_file), delimiter=",")
    except ValueError:
        pose = np.loadtxt(str(pose_file), delimiter=" ")
    if pose.shape != (4, 4):
        return None
    return pose


def _load_rgb(predicted_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    rgb_file = predicted_dir / f"{frame_idx:04d}_color.png"
    if not rgb_file.exists():
        return None
    rgb_bgr = cv2.imread(str(rgb_file))
    if rgb_bgr is None:
        return None
    return rgb_bgr[:, :, ::-1].copy()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _compute_plane_candidates(
    points: np.ndarray,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    angle_deg: float,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    remaining = np.arange(points.shape[0], dtype=np.int64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    best_area = -1.0
    best_plane: Optional[np.ndarray] = None
    best_inliers: Optional[np.ndarray] = None
    fallback_plane: Optional[np.ndarray] = None
    fallback_inliers: Optional[np.ndarray] = None

    for _ in range(int(max_candidates)):
        if remaining.size < 50:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[remaining].astype(np.float64))
        plane_model, inliers_local = pcd.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        inliers_local = np.asarray(inliers_local, dtype=np.int64)
        if inliers_local.size == 0:
            break
        inliers = remaining[inliers_local]

        if fallback_plane is None:
            fallback_plane = np.asarray(plane_model, dtype=np.float64)
            fallback_inliers = inliers

        normal = np.asarray(plane_model[:3], dtype=np.float64)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            remaining = np.setdiff1d(remaining, inliers)
            continue
        normal = normal / n_norm
        if float(np.dot(normal, z_axis)) < 0.0:
            normal = -normal

        angle = float(np.degrees(np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))))
        if angle <= float(angle_deg):
            inlier_xy = points[inliers][:, :2]
            area = 0.0
            if inlier_xy.shape[0] >= 3:
                try:
                    area = float(ConvexHull(inlier_xy).volume)
                except Exception:
                    xy_min = inlier_xy.min(axis=0)
                    xy_max = inlier_xy.max(axis=0)
                    area = float(np.prod(np.maximum(0.0, xy_max - xy_min)))
            if area > best_area:
                best_area = area
                best_plane = np.asarray(plane_model, dtype=np.float64)
                best_inliers = inliers

        remaining = np.setdiff1d(remaining, inliers)

    if best_plane is None or best_inliers is None:
        if fallback_plane is None or fallback_inliers is None:
            raise RuntimeError("Plane fitting failed.")
        return fallback_plane, fallback_inliers
    return best_plane, best_inliers


@dataclass(frozen=True)
class GeoCalibGroundAlignmentConfig:
    predicted_dir: Path
    depth_dir: Path
    results_dir: Path
    fused_scene_ply: Optional[Path] = None
    use_rgbd_scene: bool = True

    num_geocalib_frames: int = 12
    geocalib_frame_stride: int = 5
    geocalib_frame_indices: Optional[str] = None
    geocalib_weights: str = "pinhole"
    geocalib_camera_y_up: bool = False
    geocalib_device: Optional[str] = None
    geocalib_angle_outlier_deg: float = 15.0

    max_points: int = 200_000
    voxel_size: float = 0.05
    plane_distance_threshold: float = 0.03
    plane_ransac_n: int = 3
    plane_num_iterations: int = 2000
    plane_angle_deg: float = 10.0
    plane_max_candidates: int = 10
    recenter_xy: bool = True


@dataclass(frozen=True)
class GeoCalibGroundAlignmentResult:
    T_align: np.ndarray
    z_up_dir: np.ndarray
    frame_indices: list[int]
    ground_z: float

    # Optional debug payloads (set when return_points=True)
    points_zup: Optional[np.ndarray] = None
    colors_uint8: Optional[np.ndarray] = None
    points_zup_ds: Optional[np.ndarray] = None
    inliers_ds: Optional[np.ndarray] = None


def compute_geocalib_ground_alignment(
    cfg: GeoCalibGroundAlignmentConfig,
    *,
    return_points: bool = False,
    debug: bool = False,
) -> GeoCalibGroundAlignmentResult:
    if cfg.use_rgbd_scene:
        points_cv, colors_float01 = ground_alignment.load_scene_pointcloud_from_rgbd(
            data_dir=str(cfg.predicted_dir),
            depth_dir=str(cfg.depth_dir),
            smplx_results_dir=str(cfg.results_dir),
        )
    else:
        if cfg.fused_scene_ply is None:
            raise ValueError("fused_scene_ply is required when use_rgbd_scene=False")
        pcd = o3d.io.read_point_cloud(str(cfg.fused_scene_ply))
        points_cv = np.asarray(pcd.points, dtype=np.float64)
        colors_float01 = np.asarray(pcd.colors, dtype=np.float64)

    points_zup = ground_alignment.opencv_to_z_up(points_cv)

    if cfg.max_points > 0 and points_zup.shape[0] > cfg.max_points:
        idx = np.random.choice(points_zup.shape[0], size=cfg.max_points, replace=False)
        points_zup = points_zup[idx]
        if colors_float01 is not None and colors_float01.size:
            colors_float01 = colors_float01[idx]

    frame_indices = _parse_frame_indices(cfg.geocalib_frame_indices)
    if frame_indices is None:
        frame_indices = _discover_frame_indices(cfg.predicted_dir)
        if cfg.geocalib_frame_stride > 1:
            frame_indices = frame_indices[:: cfg.geocalib_frame_stride]
        if cfg.num_geocalib_frames > 0 and len(frame_indices) > cfg.num_geocalib_frames:
            sample_idx = np.linspace(0, len(frame_indices) - 1, cfg.num_geocalib_frames).astype(int)
            frame_indices = [frame_indices[i] for i in sample_idx]

    if cfg.geocalib_device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.geocalib_device

    try:
        from geocalib import GeoCalib
    except Exception as exc:
        raise RuntimeError(f"geocalib import failed: {exc}") from exc

    model = GeoCalib(weights=cfg.geocalib_weights).to(device)
    model.eval()

    g_list: list[np.ndarray] = []
    for idx in frame_indices:
        rgb = _load_rgb(cfg.predicted_dir, idx)
        if rgb is None:
            continue
        pose = _load_pose(cfg.predicted_dir, idx)
        if pose is None:
            continue

        img = torch.from_numpy(rgb).float().to(device) / 255.0
        img = img.permute(2, 0, 1)
        result = model.calibrate(img)

        gravity = result["gravity"]
        g_cam = gravity.vec3d[0].detach().cpu().numpy().astype(np.float64)
        if cfg.geocalib_camera_y_up:
            g_cam = np.array([g_cam[0], -g_cam[1], g_cam[2]], dtype=np.float64)

        g_world_cv = pose[:3, :3] @ g_cam
        g_world_zup = ground_alignment.opencv_to_z_up(g_world_cv[None, :])[0]
        g_list.append(_normalize(g_world_zup))

    if not g_list:
        raise RuntimeError("No valid frames for GeoCalib gravity estimation.")

    g_avg = _normalize(np.sum(np.stack(g_list, axis=0), axis=0))
    if cfg.geocalib_angle_outlier_deg > 0 and len(g_list) >= 3:
        cos_thr = float(np.cos(np.deg2rad(cfg.geocalib_angle_outlier_deg)))
        keep = [g for g in g_list if float(np.dot(g, g_avg)) >= cos_thr]
        if len(keep) >= 2:
            g_avg = _normalize(np.sum(np.stack(keep, axis=0), axis=0))

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if float(np.dot(g_avg, z_axis)) < 0.0:
        g_avg = -g_avg
    z_up_dir = g_avg
    if debug:
        print(f"[geocalib_ground_alignment] Estimated up (world Z-up): {z_up_dir.tolist()}")

    R_obj, _ = Rotation.align_vectors([z_axis], [z_up_dir])
    R_g = R_obj.as_matrix().astype(np.float64)
    if debug:
        print(f"[geocalib_ground_alignment] R_g @ z_up_dir = {(R_g @ z_up_dir).tolist()}")

    points_zup_ds = points_zup
    if cfg.voxel_size > 0:
        points_zup_ds, _ = ground_alignment.voxel_downsample(points_zup_ds, None, voxel_size=cfg.voxel_size)
    points_rot_ds = (R_g @ points_zup_ds.T).T

    _plane_model, inliers = _compute_plane_candidates(
        points_rot_ds,
        distance_threshold=cfg.plane_distance_threshold,
        ransac_n=cfg.plane_ransac_n,
        num_iterations=cfg.plane_num_iterations,
        angle_deg=cfg.plane_angle_deg,
        max_candidates=cfg.plane_max_candidates,
    )

    ground_z = float(np.mean(points_rot_ds[inliers, 2]))
    if debug:
        print(f"[geocalib_ground_alignment] ground_z (after gravity rotation) = {ground_z:.6f}")

    T_align = np.eye(4, dtype=np.float64)
    T_align[:3, :3] = R_g
    T_align[2, 3] = -ground_z

    if cfg.recenter_xy:
        aligned_all = ground_alignment.apply_transform_matrix(points_zup, T_align)
        center_xy = aligned_all[:, :2].mean(axis=0)
        T_align[0, 3] -= float(center_xy[0])
        T_align[1, 3] -= float(center_xy[1])

    colors_uint8 = None
    if colors_float01 is not None and colors_float01.size:
        colors_uint8 = (np.clip(colors_float01, 0.0, 1.0) * 255.0).astype(np.uint8)

    return GeoCalibGroundAlignmentResult(
        T_align=T_align,
        z_up_dir=z_up_dir,
        frame_indices=frame_indices,
        ground_z=ground_z,
        points_zup=points_zup if return_points else None,
        colors_uint8=colors_uint8 if return_points else None,
        points_zup_ds=points_zup_ds if return_points else None,
        inliers_ds=inliers if return_points else None,
    )


def save_transform_json(path: Path, T: np.ndarray, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transform_matrix": np.asarray(T, dtype=np.float64).round(8).tolist(),
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
