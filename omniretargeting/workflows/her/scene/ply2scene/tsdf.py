from __future__ import annotations

"""
TSDF fusion helper for HER workflow in omniretargeting.

This module is designed to be imported by `omniretargeting.workflows.her.scene.ply2scene.convert` and
keeps only the essentials needed for project-internal TSDF fusion.

Implementation choices:
- Uses Open3D's built-in TSDF integration (ScalableTSDFVolume).
- Reads intrinsics from `smplx_params.pt` (`cam_int`), consistent with convert's RGB-D path.
- Applies the same mask + morphology + depth-gradient filtering semantics as convert:
  - `mask > 0` keeps static/background pixels; other pixels are zeroed out in depth.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import open3d as o3d

from omniretargeting.workflows.her.ground_alignment import manual_viser as ground_alignment

if TYPE_CHECKING:  # pragma: no cover
    from omniretargeting.workflows.her.scene.ply2scene.convert import Ply2SceneConfig


def _depth_gradient_mask(depth_m: np.ndarray, threshold: float) -> np.ndarray:
    if depth_m.ndim != 2:
        return np.ones_like(depth_m, dtype=bool)
    dy, dx = np.gradient(depth_m)
    grad = np.sqrt(dx * dx + dy * dy)
    return grad < float(threshold)


def _load_pose_txt(path: Path) -> np.ndarray:
    try:
        pose = np.loadtxt(str(path), delimiter=",")
    except ValueError:
        pose = np.loadtxt(str(path), delimiter=" ")
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose at {path} must be 4x4, got {pose.shape}")
    return pose


def fuse_tsdf_mesh(
    paths,
    T_align: np.ndarray,
    *,
    cfg_like: "Ply2SceneConfig",
    voxel_length_m: float = 0.01,
    sdf_trunc_m: float = 0.1,
    depth_trunc_m: float = 12.0,
    integrate_stride: int = 1,
) -> o3d.geometry.TriangleMesh:
    """
    Fuse a scene mesh from RGB-D frames using TSDF integration.

    Coordinate convention:
    - Integrate in the OpenCV world coordinate system using camera-to-world poses.
    - After extraction, convert vertices to Z-up and apply `T_align` (ground alignment).
    """
    if not hasattr(o3d.pipelines.integration, "ScalableTSDFVolume"):
        raise RuntimeError(
            "Open3D ScalableTSDFVolume not available in this Open3D build; "
            "please upgrade Open3D to use TSDF fusion."
        )

    avg_scene_scale = ground_alignment.load_scale_factor(str(paths.depth_recovered), str(paths.all_results_video))
    frame_dirs = sorted(paths.results_dir.glob("[0-9][0-9][0-9][0-9]"))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length_m),
        sdf_trunc=float(sdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    stride = max(1, int(integrate_stride))
    for frame_dir in frame_dirs[::stride]:
        frame_id = frame_dir.name

        rgb_file = paths.predicted_dir / f"{frame_id}_color.png"
        depth_file = paths.predicted_dir / f"{frame_id}_depth.png"
        mask_file = paths.predicted_dir / f"{frame_id}_mask.png"
        pose_file = paths.predicted_dir / f"{frame_id}_pose.txt"
        if not pose_file.exists():
            pose_file = paths.predicted_dir / f"{frame_id}_pose.pi3.txt"

        if not rgb_file.exists() or not depth_file.exists() or not pose_file.exists():
            continue

        smplx_file = frame_dir / "smplx_params.pt"
        if not smplx_file.exists():
            continue

        rgb_image = cv2.imread(str(rgb_file))[:, :, ::-1].copy()
        depth_map_m = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        camera_pose = _load_pose_txt(pose_file)
        camera_pose_scaled = camera_pose.copy()
        camera_pose_scaled[:3, 3] = camera_pose[:3, 3] * float(avg_scene_scale)

        smplx_data = ground_alignment.torch.load(str(smplx_file), map_location="cpu", weights_only=False)  # type: ignore[attr-defined]
        cam_int = np.asarray(smplx_data["cam_int"].numpy(), dtype=np.float64)

        depth_map_scaled = depth_map_m * float(avg_scene_scale)

        valid_mask = depth_map_scaled > 0
        if float(cfg_like.depth_gradient_thr) > 0:
            valid_mask = valid_mask & _depth_gradient_mask(depth_map_scaled, float(cfg_like.depth_gradient_thr))

        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            scene_keep_mask = mask > 0
            if int(cfg_like.morphology_kernel_size) > 1:
                k = int(cfg_like.morphology_kernel_size)
                kernel = np.ones((k, k), np.uint8)
                scene_keep_mask = cv2.erode(scene_keep_mask.astype(np.uint8), kernel) > 0
            scene_keep_mask = scene_keep_mask & valid_mask
        else:
            scene_keep_mask = valid_mask

        depth_filtered = depth_map_scaled.copy()
        depth_filtered[~scene_keep_mask] = 0.0

        H, W = depth_filtered.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(W),
            int(H),
            float(cam_int[0, 0]),
            float(cam_int[1, 1]),
            float(cam_int[0, 2]),
            float(cam_int[1, 2]),
        )

        color_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_filtered.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc_m),
            convert_rgb_to_intensity=False,
        )

        # Open3D expects world-to-camera extrinsics; our poses are camera-to-world.
        extrinsic = np.linalg.inv(camera_pose_scaled)
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = ground_alignment.opencv_to_z_up(vertices)
    vertices = ground_alignment.apply_transform_matrix(vertices, np.asarray(T_align, dtype=np.float64))
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))

    mesh.compute_vertex_normals()
    return mesh
