"""
Reconstruction + retarget visualization (customized).

This script visualizes:
- reconstructed scene point cloud (RGB-D fusion or fallback PLY),
- reconstructed SMPL-X human mesh,
- retargeted robot motion (qpos .npz),
in a single Viser viewer, reusing the pipeline artifacts saved under `holosoma_runs/`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import cv2
import open3d as o3d
import smplx
import torch
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from omniretargeting.workflows.her.config import ManualPaths, RobotType, SeqName
from omniretargeting.workflows.her.ground_alignment import manual_viser as ground_alignment
from omniretargeting.workflows.her.io.contact import load_contact_flags as load_contact_flags_npz
from omniretargeting.workflows.her.io.joints_map import OMNI_SMPLX22_JOINTS, remap_smplx_to_omni_smplx22
from omniretargeting.workflows.her.paths import get_sequence_paths
from omniretargeting.workflows.her.viz.viser_utils_recon import create_motion_control_sliders_with_callbacks


def _load_npz_qpos(npz_path: Path) -> tuple[np.ndarray, int]:
    data = np.load(str(npz_path), allow_pickle=True)
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data.files else 30
    return qpos, fps


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _apply_display_transform(points: np.ndarray, *, z_min: float, scale: float, apply_z_min: bool, apply_scale: bool) -> np.ndarray:
    pts = points.copy()
    if apply_z_min:
        pts[..., 2] -= float(z_min)
    if apply_scale:
        pts *= float(scale)
    return pts


def _downsample(points: np.ndarray, colors: np.ndarray | None, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if points.shape[0] <= max_points:
        return points, colors
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    pts = points[idx]
    cols = colors[idx] if colors is not None else None
    return pts, cols


def _opencv_to_zup_rotation(R_cv: np.ndarray) -> np.ndarray:
    # Map OpenCV axes to Z-up: x->x, y->-z, z->y (row-wise mapping).
    M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    return M @ R_cv


def _create_camera_frustum_lines(cam_pose: np.ndarray, size: float = 0.15):
    center = cam_pose[:3, 3]
    rotation = cam_pose[:3, :3]
    aspect = 4.0 / 3.0
    half_width = size * aspect
    half_height = size
    depth = size * 2
    corners_cam = np.array(
        [
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
        ]
    )
    corners_world = center + (rotation @ corners_cam.T).T
    lines = []
    for corner in corners_world:
        lines.append((center, corner))
    for i in range(4):
        lines.append((corners_world[i], corners_world[(i + 1) % 4]))
    return lines


_CUBE_VERTS_UNIT = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

_CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=np.int32,
)


def _boxes_to_mesh(centers: np.ndarray, half_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    half_sizes = np.asarray(half_sizes, dtype=np.float32).reshape(-1, 3)
    if centers.shape[0] != half_sizes.shape[0]:
        raise ValueError(f"centers ({centers.shape[0]}) and half_sizes ({half_sizes.shape[0]}) mismatch")

    n = int(centers.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    verts = np.empty((n * 8, 3), dtype=np.float32)
    faces = np.empty((n * 12, 3), dtype=np.int32)
    for i in range(n):
        v0 = i * 8
        f0 = i * 12
        verts[v0 : v0 + 8] = centers[i][None, :] + _CUBE_VERTS_UNIT * half_sizes[i][None, :]
        faces[f0 : f0 + 12] = _CUBE_FACES + v0
    return verts, faces


_SMPLX22_SKELETON = [
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Foot"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Foot"),
    ("Pelvis", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Spine3"),
    ("Spine3", "Neck"),
    ("Neck", "Head"),
    ("Spine3", "L_Collar"),
    ("L_Collar", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Spine3", "R_Collar"),
    ("R_Collar", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
]


_SMPLX_CONTACT_JOINT_IDS = [7, 10, 8, 11, 20, 21]


def _smplx22_skeleton_edges() -> list[tuple[int, int]]:
    name_to_idx = {name: idx for idx, name in enumerate(OMNI_SMPLX22_JOINTS)}
    edges = []
    for a, b in _SMPLX22_SKELETON:
        if a in name_to_idx and b in name_to_idx:
            edges.append((name_to_idx[a], name_to_idx[b]))
    return edges


def _build_skeleton_lines(
    joints_frame: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    c = np.array(color, dtype=np.uint8)
    for i, j in edges:
        points.append(np.stack([joints_frame[i], joints_frame[j]], axis=0))
        colors.append(np.stack([c, c], axis=0))
    if not points:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8)
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def _load_contact_flags(pt_path: str) -> Optional[np.ndarray]:
    results_path = Path(pt_path)
    if not results_path.exists():
        print(f"[viser_player_recon] Contact source not found: {results_path}")
        return None

    results = torch.load(results_path, map_location="cpu", weights_only=False)
    flags: list[np.ndarray] = []
    for frame_data in results:
        if not isinstance(frame_data, dict):
            flags.append(np.zeros(len(_SMPLX_CONTACT_JOINT_IDS), dtype=bool))
            continue
        conf_logit = frame_data.get("static_conf_logits")
        if conf_logit is None:
            flags.append(np.zeros(len(_SMPLX_CONTACT_JOINT_IDS), dtype=bool))
            continue
        conf_logit = np.asarray(conf_logit, dtype=np.float32).reshape(-1)
        if conf_logit.shape[0] != len(_SMPLX_CONTACT_JOINT_IDS):
            conf_logit = conf_logit[: len(_SMPLX_CONTACT_JOINT_IDS)]
        sigmoid_score = 1.0 / (1.0 + np.exp(-conf_logit))
        flags.append(sigmoid_score > 0.3)
    if not flags:
        return None
    return np.stack(flags, axis=0)


def _load_smplx_joints_base(
    pt_path: str, scale_factors_path: str, smpl_model_path: str
) -> Optional[np.ndarray]:
    """Load SMPL-X joints in Z-up world (not ground-aligned, no z_min, no scale)."""
    results_path = Path(pt_path)
    if not results_path.exists():
        print(f"[viser_player_recon] SMPL-X results not found: {results_path}")
        return None

    results = torch.load(results_path, map_location="cpu", weights_only=False)
    avg_scale_factor = ground_alignment.load_scale_factor(scale_factors_path, pt_path)

    has_joints = any(isinstance(frame, dict) and frame.get("joints") is not None for frame in results)
    smplx_model = None
    if not has_joints:
        smplx_model = smplx.create(
            model_path=str(smpl_model_path),
            model_type="smplx",
            gender="male",
            use_face_contour=True,
        )

    joints_seq: list[np.ndarray] = []
    for frame_data in results:
        if not isinstance(frame_data, dict):
            continue
        camera_pose = frame_data.get("camera_pose")
        if camera_pose is None:
            continue

        if has_joints:
            joints_cam = frame_data.get("joints")
            if joints_cam is None:
                continue
        else:
            pose = frame_data.get("pose")
            betas = frame_data.get("betas")
            transl = frame_data.get("transl")
            if pose is None or betas is None or transl is None:
                continue

            pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)
            betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
            transl_tensor = torch.tensor(transl, dtype=torch.float32).unsqueeze(0)

            global_orient = pose_tensor[:, :3]
            body_pose = pose_tensor[:, 3:66]
            jaw_pose = pose_tensor[:, 66:69]
            leye_pose = pose_tensor[:, 69:72]
            reye_pose = pose_tensor[:, 72:75]

            with torch.no_grad():
                out = smplx_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    jaw_pose=jaw_pose,
                    leye_pose=leye_pose,
                    reye_pose=reye_pose,
                    betas=betas_tensor,
                    transl=transl_tensor,
                    return_verts=False,
                )
                joints_cam = out.joints[0].detach().cpu().numpy()

        joints_cam = np.asarray(joints_cam, dtype=np.float64)
        camera_pose = np.asarray(camera_pose, dtype=np.float64)
        joints_world = ground_alignment.transform_smplx_to_world(joints_cam, camera_pose, scale_factor=avg_scale_factor)
        joints_world = ground_alignment.opencv_to_z_up(joints_world)
        try:
            joints_22 = remap_smplx_to_omni_smplx22(joints_world[None, ...])[0]
        except ValueError:
            joints_22 = remap_smplx_to_omni_smplx22(
                joints_world[None, ...],
                assume_input_is_omni_smplx22=True,
            )[0]
        joints_seq.append(joints_22.astype(np.float32))

    if not joints_seq:
        return None
    return np.stack(joints_seq, axis=0)

@dataclass
class ReconViewerConfig:
    seq: SeqName
    robot: RobotType = "g1"
    manual: ManualPaths = ManualPaths()

    scene_source: Literal["rgbd", "ply"] = "rgbd"
    world_mode: Literal["human", "retarget", "retarget_no_zmin", "retarget_no_scale"] = "retarget"
    """human: aligned reconstruction; retarget: apply z_min + scale; retarget_no_zmin/retarget_no_scale toggle each."""

    qpos_npz: Optional[Path] = None
    robot_urdf: Optional[str] = None
    # Downsampling (remote-friendly defaults)
    scene_max_points: int = 40_000
    human_max_points: int = 10_000
    frame_max_points: int = 20_000
    show_collision_proxy: bool = False
    """If True and pipeline provides a collision proxy, visualize it (useful to debug 'penetration' reports)."""
    show_collision_mesh: bool = False
    """If True and pipeline provides a collision mesh, visualize it (decimated if needed)."""
    collision_mesh_max_triangles: int = 200_000
    """If collision mesh is too dense, decimate to this triangle count for visualization."""


def main(cfg: ReconViewerConfig) -> None:
    paths = get_sequence_paths(seq=cfg.seq, robot=cfg.robot, manual=cfg.manual)
    run_summary = _load_json(paths.pipeline_config_json) if paths.pipeline_config_json.exists() else {}

    scale_factor = float(run_summary.get("scale_factor", 1.0))
    z_min_val = run_summary.get("vertical_bias_z_min_human_m", 0.0)
    z_min_human = float(z_min_val) if z_min_val is not None else 0.0
    T_align = np.asarray(_load_json(paths.ground_transform_json)["transform_matrix"], dtype=np.float64)

    qpos_path = cfg.qpos_npz or (paths.retarget_save_dir / f"{cfg.seq}.npz")
    qpos, fps = _load_npz_qpos(qpos_path)

    # Load scene (OpenCV-ish), convert to Z-up, then apply ground alignment transform.
    if cfg.scene_source == "rgbd":
        scene_points_cv, scene_colors = ground_alignment.load_scene_pointcloud_from_rgbd(
            data_dir=str(paths.predicted_dir),
            depth_dir=str(paths.depth_recovered),
            smplx_results_dir=str(paths.results_dir),
        )
        scene_points = ground_alignment.opencv_to_z_up(scene_points_cv)
        scene_points = ground_alignment.apply_transform_matrix(scene_points, T_align)
    else:
        pcd = o3d.io.read_point_cloud(str(paths.aligned_scene_ply))
        scene_points = np.asarray(pcd.points, dtype=np.float64)
        scene_colors = np.asarray(pcd.colors, dtype=np.float32)
    if scene_colors is None or scene_colors.size == 0:
        scene_colors = np.full((scene_points.shape[0], 3), 0.7, dtype=np.float32)
    scene_points_base = scene_points.copy()
    scene_colors_base = scene_colors.copy() if scene_colors is not None else None

    # Human mesh: base world (Z-up) then apply alignment.
    human_vertices = ground_alignment.load_human_mesh_vertices_base(str(paths.all_results_video), str(paths.depth_recovered))
    human_faces = None
    if human_vertices is not None:
        human_faces = ground_alignment.load_smplx_faces(str(cfg.manual.smpl_model_path))
        human_vertices = np.stack([ground_alignment.apply_transform_matrix(v, T_align) for v in human_vertices], axis=0)
    human_vertices_base = human_vertices.copy() if human_vertices is not None else None

    # Human joints: base world (Z-up) then apply alignment.
    human_joints_base = None
    human_joints = _load_smplx_joints_base(
        str(paths.all_results_video), str(paths.depth_recovered), str(cfg.manual.smpl_model_path)
    )
    if human_joints is not None:
        human_joints = np.stack([ground_alignment.apply_transform_matrix(j, T_align) for j in human_joints], axis=0)
    if human_joints is not None:
        human_joints_base = human_joints.copy()
    contact_path = paths.prepared_data_dir / f"{cfg.seq}_contact.npz"
    contact_flags = load_contact_flags_npz(
        contact_path,
        num_frames=human_joints.shape[0] if human_joints is not None else None,
    )
    if contact_flags is None:
        contact_flags = _load_contact_flags(str(paths.all_results_video))

    scene_points = _apply_display_transform(
        scene_points_base, z_min=0, scale=scale_factor, apply_z_min=False, apply_scale=True
    )
    if human_vertices_base is not None:
        human_vertices = _apply_display_transform(
            human_vertices_base, z_min=z_min_human, scale=scale_factor, apply_z_min=False, apply_scale=True
        )
    if human_joints_base is not None:
        human_joints = _apply_display_transform(
            human_joints_base, z_min=z_min_human, scale=scale_factor, apply_z_min=False, apply_scale=True
        )

    # Viewer
    server = viser.ViserServer()
    server.scene.add_frame("/world_axes", show_axes=True)

    # Scene point cloud
    scene_points, scene_colors = _downsample(scene_points, scene_colors, cfg.scene_max_points)
    colors_uint8 = (np.clip(scene_colors, 0.0, 1.0) * 255.0).astype(np.uint8) if scene_colors is not None else None
    scene_handle = server.scene.add_point_cloud(
        "/scene/cloud",
        points=scene_points.astype(np.float32),
        colors=colors_uint8,
        point_size=0.02,
        visible=True,
    )

    # Collision mesh (from ply2scene output), decimated for remote-friendly visualization.
    collision_mesh_handle = {"mesh": None}
    collision_mesh_path = run_summary.get("terrain_mesh_path")
    if collision_mesh_path is not None:
        mesh_path = Path(str(collision_mesh_path))
        if mesh_path.exists():
            try:
                mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
                if mesh_o3d is not None and len(mesh_o3d.triangles) > 0:
                    target_tris = int(getattr(cfg, "collision_mesh_max_triangles", 200_000))
                    if target_tris > 0 and len(mesh_o3d.triangles) > target_tris:
                        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_tris)
                    mesh_o3d.remove_duplicated_vertices()
                    mesh_o3d.remove_duplicated_triangles()
                    mesh_o3d.remove_degenerate_triangles()
                    mesh_o3d.remove_unreferenced_vertices()
                    mesh_o3d.compute_vertex_normals()

                    verts = np.asarray(mesh_o3d.vertices, dtype=np.float32)
                    faces = np.asarray(mesh_o3d.triangles, dtype=np.int32)

                    # Viewer uses retarget world: scale by scale_factor (robot/human).
                    verts = _apply_display_transform(
                        verts, z_min=0.0, scale=scale_factor, apply_z_min=False, apply_scale=True
                    ).astype(np.float32)

                    collision_mesh_handle["mesh"] = server.scene.add_mesh_simple(
                        "/scene/collision_mesh",
                        vertices=verts,
                        faces=faces,
                        color=(160, 240, 120),
                        opacity=0.15,
                        visible=bool(cfg.show_collision_mesh),
                    )
            except Exception as exc:
                print(f"[viser_player_recon] Failed to load collision mesh {mesh_path}: {exc}")

    # Collision proxy (if available)
    collision_proxy_handle = {"mesh": None}
    collision_proxy_npz = run_summary.get("collision_proxy_npz")
    if collision_proxy_npz is not None:
        proxy_path = Path(str(collision_proxy_npz))
        if proxy_path.exists():
            proxy = np.load(str(proxy_path), allow_pickle=True)
            centers = np.asarray(proxy["centers"], dtype=np.float32)
            half_sizes = np.asarray(proxy["half_sizes"], dtype=np.float32)
            verts, faces = _boxes_to_mesh(centers, half_sizes)
            if verts.shape[0] > 0 and faces.shape[0] > 0:
                collision_proxy_handle["mesh"] = server.scene.add_mesh_simple(
                    "/scene/collision_proxy",
                    vertices=verts,
                    faces=faces,
                    color=(60, 160, 255),
                    opacity=0.08,
                    visible=bool(cfg.show_collision_proxy),
                )

    # Human mesh
    human_handle = {"mesh": None}
    if human_vertices is not None and human_faces is not None:
        server.scene.add_frame("/human", show_axes=False)
        human_handle["mesh"] = server.scene.add_mesh_simple(
            "/human/mesh",
            vertices=human_vertices[0].astype(np.float32),
            faces=human_faces,
            color=(200, 200, 200),
            opacity=0.7,
        )
    # Human point cloud
    human_pc_handle = {"pc": None}
    # Human skeleton
    skeleton_handle = {"lines": None}
    joints_handle = {"pc": None}
    skeleton_edges = _smplx22_skeleton_edges() if human_joints is not None else []
    contact_joint_indices: list[int] = []
    if human_joints is not None:
        for smplx_id in _SMPLX_CONTACT_JOINT_IDS:
            if smplx_id < human_joints.shape[1]:
                contact_joint_indices.append(smplx_id)

    if human_joints is not None:
        default_joint_color = np.array([255, 200, 0], dtype=np.uint8)
        contact_joint_color = np.array([160, 0, 200], dtype=np.uint8)
        joints_colors = np.tile(default_joint_color, (human_joints.shape[1], 1))
        if contact_flags is not None and contact_joint_indices:
            frame_idx = 0 if contact_flags.shape[0] > 0 else None
            if frame_idx is not None:
                flags = contact_flags[frame_idx]
                for flag_idx, joint_idx in enumerate(contact_joint_indices):
                    if flag_idx < flags.shape[0] and bool(flags[flag_idx]):
                        joints_colors[joint_idx] = contact_joint_color
        joints_handle["pc"] = server.scene.add_point_cloud(
            "/human/joints",
            points=human_joints[0].astype(np.float32),
            colors=joints_colors,
            point_size=0.015,
            visible=True,
        )
        if skeleton_edges:
            line_pts, line_cols = _build_skeleton_lines(human_joints[0], skeleton_edges, color=(200, 200, 200))
            skeleton_handle["lines"] = server.scene.add_line_segments(
                "/human/skeleton",
                points=line_pts,
                colors=line_cols,
                line_width=2.0,
                visible=True,
            )

    # Robot
    if cfg.robot_urdf is not None:
        robot_urdf_path = cfg.robot_urdf
    else:
        robot_urdf_path = run_summary.get("robot_urdf_path")
        if robot_urdf_path is None:
            raise ValueError(
                "robot_urdf is not provided and pipeline_config.json has no 'robot_urdf_path'. "
                "Please pass --robot-urdf explicitly."
            )
    robot_urdf = yourdfpy.URDF.load(robot_urdf_path, load_meshes=True, build_scene_graph=True)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot")
    robot_dof = len(vr.get_actuated_joint_limits())

    # Per-frame projected cloud state
    frame_cloud_handle = {"pc": None}
    camera_frame_handle = {"frame": None}

    avg_scene_scale = ground_alignment.load_scale_factor(str(paths.depth_recovered), str(paths.all_results_video))

    @lru_cache(maxsize=256)
    def _load_frame_cloud(frame_idx: int, keep_human: bool) -> tuple[np.ndarray, np.ndarray] | None:
        frame_id = f"{frame_idx:04d}"
        rgb_file = paths.predicted_dir / f"{frame_id}_color.png"
        depth_file = paths.predicted_dir / f"{frame_id}_depth.png"
        if not rgb_file.exists() or not depth_file.exists():
            return None

        rgb_image = cv2.imread(str(rgb_file))[:, :, ::-1]
        depth_map_m = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_file = paths.predicted_dir / f"{frame_id}_mask.png"
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if keep_human:
                keep_mask = (mask == 0).astype(np.uint8)
            else:
                keep_mask = (mask > 0).astype(np.uint8)
        else:
            keep_mask = None

        pose_file = paths.predicted_dir / f"{frame_id}_pose.txt"
        if not pose_file.exists():
            pose_file = paths.predicted_dir / f"{frame_id}_pose.pi3.txt"
        if not pose_file.exists():
            return None

        try:
            camera_pose = np.loadtxt(str(pose_file), delimiter=",")
        except ValueError:
            camera_pose = np.loadtxt(str(pose_file), delimiter=" ")

        smplx_file = paths.results_dir / frame_id / "smplx_params.pt"
        if not smplx_file.exists():
            return None

        smplx_data = ground_alignment.torch.load(str(smplx_file), map_location="cpu", weights_only=False)  # type: ignore[attr-defined]
        cam_int = smplx_data["cam_int"].numpy()

        depth_map_scaled = depth_map_m * float(avg_scene_scale)
        camera_pose_scaled = camera_pose.copy()
        camera_pose_scaled[:3, 3] = camera_pose[:3, 3] * float(avg_scene_scale)

        pts, cols = ground_alignment.create_pointcloud_from_rgbd(
            rgb_image, depth_map_scaled, cam_int, camera_pose_scaled, keep_mask
        )

        pts = ground_alignment.opencv_to_z_up(pts)
        pts = ground_alignment.apply_transform_matrix(pts, T_align)
        cols = cols.astype(np.float32) / 255.0
        return pts, cols

    def _update_camera_frame(frame_idx: int) -> None:
        frame_id = f"{frame_idx:04d}"
        pose_file = paths.predicted_dir / f"{frame_id}_pose.txt"
        if not pose_file.exists():
            pose_file = paths.predicted_dir / f"{frame_id}_pose.pi3.txt"
        if not pose_file.exists():
            return
        try:
            camera_pose = np.loadtxt(str(pose_file), delimiter=",")
        except ValueError:
            camera_pose = np.loadtxt(str(pose_file), delimiter=" ")
        camera_pose[:3, 3] = camera_pose[:3, 3] * float(avg_scene_scale)
        cam_pos = ground_alignment.opencv_to_z_up(camera_pose[:3, 3].reshape(1, 3)).reshape(3)
        cam_pos = ground_alignment.apply_transform_matrix(cam_pos[None, :], T_align)[0]
        cam_pos = _apply_display_transform(
            cam_pos[None, :], z_min=0.0, scale=scale_factor, apply_z_min=False, apply_scale=True
        )[0]
        if camera_frame_handle["frame"] is None:
            camera_frame_handle["frame"] = server.scene.add_frame(
                "/camera/frame",
                show_axes=True,
                position=cam_pos,
                axes_length=0.1,
            )
        else:
            camera_frame_handle["frame"].position = cam_pos

    def on_frame(i: int) -> None:
        if human_handle["mesh"] is not None and human_vertices is not None:
            human_handle["mesh"].vertices = human_vertices[i].astype(np.float32)

        if human_pc_handle["pc"] is not None:
            human_pts = _load_frame_cloud(i, keep_human=True)
            if human_pts is not None:
                pts_raw, cols_raw = human_pts
                pts = _apply_display_transform(
                    pts_raw, z_min=z_min_human, scale=scale_factor, apply_z_min=False, apply_scale=True
                )
                pts, cols = pts, cols_raw
                pts, cols = _downsample(pts, cols_raw, cfg.human_max_points)
                human_pc_handle["pc"].points = pts.astype(np.float32)
                if cols is not None:
                    human_pc_handle["pc"].colors = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)

        if joints_handle["pc"] is not None and human_joints_base is not None:
            joints_frame = _apply_display_transform(
                human_joints_base[i], z_min=z_min_human, scale=scale_factor, apply_z_min=False, apply_scale=True
            )
            joints_handle["pc"].points = joints_frame.astype(np.float32)
            if contact_flags is not None and contact_joint_indices:
                default_joint_color = np.array([255, 200, 0], dtype=np.uint8)
                contact_joint_color = np.array([160, 0, 200], dtype=np.uint8)
                colors = np.tile(default_joint_color, (joints_frame.shape[0], 1))
                frame_idx = min(i, contact_flags.shape[0] - 1)
                flags = contact_flags[frame_idx]
                for flag_idx, joint_idx in enumerate(contact_joint_indices):
                    if flag_idx < flags.shape[0] and bool(flags[flag_idx]):
                        colors[joint_idx] = contact_joint_color
                joints_handle["pc"].colors = colors
            if skeleton_handle["lines"] is not None and skeleton_edges:
                line_pts, _ = _build_skeleton_lines(joints_frame, skeleton_edges, color=(200, 200, 200))
                skeleton_handle["lines"].points = line_pts

        if frame_cloud_handle["pc"] is not None:
            frame_pts = _load_frame_cloud(i, keep_human=False)
            if frame_pts is not None:
                pts_raw, cols_raw = frame_pts
                pts = _apply_display_transform(
                    pts_raw, z_min=0, scale=scale_factor, apply_z_min=False, apply_scale=True
                )
                pts, cols = pts, cols_raw
                pts, cols = _downsample(pts, cols_raw, cfg.frame_max_points)
                frame_cloud_handle["pc"].points = pts.astype(np.float32)
                if cols is not None:
                    frame_cloud_handle["pc"].colors = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)

        if camera_frame_handle["frame"] is not None:
            _update_camera_frame(i)

    create_motion_control_sliders_with_callbacks(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        initial_fps=fps,
        # IMPORTANT: Interpolation can introduce in-between-frame penetrations even if every saved
        # frame is collision-feasible. Default to discrete playback.
        initial_interp_mult=1,
        loop=True,
        on_frame=on_frame,
    )

    # ---------- UI toggles ----------
    with server.gui.add_folder("Scene / Mesh"):
        show_scene_cb = server.gui.add_checkbox("Show scene cloud", initial_value=True)
        show_frame_cb = server.gui.add_checkbox("Show frame cloud", initial_value=False)
        show_human_mesh_cb = server.gui.add_checkbox("Show human mesh", initial_value=human_handle["mesh"] is not None)
        show_human_pc_cb = server.gui.add_checkbox("Show human point cloud", initial_value=False)
        show_human_skeleton_cb = server.gui.add_checkbox(
            "Show human skeleton", initial_value=joints_handle["pc"] is not None
        )
        show_collision_mesh_cb = None
        if collision_mesh_handle["mesh"] is not None:
            show_collision_mesh_cb = server.gui.add_checkbox(
                "Show collision mesh", initial_value=bool(cfg.show_collision_mesh)
            )
        show_proxy_cb = None
        if collision_proxy_handle["mesh"] is not None:
            show_proxy_cb = server.gui.add_checkbox(
                "Show collision proxy", initial_value=bool(cfg.show_collision_proxy)
            )
        show_robot_mesh_cb = server.gui.add_checkbox("Show robot meshes", initial_value=True)
        show_camera_cb = server.gui.add_checkbox("Show camera", initial_value=False)

    @show_scene_cb.on_update
    def _(_evt):
        scene_handle.visible = bool(show_scene_cb.value)

    if collision_mesh_handle["mesh"] is not None and show_collision_mesh_cb is not None:
        @show_collision_mesh_cb.on_update
        def _(_evt):  # type: ignore[no-redef]
            collision_mesh_handle["mesh"].visible = bool(show_collision_mesh_cb.value)

    if collision_proxy_handle["mesh"] is not None and show_proxy_cb is not None:
        @show_proxy_cb.on_update
        def _(_evt):  # type: ignore[no-redef]
            collision_proxy_handle["mesh"].visible = bool(show_proxy_cb.value)

    @show_frame_cb.on_update
    def _(_evt):
        if show_frame_cb.value:
            if frame_cloud_handle["pc"] is None:
                frame_cloud_handle["pc"] = server.scene.add_point_cloud(
                    "/scene/frame_cloud",
                    points=np.zeros((1, 3), dtype=np.float32),
                    colors=np.zeros((1, 3), dtype=np.uint8),
                    point_size=0.02,
                    visible=True,
                )
            else:
                frame_cloud_handle["pc"].visible = True
            on_frame(0)
        else:
            if frame_cloud_handle["pc"] is not None:
                frame_cloud_handle["pc"].visible = False

    @show_human_mesh_cb.on_update
    def _(_evt):
        if human_handle["mesh"] is not None:
            human_handle["mesh"].visible = bool(show_human_mesh_cb.value)

    @show_human_pc_cb.on_update
    def _(_evt):
        if show_human_pc_cb.value:
            if human_pc_handle["pc"] is None:
                human_pts = _load_frame_cloud(0, keep_human=True)
                if human_pts is None:
                    return
                pts_raw, cols_raw = human_pts
                pts = _apply_display_transform(
                    pts_raw, z_min=z_min_human, scale=scale_factor, apply_z_min=False, apply_scale=True
                )
                pts, cols = pts, cols_raw
                pts, cols = _downsample(pts, cols_raw, cfg.human_max_points)
                colors = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8) if cols is not None else None
                human_pc_handle["pc"] = server.scene.add_point_cloud(
                    "/human/points",
                    points=pts.astype(np.float32),
                    colors=colors,
                    point_size=0.01,
                    visible=True,
                )
            elif human_pc_handle["pc"] is not None:
                human_pc_handle["pc"].visible = True
            on_frame(0)
        else:
            if human_pc_handle["pc"] is not None:
                human_pc_handle["pc"].visible = False

    @show_human_skeleton_cb.on_update
    def _(_evt):
        visible = bool(show_human_skeleton_cb.value)
        if joints_handle["pc"] is not None:
            joints_handle["pc"].visible = visible
        if skeleton_handle["lines"] is not None:
            skeleton_handle["lines"].visible = visible

    @show_robot_mesh_cb.on_update
    def _(_evt):
        vr.show_visual = bool(show_robot_mesh_cb.value)

    @show_camera_cb.on_update
    def _(_evt):
        if show_camera_cb.value:
            _update_camera_frame(0)
            if camera_frame_handle["frame"] is not None:
                camera_frame_handle["frame"].visible = True
        else:
            if camera_frame_handle["frame"] is not None:
                camera_frame_handle["frame"].visible = False

    print("[viser_player_recon] Open the viewer URL printed above. Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(ReconViewerConfig))
