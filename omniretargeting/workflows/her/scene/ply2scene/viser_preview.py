from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import open3d as o3d
import tyro
import viser  # type: ignore[import-not-found]

from omniretargeting.workflows.her.config import ManualPaths, RobotType, SeqName
from omniretargeting.workflows.her.ground_alignment import manual_viser as ground_alignment
from omniretargeting.workflows.her.paths import get_sequence_paths
from omniretargeting.workflows.her.viz.mujoco_utils import _world_mesh_from_geom


@dataclass
class PreviewConfig:
    seq: SeqName
    robot: RobotType = "g1"
    manual: ManualPaths = ManualPaths()

    scene_obj: Optional[Path] = None
    scene_xml: Optional[Path] = None
    scene_ply: Optional[Path] = None
    scene_source: Literal["rgbd", "fused_ply", "aligned_ply", "ply"] = "rgbd"

    max_points: int = 150_000
    show_points: bool = False
    apply_scale_factor: bool = True
    scale_factor: Optional[float] = None

    point_size: float = 0.02
    mesh_opacity: float = 0.6


def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    T = np.asarray(data["transform_matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"transform_matrix in {path} must be 4x4, got {T.shape}")
    return T


def _load_scale_factor(path: Path) -> float:
    if not path.exists():
        return 1.0
    with open(path, "r") as f:
        data = json.load(f)
    return float(data.get("scale_factor", 1.0))


def _points_from_rgbd(paths, T_align: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    points_cv, colors = ground_alignment.load_scene_pointcloud_from_rgbd(
        data_dir=str(paths.predicted_dir),
        depth_dir=str(paths.depth_recovered),
        smplx_results_dir=str(paths.results_dir),
    )
    points = ground_alignment.opencv_to_z_up(points_cv)
    points = ground_alignment.apply_transform_matrix(points, T_align)
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    return points.astype(np.float64), colors


def _points_from_ply(path: Path, *, apply_z_up: bool, T_align: Optional[np.ndarray], max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"PLY not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    if apply_z_up:
        points = ground_alignment.opencv_to_z_up(points)
    if T_align is not None:
        points = ground_alignment.apply_transform_matrix(points, T_align)
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    return points.astype(np.float64), colors


def _load_mesh_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, faces


def _load_mujoco_meshes(scene_xml: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        import mujoco  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"mujoco import failed: {exc}") from exc

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        V, F = _world_mesh_from_geom(model, data, gid, name)
        if V is None or F is None or V.size == 0 or F.size == 0:
            continue
        meshes.append((V.astype(np.float32), F.astype(np.int32)))
    return meshes


def main(cfg: PreviewConfig) -> None:
    paths = get_sequence_paths(seq=cfg.seq, robot=cfg.robot, manual=cfg.manual)
    T_align = _load_transform_json(paths.ground_transform_json)

    scale = cfg.scale_factor
    if scale is None:
        scale = _load_scale_factor(paths.pipeline_config_json)

    server = viser.ViserServer()
    server.scene.add_frame("/world_axes", show_axes=True)

    pc_handle = None
    if cfg.show_points:
        if cfg.scene_source == "rgbd":
            points, colors = _points_from_rgbd(paths, T_align, cfg.max_points)
        else:
            if cfg.scene_source == "fused_ply":
                ply_path = paths.fused_scene_ply
                apply_z_up = True
                apply_T = True
            elif cfg.scene_source == "aligned_ply":
                ply_path = paths.aligned_scene_ply
                apply_z_up = False
                apply_T = False
            else:
                ply_path = cfg.scene_ply
                apply_z_up = True
                apply_T = True
            if ply_path is None:
                raise ValueError("scene_ply must be provided when scene_source='ply'")
            points, colors = _points_from_ply(
                ply_path,
                apply_z_up=apply_z_up,
                T_align=T_align if apply_T else None,
                max_points=cfg.max_points,
            )
        if cfg.apply_scale_factor:
            points *= float(scale)

        colors_uint8 = None
        if colors is not None and colors.size > 0:
            colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif points.size > 0:
            colors_uint8 = np.full((points.shape[0], 3), 180, dtype=np.uint8)

        pc_handle = server.scene.add_point_cloud(
            "/scene/points",
            points=points.astype(np.float32),
            colors=colors_uint8,
            point_size=float(cfg.point_size),
            visible=True,
        )

    mesh_handle = {"mesh": None}
    if cfg.scene_obj is not None:
        v, f = _load_mesh_vertices(cfg.scene_obj)
        if cfg.apply_scale_factor:
            v = v * float(scale)
        mesh_handle["mesh"] = server.scene.add_mesh_simple(
            "/scene/mesh_obj",
            vertices=v,
            faces=f,
            color=(180, 180, 180),
            opacity=float(cfg.mesh_opacity),
            side=2,  # 2 = THREE.DoubleSide, show both sides of the mesh
        )

    mjcf_handles: list = []
    if cfg.scene_xml is not None:
        try:
            mjcf_meshes = _load_mujoco_meshes(cfg.scene_xml)
        except Exception as exc:
            print(f"[ply2scene preview] MJCF load failed: {exc}")
            mjcf_meshes = []
        for i, (v, f) in enumerate(mjcf_meshes):
            h = server.scene.add_mesh_simple(
                f"/scene/mesh_mjcf_{i}",
                vertices=v,
                faces=f,
                color=(50, 150, 255),
                opacity=float(cfg.mesh_opacity)
            )
            mjcf_handles.append(h)

    with server.gui.add_folder("Display"):
        show_pc = server.gui.add_checkbox("Show points", initial_value=pc_handle is not None)
        show_obj = server.gui.add_checkbox("Show obj mesh", initial_value=mesh_handle["mesh"] is not None)
        show_mjcf = server.gui.add_checkbox("Show mjcf meshes", initial_value=len(mjcf_handles) > 0)
        point_size = server.gui.add_slider("Point size", min=0.002, max=0.08, step=0.002, initial_value=cfg.point_size)
        opacity = server.gui.add_slider("Mesh opacity", min=0.05, max=1.0, step=0.05, initial_value=cfg.mesh_opacity)

    @show_pc.on_update
    def _(_evt):
        if pc_handle is not None:
            pc_handle.visible = bool(show_pc.value)

    @show_obj.on_update
    def _(_evt):
        if mesh_handle["mesh"] is not None:
            mesh_handle["mesh"].visible = bool(show_obj.value)

    @show_mjcf.on_update
    def _(_evt):
        for h in mjcf_handles:
            h.visible = bool(show_mjcf.value)

    @point_size.on_update
    def _(_evt):
        if pc_handle is not None:
            pc_handle.point_size = float(point_size.value)

    @opacity.on_update
    def _(_evt):
        if mesh_handle["mesh"] is not None:
            mesh_handle["mesh"].opacity = float(opacity.value)
        for h in mjcf_handles:
            h.opacity = float(opacity.value)

    print("[ply2scene preview] Open the viewer URL printed above. Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(PreviewConfig))
