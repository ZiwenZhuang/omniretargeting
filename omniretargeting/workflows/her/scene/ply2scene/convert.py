from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import open3d as o3d
import tyro

from omniretargeting.workflows.her.config import ManualPaths, RobotType, SeqName
from omniretargeting.workflows.her.ground_alignment import manual_viser as ground_alignment
from omniretargeting.workflows.her.paths import get_sequence_paths


@dataclass
class Ply2SceneConfig:
    seq: SeqName
    robot: RobotType = "g1"
    manual: ManualPaths = ManualPaths()

    output_dir: Optional[Path] = None
    """Defaults to <run_dir>/artifacts/ply2scene/scene."""

    # Point cloud generation
    max_points: int = 2_000_000
    roi_half_extent_m: tuple[float, float, float] = (0.8, 0.8, 1)
    disable_roi_crop: bool = False
    """If True, skip ROI cropping and use the full accumulated point cloud."""

    depth_gradient_thr: float = 0
    """Depth gradient filtering threshold. Set <=0 to disable."""

    collision_z_min_m: float | None = None
    """If set, drop ALL input point-cloud points with z < this threshold (meters, z-up)."""

    keep_largest_component: bool = False
    """If True, keep only the largest connected component in post-processing."""
    # Morphological operations for mask edge filtering
    morphology_kernel_size: int = 0
    """Size of the rectangular kernel for morphological operations on mask edges."""

    # Density control + normals--nksr-config
    voxel_size: float = 0.05
    voxel_max_points_per_cell: int = 40
    normal_radius: float = 0.1
    normal_max_nn: int = 60
    orient_normals_k: int = 60

    # Reconstruction
    mesh_method: Literal["auto", "poisson", "nksr", "tsdf"] = "tsdf"
    """Mesh reconstruction backend. 'auto' prefers NKSR if available, otherwise falls back to Poisson."""

    use_ball_pivoting: bool = False
    poisson_depth_coarse: int = 7
    poisson_depth_final: int = 9
    poisson_density_quantile: float = 0.02
    bpa_radii: tuple[float, float, float] = (0.03, 0.06, 0.12)

    # NKSR (VideoMimic-style two-round reconstruction)
    nksr_config: str = "ks"
    """NKSR pretrained config name (default matches VideoMimic)."""

    nksr_checkpoint_path: Optional[Path] = None
    """Optional local checkpoint file to avoid downloading (copied into torch hub cache if needed)."""

    nksr_detail_level: float = 0.1
    nksr_mise_iter: int = 1
    nksr_two_round: bool = True
    nksr_device: Literal["auto", "cpu", "cuda"] = "cuda"
    nksr_max_points: Optional[int] = None
    """If set, randomly subsample input points for NKSR (speed/memory)."""

    hole_fill_resolution: int = 512
    hole_fill_knn: int = 64
    hole_fill_power: float = 1
    hole_fill_top_z_margin_m: float = 5.0
    hole_fill_max_points: int = 2_000_000
    hole_fill_use_convex_hull: bool = False

    # Post-processing
    crop_to_aabb: bool = True

    # Scale handling
    mesh_scale_factor: Optional[float] = None
    """If None, read scale_factor from pipeline_config_json; used in URDF/MJCF mesh scale tags."""

    # MJCF includes
    write_includes: bool = False


def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    T = np.asarray(data["transform_matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"transform_matrix in {path} must be 4x4, got {T.shape}")
    return T


def _load_scale_factor(path: Path) -> float:
    with open(path, "r") as f:
        data = json.load(f)
    return float(data.get("scale_factor", 1.0))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _depth_gradient_mask(depth_m: np.ndarray, threshold: float) -> np.ndarray:
    if depth_m.ndim != 2:
        return np.ones_like(depth_m, dtype=bool)
    dy, dx = np.gradient(depth_m)
    grad = np.sqrt(dx * dx + dy * dy)
    return grad < float(threshold)


def _voxel_cap_sample(
    points: np.ndarray, colors: np.ndarray | None, voxel_size: float, max_per_voxel: int
) -> tuple[np.ndarray, np.ndarray | None]:
    if points.size == 0:
        return points, colors
    voxel_coords = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    keep_indices: list[int] = []
    for voxel_id in np.unique(inverse):
        idx = np.where(inverse == voxel_id)[0]
        if idx.size <= max_per_voxel:
            keep_indices.extend(idx.tolist())
        else:
            chosen = np.random.choice(idx, size=int(max_per_voxel), replace=False)
            keep_indices.extend(chosen.tolist())
    keep = np.asarray(keep_indices, dtype=np.int64)
    pts = points[keep]
    cols = colors[keep] if colors is not None else None
    return pts, cols


def _points_from_rgbd(
    paths, T_align: np.ndarray, cfg: Ply2SceneConfig
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    avg_scene_scale = ground_alignment.load_scale_factor(str(paths.depth_recovered), str(paths.all_results_video))
    frame_dirs = sorted(paths.results_dir.glob("[0-9][0-9][0-9][0-9]"))

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    first_center: np.ndarray | None = None

    for frame_dir in frame_dirs:
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

        rgb_image = cv2.imread(str(rgb_file))[:, :, ::-1]
        depth_map_m = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        try:
            camera_pose = np.loadtxt(str(pose_file), delimiter=",")
        except ValueError:
            camera_pose = np.loadtxt(str(pose_file), delimiter=" ")

        smplx_data = ground_alignment.torch.load(str(smplx_file), map_location="cpu", weights_only=False)  # type: ignore[attr-defined]
        cam_int = smplx_data["cam_int"].numpy()
        vertices = np.asarray(smplx_data.get("vertices"), dtype=np.float64)

        depth_map_scaled = depth_map_m * float(avg_scene_scale)
        camera_pose_scaled = camera_pose.copy()
        camera_pose_scaled[:3, 3] = camera_pose[:3, 3] * float(avg_scene_scale)

        valid_mask = depth_map_scaled > 0
        if float(cfg.depth_gradient_thr) > 0:
            valid_mask = valid_mask & _depth_gradient_mask(depth_map_scaled, float(cfg.depth_gradient_thr))

        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            scene_keep_mask = mask > 0
            # Apply morphological erosion to remove noisy edge pixels
            if cfg.morphology_kernel_size > 1:
                kernel = np.ones((cfg.morphology_kernel_size, cfg.morphology_kernel_size), np.uint8)
                scene_keep_mask = cv2.erode(scene_keep_mask.astype(np.uint8), kernel) > 0
            if scene_keep_mask is not None:
                scene_keep_mask = scene_keep_mask & valid_mask
        else:
            scene_keep_mask = valid_mask

        pts, cols = ground_alignment.create_pointcloud_from_rgbd(
            rgb_image, depth_map_scaled, cam_int, camera_pose_scaled, scene_keep_mask
        )

        if pts.size == 0:
            continue

        pts = ground_alignment.opencv_to_z_up(pts)
        pts = ground_alignment.apply_transform_matrix(pts, T_align)

        if vertices.size > 0:
            v_world = ground_alignment.transform_smplx_to_world(vertices, camera_pose_scaled, scale_factor=1.0)
            v_world = ground_alignment.opencv_to_z_up(v_world)
            v_world = ground_alignment.apply_transform_matrix(v_world, T_align)
            center = np.median(v_world, axis=0)
        else:
            center = np.median(pts, axis=0)
        if first_center is None:
            first_center = center.astype(np.float64)

        extent = np.asarray(cfg.roi_half_extent_m, dtype=np.float64)
        if extent.shape != (3,):
            raise ValueError(f"roi_half_extent_m must be a 3-tuple (x, y, z), got {extent}")
        if not cfg.disable_roi_crop:
            keep = np.all(np.abs(pts - center[None, :]) <= extent[None, :], axis=1)
            pts = pts[keep]
            cols = cols[keep] if cols is not None else None

            if pts.size == 0:
                continue

        all_points.append(pts)
        if cols is not None:
            all_colors.append(cols.astype(np.float32) / 255.0)

    if not all_points:
        raise ValueError("No valid frames found for point cloud generation.")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors) if all_colors else None

    points, colors = _voxel_cap_sample(points, colors, cfg.voxel_size, int(cfg.voxel_max_points_per_cell))

    if points.shape[0] > cfg.max_points:
        idx = np.random.choice(points.shape[0], size=cfg.max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    return points.astype(np.float64), colors, first_center


def _prepare_point_cloud(
    points: np.ndarray, colors: np.ndarray | None, cfg: Ply2SceneConfig
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and colors.size > 0:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(cfg.normal_radius),
            max_nn=int(cfg.normal_max_nn),
        )
    )
    if cfg.orient_normals_k > 0:
        pcd.orient_normals_consistent_tangent_plane(int(cfg.orient_normals_k))
    return pcd


def _select_mesh_method(cfg: Ply2SceneConfig) -> Literal["poisson", "nksr", "tsdf"]:
    if cfg.mesh_method != "auto":
        return cfg.mesh_method

    try:
        import nksr  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return "poisson"
    return "nksr"


def _orient_normals_towards_top_view(
    points: np.ndarray, cfg: Ply2SceneConfig, *, camera_xy: tuple[float, float] | None = None
) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(cfg.normal_radius),
            max_nn=int(cfg.normal_max_nn),
        )
    )

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    top_z = float(bbox.max_bound[2] + cfg.hole_fill_top_z_margin_m)
    if camera_xy is None:
        camera_location = (float(center[0]), float(center[1]), top_z)
    else:
        camera_location = (float(camera_xy[0]), float(camera_xy[1]), top_z)

    # VideoMimic uses orient_normals_towards_camera_location() (global view point).
    pcd.orient_normals_towards_camera_location(camera_location)
    return np.asarray(pcd.normals).astype(np.float64)


def _nksr_reconstruct(
    points: np.ndarray, normals: np.ndarray, cfg: Ply2SceneConfig, *, reconstructor=None
) -> tuple[o3d.geometry.TriangleMesh, object]:
    import torch
    import nksr

    if cfg.nksr_device == "cuda":
        device = torch.device("cuda")
    elif cfg.nksr_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if reconstructor is None:
        config_to_use: str | dict = cfg.nksr_config
        if cfg.nksr_checkpoint_path is not None:
            # NKSR's configs.py supports local paths if `url` is a filesystem path.
            # The checkpoint is expected to be a dict with a `state_dict` entry; if not,
            # wrap it into that format in /tmp to keep NKSR's loading path unchanged.
            ckpt_obj = torch.load(str(cfg.nksr_checkpoint_path), map_location="cpu")
            if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
                ckpt_path = cfg.nksr_checkpoint_path
            elif isinstance(ckpt_obj, dict):
                ckpt_path = Path("/tmp") / f"nksr_wrapped_{cfg.nksr_checkpoint_path.stem}.pth"
                torch.save({"state_dict": ckpt_obj}, str(ckpt_path))
            else:
                raise ValueError(
                    f"Unsupported NKSR checkpoint format at {cfg.nksr_checkpoint_path}: expected dict or dict['state_dict']"
                )

            config_to_use = {"parent": cfg.nksr_config, "url": str(ckpt_path)}

        reconstructor = nksr.Reconstructor(device, config=config_to_use)

    input_xyz = torch.from_numpy(points.astype(np.float32)).to(device)
    input_normal = torch.from_numpy(normals.astype(np.float32)).to(device)

    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=float(cfg.nksr_detail_level))
    dual_mesh = field.extract_dual_mesh(mise_iter=int(cfg.nksr_mise_iter))
    vertices = dual_mesh.v.detach().cpu().numpy()
    faces = dual_mesh.f.detach().cpu().numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    return mesh, reconstructor


def _fill_holes_topdown(points_mesh: o3d.geometry.TriangleMesh, cfg: Ply2SceneConfig) -> np.ndarray:
    """
    VideoMimic-style hole filling:
    - Top-down ray casting to get a height map over XY
    - Fill misses inside footprint convex hull via IDW interpolation
    Returns infilled points (x, y, z).
    """
    if len(points_mesh.triangles) == 0 or len(points_mesh.vertices) == 0:
        return np.empty((0, 3), dtype=np.float64)

    bbox = points_mesh.get_axis_aligned_bounding_box()
    min_x, min_y, _ = bbox.min_bound
    max_x, max_y, max_z = bbox.max_bound

    top_z = float(max_z + cfg.hole_fill_top_z_margin_m)
    resolution = int(cfg.hole_fill_resolution)
    xs = np.linspace(float(min_x), float(max_x), resolution, dtype=np.float64)
    ys = np.linspace(float(min_y), float(max_y), resolution, dtype=np.float64)
    sample_xs, sample_ys = np.meshgrid(xs, ys, indexing="xy")
    origins = np.stack([sample_xs.ravel(), sample_ys.ravel(), np.full(sample_xs.size, top_z)], axis=1)
    directions = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (origins.shape[0], 1))

    hit_mask: np.ndarray
    hit_z: np.ndarray

    # Prefer Open3D tensor raycasting (fast); fall back to trimesh otherwise.
    use_o3d_t = hasattr(o3d, "t") and hasattr(getattr(o3d, "t", None), "geometry")
    if use_o3d_t:
        try:
            rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(points_mesh)  # type: ignore[attr-defined]
            scene = o3d.t.geometry.RaycastingScene()  # type: ignore[attr-defined]
            _ = scene.add_triangles(tmesh)
            ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
            t_hit = ans["t_hit"].numpy()
            hit_mask = np.isfinite(t_hit) & (t_hit > 0)
            hit_z = top_z - t_hit[hit_mask].astype(np.float64)
        except Exception:
            use_o3d_t = False

    if not use_o3d_t:
        import trimesh

        tri = trimesh.Trimesh(
            vertices=np.asarray(points_mesh.vertices),
            faces=np.asarray(points_mesh.triangles),
            process=False,
        )
        locations, index_ray, _ = tri.ray.intersects_location(origins.astype(np.float64), directions.astype(np.float64))
        hit_mask = np.zeros(origins.shape[0], dtype=bool)
        hit_z_full = np.full(origins.shape[0], np.nan, dtype=np.float64)
        if len(locations) > 0:
            hit_mask[index_ray] = True
            hit_z_full[index_ray] = locations[:, 2].astype(np.float64)
        hit_z = hit_z_full[hit_mask]

    if int(hit_mask.sum()) < 16:
        return np.empty((0, 3), dtype=np.float64)

    hit_xy = origins[hit_mask, :2]
    miss_xy = origins[~hit_mask, :2]
    if miss_xy.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    if cfg.hole_fill_use_convex_hull and hit_xy.shape[0] >= 3:
        try:
            from scipy.spatial import ConvexHull, Delaunay

            hull = ConvexHull(hit_xy)
            hull_xy = hit_xy[hull.vertices]
            tri = Delaunay(hull_xy)
            inside = tri.find_simplex(miss_xy) >= 0
            miss_xy = miss_xy[inside]
        except Exception:
            pass

    if miss_xy.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    from scipy.spatial import cKDTree

    kd = cKDTree(hit_xy)
    k = int(min(int(cfg.hole_fill_knn), hit_xy.shape[0]))
    d, idx = kd.query(miss_xy, k=k)
    if k == 1:
        d = d[:, None]
        idx = idx[:, None]

    weights = 1.0 / (np.power(d, float(cfg.hole_fill_power)) + 1e-8)
    z = (hit_z[idx] * weights).sum(axis=1) / weights.sum(axis=1)
    pts = np.column_stack([miss_xy[:, 0], miss_xy[:, 1], z]).astype(np.float64)

    if int(cfg.hole_fill_max_points) > 0 and pts.shape[0] > int(cfg.hole_fill_max_points):
        sel = np.random.choice(pts.shape[0], size=int(cfg.hole_fill_max_points), replace=False)
        pts = pts[sel]

    return pts


def _postprocess_mesh(
    mesh: o3d.geometry.TriangleMesh, cfg: Ply2SceneConfig, *, crop_aabb: o3d.geometry.AxisAlignedBoundingBox | None = None
) -> o3d.geometry.TriangleMesh:
    if crop_aabb is not None and cfg.crop_to_aabb:
        mesh = mesh.crop(crop_aabb)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    if cfg.keep_largest_component:
        # Keep the largest connected component if the API exists.
        try:
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            if cluster_n_triangles.size > 0:
                largest = int(cluster_n_triangles.argmax())
                remove_mask = (triangle_clusters != largest).tolist()
                try:
                    mesh.remove_triangles_by_mask(remove_mask)
                    mesh.remove_unreferenced_vertices()
                except Exception:
                    keep_triangles = np.where(triangle_clusters == largest)[0]
                    mesh = mesh.select_by_index(keep_triangles, cleanup=True)
        except Exception:
            pass

    try:
        mesh.orient_triangles()
    except Exception:
        pass
    mesh.compute_vertex_normals()
    return mesh


def _drop_mesh_vertices_below_z(
    mesh: o3d.geometry.TriangleMesh,
    z_min: float,
    cfg: Ply2SceneConfig,
) -> o3d.geometry.TriangleMesh:
    """(Deprecated) Hard filter: remove vertices (and any triangles touching them) with z < z_min."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        return mesh

    keep_v = vertices[:, 2] >= float(z_min)
    if bool(keep_v.all()):
        return mesh

    keep_t = keep_v[triangles].all(axis=1)
    triangles_kept = triangles[keep_t]
    if triangles_kept.size == 0:
        return o3d.geometry.TriangleMesh()

    used = np.unique(triangles_kept.reshape(-1))
    remap = -np.ones(vertices.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)

    new_vertices = vertices[used]
    new_triangles = remap[triangles_kept]

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(new_vertices)
    out.triangles = o3d.utility.Vector3iVector(new_triangles.astype(np.int32))
    return _postprocess_mesh(out, cfg)


def _orient_mesh_towards_point(
    mesh: o3d.geometry.TriangleMesh, center: np.ndarray, cfg: Ply2SceneConfig
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        return mesh

    center = center.astype(np.float64)
    target = np.array(
        [center[0], center[1], center[2] + float(cfg.hole_fill_top_z_margin_m)],
        dtype=np.float64,
    )

    triangles = np.asarray(mesh.triangles).astype(np.int64)
    vertices = np.asarray(mesh.vertices).astype(np.float64)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centers = (v0 + v1 + v2) / 3.0
    to_target = target[None, :] - centers
    flip = np.einsum("ij,ij->i", normals, to_target) < 0
    if np.any(flip):
        tmp = triangles[flip, 1].copy()
        triangles[flip, 1] = triangles[flip, 2]
        triangles[flip, 2] = tmp
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        mesh.compute_vertex_normals()
    return mesh


def _reconstruct_mesh(pcd: o3d.geometry.PointCloud, cfg: Ply2SceneConfig, *, depth: int) -> o3d.geometry.TriangleMesh:
    if cfg.use_ball_pivoting:
        radii = o3d.utility.DoubleVector([float(r) for r in cfg.bpa_radii])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        densities = None
    else:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))

    if densities is not None:
        densities = np.asarray(densities)
        threshold = float(np.quantile(densities, cfg.poisson_density_quantile))
        keep = densities >= threshold
        mesh = mesh.select_by_index(np.where(keep)[0])

    if cfg.crop_to_aabb:
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

    return _postprocess_mesh(mesh, cfg)


def _write_urdf(path: Path, scale: float) -> None:
    urdf = f"""<?xml version="1.0"?>
<robot name="scene">
  <link name="scene_link">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="meshes/scene_visual.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="meshes/scene_collision.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    path.write_text(urdf)


def _write_mjcf(path: Path, scale: float) -> None:
    xml = f"""<mujoco model="scene">
  <compiler meshdir="meshes"/>

  <asset>
    <mesh name="scene_visual" file="scene_visual.obj" scale="{scale} {scale} {scale}"/>
    <mesh name="scene_collision" file="scene_collision.obj" scale="{scale} {scale} {scale}"/>
  </asset>

  <worldbody>
    <body name="scene">
      <geom name="scene_collision" type="mesh" mesh="scene_collision" contype="1" conaffinity="1" rgba="0 0 0 0"/>
      <geom name="scene_visual" type="mesh" mesh="scene_visual" contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml)


def _write_includes(assets_path: Path, body_path: Path, scale: float) -> None:
    assets_xml = f"""<mujocoinclude>
  <mesh name="scene_visual" file="meshes/scene_visual.obj" scale="{scale} {scale} {scale}"/>
  <mesh name="scene_collision" file="meshes/scene_collision.obj" scale="{scale} {scale} {scale}"/>
</mujocoinclude>
"""
    body_xml = """<mujocoinclude>
  <body name="scene">
    <geom name="scene_collision" type="mesh" mesh="scene_collision" contype="1" conaffinity="1" rgba="0 0 0 0"/>
    <geom name="scene_visual" type="mesh" mesh="scene_visual" contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>
  </body>
</mujocoinclude>
"""
    assets_path.write_text(assets_xml)
    body_path.write_text(body_xml)


def main(cfg: Ply2SceneConfig) -> None:
    paths = get_sequence_paths(seq=cfg.seq, robot=cfg.robot, manual=cfg.manual)
    T_align = _load_transform_json(paths.ground_transform_json)

    output_dir = cfg.output_dir or (paths.run_dir / "artifacts" / "ply2scene" / "scene")
    meshes_dir = output_dir / "meshes"
    _ensure_dir(meshes_dir)

    scale = cfg.mesh_scale_factor
    if scale is None:
        if not paths.pipeline_config_json.exists():
            raise FileNotFoundError(
                "pipeline_config_json not found, so mesh scale_factor is unknown. "
                "Either run `omniretargeting.workflows.her.pipeline` first (so it writes scale_factor), "
                "or pass `--mesh-scale-factor <scale_factor>` to `ply2scene.convert`."
            )
        scale = _load_scale_factor(paths.pipeline_config_json)

    method = _select_mesh_method(cfg)
    hole_fill_points_count = 0

    points_count = 0

    if method == "tsdf":
        from omniretargeting.workflows.her.scene.ply2scene.tsdf import fuse_tsdf_mesh

        mesh_visual = fuse_tsdf_mesh(paths, T_align, cfg_like=cfg)

        # if cfg.collision_z_min_m is not None and len(mesh_visual.vertices) > 0 and len(mesh_visual.triangles) > 0:
        #     z_min = float(cfg.collision_z_min_m)
        #     before_v = int(len(mesh_visual.vertices))
        #     before_t = int(len(mesh_visual.triangles))

        #     bbox = mesh_visual.get_axis_aligned_bounding_box()
        #     min_bound = bbox.min_bound.copy()
        #     max_bound = bbox.max_bound.copy()
        #     min_bound[2] = max(float(min_bound[2]), z_min)
        #     mesh_visual = mesh_visual.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

        #     after_v = int(len(mesh_visual.vertices))
        #     after_t = int(len(mesh_visual.triangles))
        #     print(f"[ply2scene] collision_z_min_m={z_min:.4f}: mesh v {before_v}->{after_v}, t {before_t}->{after_t}")
        #     if after_t == 0:
        #         raise ValueError(f"collision_z_min_m={z_min:.4f} removed all triangles; lower the threshold.")

        # mesh_visual = _postprocess_mesh(mesh_visual, cfg)
        mesh_collision = mesh_visual
    else:
        points, colors, first_center = _points_from_rgbd(paths, T_align, cfg)
        points_count = int(points.shape[0])

        if cfg.collision_z_min_m is not None:
            z_min = float(cfg.collision_z_min_m)
            before_n = int(points.shape[0])
            keep = points[:, 2] >= z_min
            points = points[keep]
            if colors is not None:
                colors = colors[keep]
            after_n = int(points.shape[0])
            points_count = after_n
            print(f"[ply2scene] collision_z_min_m={z_min:.4f}: points {before_n}->{after_n}")
            if after_n == 0:
                raise ValueError(f"collision_z_min_m={z_min:.4f} removed all points; lower the threshold.")

        if method == "poisson":
            pcd = _prepare_point_cloud(points, colors, cfg)
            mesh_visual = _reconstruct_mesh(pcd, cfg, depth=int(cfg.poisson_depth_coarse))
            if first_center is not None:
                mesh_visual = _orient_mesh_towards_point(mesh_visual, first_center, cfg)
            mesh_collision = mesh_visual
        else:
            pts = points
            if (
                cfg.nksr_max_points is not None
                and int(cfg.nksr_max_points) > 0
                and pts.shape[0] > int(cfg.nksr_max_points)
            ):
                idx = np.random.choice(pts.shape[0], size=int(cfg.nksr_max_points), replace=False)
                pts = pts[idx]

            crop_aabb = None
            if cfg.crop_to_aabb:
                crop_aabb = o3d.geometry.AxisAlignedBoundingBox(
                    pts.min(axis=0).astype(np.float64),
                    pts.max(axis=0).astype(np.float64),
                )

            normals = _orient_normals_towards_top_view(pts, cfg)
            mesh_coarse, recon = _nksr_reconstruct(pts, normals, cfg)
            mesh_coarse = _postprocess_mesh(mesh_coarse, cfg, crop_aabb=crop_aabb)
            if first_center is not None:
                mesh_coarse = _orient_mesh_towards_point(mesh_coarse, first_center, cfg)

            mesh_final = mesh_coarse
            if cfg.nksr_two_round:
                fill_pts = _fill_holes_topdown(mesh_coarse, cfg)
                hole_fill_points_count = int(fill_pts.shape[0])
                if fill_pts.size > 0:
                    fill_normals = _orient_normals_towards_top_view(fill_pts, cfg)
                    combined_pts = np.concatenate([pts, fill_pts], axis=0)
                    combined_normals = np.concatenate([normals, fill_normals], axis=0)
                    mesh_final, _ = _nksr_reconstruct(combined_pts, combined_normals, cfg, reconstructor=recon)
                    mesh_final = _postprocess_mesh(mesh_final, cfg, crop_aabb=crop_aabb)
                    if first_center is not None:
                        mesh_final = _orient_mesh_towards_point(mesh_final, first_center, cfg)

            mesh_visual = mesh_final
            mesh_collision = mesh_visual

    visual_path = meshes_dir / "scene_visual.obj"
    collision_path = meshes_dir / "scene_collision.obj"
    # Write an additional collision mesh copy that the omniretargeting pipeline can treat as
    # a stable, internal artifact. This helps avoid accidental overwrites by other tooling
    # that might also write `scene_collision.obj` under a shared runs_root.
    collision_omni_path = meshes_dir / "scene_collision_omni.obj"

    o3d.io.write_triangle_mesh(str(visual_path), mesh_visual, write_triangle_uvs=False)
    o3d.io.write_triangle_mesh(str(collision_path), mesh_collision, write_triangle_uvs=False)
    o3d.io.write_triangle_mesh(str(collision_omni_path), mesh_collision, write_triangle_uvs=False)

    _write_urdf(output_dir / "scene.urdf", float(scale))
    _write_mjcf(output_dir / "scene.xml", float(scale))

    if cfg.write_includes:
        _write_includes(output_dir / "scene_assets.xml", output_dir / "scene_body.xml", float(scale))

    meta = {
        "seq": cfg.seq,
        "robot": cfg.robot,
        "output_dir": str(output_dir),
        "mesh_scale_factor": float(scale),
        "paths": {
            "predicted_dir": str(paths.predicted_dir),
            "depth_recovered": str(paths.depth_recovered),
            "results_dir": str(paths.results_dir),
            "ground_transform_json": str(paths.ground_transform_json),
            "pipeline_config_json": str(paths.pipeline_config_json),
        },
        "points_count": int(points_count),
        "mesh_visual_triangles": int(len(mesh_visual.triangles)),
        "mesh_collision_triangles": int(len(mesh_collision.triangles)),
        "collision_mesh_omni_path": str(collision_omni_path),
        "mesh_method": method,
        "hole_fill_points_count": hole_fill_points_count,
        "config": asdict(cfg),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"[ply2scene] Wrote scene to: {output_dir}")
    print(f"[ply2scene] Visual mesh: {visual_path}")
    print(f"[ply2scene] Collision mesh: {collision_path}")
    print(f"[ply2scene] URDF: {output_dir / 'scene.urdf'}")
    print(f"[ply2scene] MJCF: {output_dir / 'scene.xml'}")


if __name__ == "__main__":
    main(tyro.cli(Ply2SceneConfig))
