"""
Collision proxy generation for climbing_scene.

Motivation:
- MuJoCo distance queries against triangle-mesh geoms often clamp penetration to dist==0 and
  may not provide robust signed distances.
- Holosoma climbing uses primitive collision proxies (e.g., multi-boxes) so non-penetration
  constraints can be enforced reliably.

This module builds a voxel-box proxy from the reconstructed scene mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import tyro


@dataclass(frozen=True)
class CollisionProxyArgs:
    """Voxel-box collision proxy config."""

    voxel_pitch_m: float = 0.2
    """Voxel pitch in *scaled* world meters."""

    max_boxes: int = 512
    """Cap number of proxy boxes (randomly subsampled if exceeded)."""

    halfsize_margin_m: float = 0.0
    """Extra margin added to each box half-size (meters)."""

    hollow: bool = True
    """If True, keep only surface voxels (much fewer boxes than fill)."""

    roi_margin_m: float = 1.0
    """Expand ROI computed from the motion AABB by this margin (meters)."""

    z_min_m: float | None = None
    """Optional z-min filter for box centers (meters)."""

    z_max_m: float | None = None
    """Optional z-max filter for box centers (meters)."""

    seed: int = 0
    """Random seed for subsampling boxes."""


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        # Concatenate all geometries in the scene.
        geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No mesh geometries found in scene: {path}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)} for {path}")
    return mesh


def compute_roi_from_joints(
    joints_world: np.ndarray, *, margin_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute an axis-aligned ROI from a (T, J, 3) joint trajectory.
    Returns (roi_min, roi_max), both (3,).
    """
    if joints_world.ndim != 3 or joints_world.shape[-1] != 3:
        raise ValueError(f"Expected joints_world shape (T,J,3), got {joints_world.shape}")
    pts = joints_world.reshape(-1, 3)
    roi_min = np.min(pts, axis=0)
    roi_max = np.max(pts, axis=0)
    margin = float(margin_m)
    return roi_min - margin, roi_max + margin


def build_voxel_box_proxy(
    *,
    scene_mesh_path: Path,
    terrain_scale: float,
    roi_min: Optional[np.ndarray],
    roi_max: Optional[np.ndarray],
    cfg: CollisionProxyArgs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a voxel-box proxy from a scene mesh.

    Returns:
      centers: (N,3) box centers (scaled world)
      half_sizes: (N,3) MuJoCo box half-sizes (scaled world)
    """
    mesh = _load_trimesh(scene_mesh_path)
    mesh = mesh.copy()
    mesh.apply_scale(float(terrain_scale))

    # Optional coarse crop by triangle centroid to reduce voxelization cost.
    if roi_min is not None and roi_max is not None and mesh.faces is not None and len(mesh.faces) > 0:
        tri_centers = mesh.triangles_center
        keep = np.all((tri_centers >= roi_min[None, :]) & (tri_centers <= roi_max[None, :]), axis=1)
        if np.any(keep):
            idx = np.nonzero(keep)[0]
            mesh = mesh.submesh([idx], append=True, repair=False)

    pitch = float(cfg.voxel_pitch_m)
    if pitch <= 0:
        raise ValueError("voxel_pitch_m must be > 0")

    vox = mesh.voxelized(pitch=pitch)
    vox = vox.fill()
    if cfg.hollow:
        vox = vox.hollow()

    centers = np.asarray(vox.points, dtype=np.float64)

    if centers.size == 0:
        return centers.reshape(0, 3), centers.reshape(0, 3)

    # ROI filter (post-voxelization).
    if roi_min is not None and roi_max is not None:
        keep = np.all((centers >= roi_min[None, :]) & (centers <= roi_max[None, :]), axis=1)
        centers = centers[keep]

    if cfg.z_min_m is not None:
        centers = centers[centers[:, 2] >= float(cfg.z_min_m)]
    if cfg.z_max_m is not None:
        centers = centers[centers[:, 2] <= float(cfg.z_max_m)]

    if centers.shape[0] == 0:
        return centers.reshape(0, 3), centers.reshape(0, 3)

    max_boxes = int(cfg.max_boxes)
    if max_boxes > 0 and centers.shape[0] > max_boxes:
        rng = np.random.default_rng(int(cfg.seed))
        idx = rng.choice(centers.shape[0], size=max_boxes, replace=False)
        centers = centers[idx]

    half = pitch * 0.5 + float(cfg.halfsize_margin_m)
    half_sizes = np.full((centers.shape[0], 3), half, dtype=np.float64)
    return centers, half_sizes


def save_proxy_npz(path: Path, *, centers: np.ndarray, half_sizes: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), centers=centers.astype(np.float32), half_sizes=half_sizes.astype(np.float32), meta=meta)


@dataclass(frozen=True)
class BuildProxyCli:
    scene_mesh_path: Path
    out_npz: Path
    terrain_scale: float = 1.0
    roi_min: tuple[float, float, float] | None = None
    roi_max: tuple[float, float, float] | None = None
    cfg: CollisionProxyArgs = CollisionProxyArgs()


def main(args: BuildProxyCli) -> None:
    roi_min = np.asarray(args.roi_min, dtype=np.float64) if args.roi_min is not None else None
    roi_max = np.asarray(args.roi_max, dtype=np.float64) if args.roi_max is not None else None
    centers, half_sizes = build_voxel_box_proxy(
        scene_mesh_path=args.scene_mesh_path,
        terrain_scale=float(args.terrain_scale),
        roi_min=roi_min,
        roi_max=roi_max,
        cfg=args.cfg,
    )
    meta = {
        "scene_mesh_path": str(args.scene_mesh_path),
        "terrain_scale": float(args.terrain_scale),
        "voxel_pitch_m": float(args.cfg.voxel_pitch_m),
        "max_boxes": int(args.cfg.max_boxes),
        "halfsize_margin_m": float(args.cfg.halfsize_margin_m),
        "hollow": bool(args.cfg.hollow),
        "roi_min": roi_min.tolist() if roi_min is not None else None,
        "roi_max": roi_max.tolist() if roi_max is not None else None,
        "z_min_m": args.cfg.z_min_m,
        "z_max_m": args.cfg.z_max_m,
        "seed": int(args.cfg.seed),
        "num_boxes": int(centers.shape[0]),
    }
    save_proxy_npz(args.out_npz, centers=centers, half_sizes=half_sizes, meta=meta)
    print(f"[collision-proxy] wrote {args.out_npz} boxes={centers.shape[0]}")


if __name__ == "__main__":
    main(tyro.cli(BuildProxyCli))

