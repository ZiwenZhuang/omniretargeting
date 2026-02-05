"""
Unified HER pipeline runner for omniretargeting.

Stages:
1) Ground alignment (manual/cached/geocalib)
2) Prepare SMPL-X22 trajectory + contact artifacts from HER outputs
3) Run omniretargeting (robot_only or climbing_scene)
4) Save pipeline summary
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import trimesh
import tyro

from omniretargeting.workflows.her.config import PipelineArgs
from omniretargeting.workflows.her.ground_alignment import manual_viser
from omniretargeting.workflows.her.ground_alignment.geocalib import (
    GeoCalibGroundAlignmentConfig,
    compute_geocalib_ground_alignment,
)
from omniretargeting.workflows.her.io.contact import load_contact_flags, save_contact_logits_from_results
from omniretargeting.workflows.her.io.her_results import (
    compute_smplx_joints,
    convert_results_to_omni_smplx22,
    load_all_results_video,
    save_omni_smplx22_npz,
)
from omniretargeting.workflows.her.io.joints_map import OMNI_SMPLX22_JOINTS
from omniretargeting.workflows.her.io.robot_mapping import (
    DEFAULT_ROBOT_URDF_PATHS,
    get_default_joint_mapping,
)
from omniretargeting.workflows.her.motion.height_correction import apply_contact_height_correction
from omniretargeting.workflows.her.paths import (
    ROBOT_HEIGHTS_M,
    ensure_run_dirs,
    get_scale_factor,
    get_sequence_paths,
)
from omniretargeting.workflows.her.scene.collision_proxy import (
    CollisionProxyArgs,
    build_voxel_box_proxy,
    compute_roi_from_joints,
    save_proxy_npz,
)


def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r") as file:
        data = json.load(file)
    transform = np.asarray(data["transform_matrix"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform_matrix in {path} must be 4x4, got {transform.shape}")
    return transform


def _save_transform_json(path: Path, transform: np.ndarray, *, seq: str, robot: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transform_matrix": np.asarray(transform, dtype=np.float64).round(8).tolist(),
        "timestamp": datetime.now().isoformat(),
        "seq": seq,
        "robot": robot,
    }
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _write_pipeline_config(
    *,
    path: Path,
    args: PipelineArgs,
    paths,
    scale_factor: float,
    stage: str,
    robot_urdf_path: Path | None = None,
    terrain_mesh_path: Path | None = None,
    terrain_scale: float | None = None,
    collision_proxy_npz: Path | None = None,
) -> None:
    payload = {
        "seq": args.seq,
        "robot": args.robot,
        "human_height_m": args.human_height_m,
        "scale_factor": float(scale_factor),
        "retarget_mode": str(args.retarget_mode),
        "stage": stage,
        "debug_frames": int(getattr(args, "debug_frames", 0)),
        "enable_penetration_constraints": bool(getattr(args, "enable_penetration_constraints", True)),
        "collision_detection_threshold": float(getattr(args, "collision_detection_threshold", 0.1)),
        "penetration_tolerance": float(getattr(args, "penetration_tolerance", 1e-3)),
        "max_penetration_constraints": int(getattr(args, "max_penetration_constraints", 64)),
        "penetration_constraint_mode": str(getattr(args, "penetration_constraint_mode", "soft")),
        "penetration_slack_weight": float(getattr(args, "penetration_slack_weight", 1e4)),
        "collision_proxy_mode": str(getattr(args, "collision_proxy_mode", "none")),
        "collision_proxy_npz": str(collision_proxy_npz) if collision_proxy_npz is not None else None,
        "ground_alignment_mode": str(args.ground_alignment_mode),
        "transform_path": str(paths.ground_transform_json),
        "prepared_smplx22_npz": str(paths.prepared_smplx22_path),
        "prepared_contact_npz": str(paths.prepared_data_dir / f"{args.seq}_contact.npz"),
        "retarget_save_dir": str(paths.retarget_save_dir),
        "robot_urdf_path": str(robot_urdf_path) if robot_urdf_path is not None else None,
        "robot_mujoco_xml_path": str(robot_urdf_path.with_suffix(".xml"))
        if robot_urdf_path is not None and robot_urdf_path.with_suffix(".xml").exists()
        else None,
        "terrain_mesh_path": str(terrain_mesh_path) if terrain_mesh_path is not None else None,
        "terrain_scale": float(terrain_scale) if terrain_scale is not None else None,
        "vertical_bias_z_min_human_m": 0.0,
        "vertical_bias_z_min_robot_m": 0.0,
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "timestamp": datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _ensure_flat_terrain(path: Path, *, size: float = 10.0, thickness: float = 0.1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    mesh = trimesh.creation.box(extents=[size, size, thickness])
    mesh.apply_translation([0.0, 0.0, -thickness * 0.5])
    mesh.export(str(path))
    return path


def _resolve_ground_alignment(args: PipelineArgs, paths) -> np.ndarray:
    mode = str(args.ground_alignment_mode)
    if mode == "manual":
        config = {
            "smpl_model_path": str(args.manual.smpl_model_path),
            "human_mesh_path": str(paths.all_results_video),
            "scale_factors_path": str(paths.depth_recovered),
            "scene_ply_path": str(paths.fused_scene_ply),
            "default_save_path": str(paths.aligned_scene_ply),
            "data_dir": str(paths.predicted_dir),
            "depth_dir": str(paths.depth_recovered),
            "smplx_results_dir": str(paths.results_dir),
            "use_rgbd_scene": True,
            "transform_output_path": str(paths.ground_transform_json),
            "exit_process_on_save": False,
        }
        transform = manual_viser.main(config)
        _save_transform_json(paths.ground_transform_json, transform, seq=args.seq, robot=args.robot)
        return transform

    if mode == "cached":
        if not paths.ground_transform_json.exists():
            raise FileNotFoundError(
                f"ground_alignment_mode='cached' but transform not found: {paths.ground_transform_json}"
            )
        return _load_transform_json(paths.ground_transform_json)

    if mode == "geocalib":
        config = GeoCalibGroundAlignmentConfig(
            predicted_dir=paths.predicted_dir,
            depth_dir=paths.depth_recovered,
            results_dir=paths.results_dir,
            fused_scene_ply=paths.fused_scene_ply,
            use_rgbd_scene=bool(args.geocalib_alignment.use_rgbd_scene),
            num_geocalib_frames=int(args.geocalib_alignment.num_geocalib_frames),
            geocalib_frame_stride=int(args.geocalib_alignment.geocalib_frame_stride),
            geocalib_frame_indices=args.geocalib_alignment.geocalib_frame_indices,
            geocalib_weights=str(args.geocalib_alignment.geocalib_weights),
            geocalib_camera_y_up=bool(args.geocalib_alignment.geocalib_camera_y_up),
            geocalib_device=args.geocalib_alignment.geocalib_device,
            geocalib_angle_outlier_deg=float(args.geocalib_alignment.geocalib_angle_outlier_deg),
            max_points=int(args.geocalib_alignment.max_points),
            voxel_size=float(args.geocalib_alignment.voxel_size),
            plane_distance_threshold=float(args.geocalib_alignment.plane_distance_threshold),
            plane_ransac_n=int(args.geocalib_alignment.plane_ransac_n),
            plane_num_iterations=int(args.geocalib_alignment.plane_num_iterations),
            plane_angle_deg=float(args.geocalib_alignment.plane_angle_deg),
            plane_max_candidates=int(args.geocalib_alignment.plane_max_candidates),
            recenter_xy=bool(args.geocalib_alignment.recenter_xy),
        )
        result = compute_geocalib_ground_alignment(
            config,
            return_points=False,
            debug=bool(args.geocalib_alignment.debug),
        )
        transform = result.T_align
        _save_transform_json(paths.ground_transform_json, transform, seq=args.seq, robot=args.robot)
        return transform

    raise ValueError(f"Unknown ground_alignment_mode: {mode}")


def _resolve_robot_urdf(args: PipelineArgs) -> Path:
    if args.robot_urdf_file is not None:
        return args.robot_urdf_file
    if args.robot in DEFAULT_ROBOT_URDF_PATHS:
        return DEFAULT_ROBOT_URDF_PATHS[args.robot]
    raise ValueError(f"Cannot resolve default URDF path for robot={args.robot}.")


def main(args: PipelineArgs) -> None:
    paths = get_sequence_paths(seq=args.seq, robot=args.robot, manual=args.manual)
    ensure_run_dirs(paths)
    scale_factor = get_scale_factor(args.robot, args.human_height_m)

    transform = _resolve_ground_alignment(args, paths)

    # 2) Prepare joints/contact artifacts
    video_results = load_all_results_video(paths.all_results_video)
    video_results = compute_smplx_joints(
        video_results,
        smpl_model_path=str(args.manual.smpl_model_path),
        device="cpu",
        gender="male",
        use_face_contour=True,
    )

    try:
        joints_22 = convert_results_to_omni_smplx22(
            video_results,
            scale_factors_path=paths.depth_recovered,
            scale_mode="average",
            constant_scale_factor=1.0,
            transform_matrix=transform,
        )
    except ValueError as exc:
        raise ValueError(
            "Failed to map HER joints to omniretargeting 22-joint standard "
            "(Pelvis/L_Hip/.../R_Wrist). "
            "Current pipeline expects full SMPL-X joint arrays from HER results. "
            "If your source already stores 22-joint arrays, we need to explicitly enable "
            "that path and verify the ordering before retargeting."
        ) from exc

    contact_npz_path = paths.prepared_data_dir / f"{args.seq}_contact.npz"
    save_contact_logits_from_results(video_results, contact_npz_path)
    contact_flags = load_contact_flags(contact_npz_path, num_frames=joints_22.shape[0])

    # Keep global z_min disabled; use contact-driven per-frame vertical correction.
    joints_22_corrected, z_bias_per_frame = apply_contact_height_correction(
        joints_22,
        contact_flags=contact_flags,
        left_foot_idx=10,
        right_foot_idx=11,
    )

    save_omni_smplx22_npz(joints_22_corrected, paths.prepared_smplx22_path)
    # Optional tensor dump for debugging/inspection.
    torch.save(torch.from_numpy(joints_22_corrected.astype(np.float32)), str(paths.prepared_pt_path))

    if str(getattr(args, "stage", "full")) == "prepare":
        _write_pipeline_config(
            path=paths.pipeline_config_json,
            args=args,
            paths=paths,
            scale_factor=scale_factor,
            stage="prepare",
        )
        print(f"[her-pipeline] Prepared artifacts for {args.seq}/{args.robot}")
        print(f"[her-pipeline] Transform: {paths.ground_transform_json}")
        print(f"[her-pipeline] Prepared joints: {paths.prepared_smplx22_path}")
        print(f"[her-pipeline] Contact: {contact_npz_path}")
        return

    # 3) Retarget with omniretargeting
    robot_urdf_path = _resolve_robot_urdf(args)
    if not robot_urdf_path.exists():
        raise FileNotFoundError(
            f"Robot URDF not found: {robot_urdf_path}. Please pass --robot-urdf-file explicitly."
        )
    robot_mujoco_xml_path = robot_urdf_path.with_suffix(".xml")
    if not robot_mujoco_xml_path.exists():
        robot_mujoco_xml_path = None

    if args.retarget_mode == "robot_only":
        terrain_mesh_path = _ensure_flat_terrain(paths.artifacts_dir / "terrain" / "flat_ground.obj")
    elif args.retarget_mode == "climbing_scene":
        scene_mesh_dir = paths.run_dir / "artifacts" / "ply2scene" / "scene" / "meshes"
        scene_mesh_path = scene_mesh_dir / "scene_collision_omni.obj"
        scene_mesh_fallback = scene_mesh_dir / "scene_collision.obj"

        if (not scene_mesh_path.exists()) and (not scene_mesh_fallback.exists()) and bool(
            getattr(args, "build_scene_if_missing", False)
        ):
            from omniretargeting.workflows.her.scene.ply2scene.convert import Ply2SceneConfig, main as ply2scene_main

            print(f"[her-pipeline] Scene mesh missing; running ply2scene: {scene_mesh_fallback}")
            ply2scene_main(
                Ply2SceneConfig(
                    seq=args.seq,
                    robot=args.robot,
                    manual=args.manual,
                    mesh_scale_factor=scale_factor,
                )
            )
        if not scene_mesh_path.exists() and scene_mesh_fallback.exists():
            scene_mesh_path = scene_mesh_fallback
        if not scene_mesh_path.exists():
            raise FileNotFoundError(
                f"Scene mesh not found: {scene_mesh_path}. "
                "Run `python -m omniretargeting.workflows.her.scene.ply2scene.convert --seq ... --robot ...` first "
                "or pass `--build-scene-if-missing true` to the pipeline."
            )
        terrain_mesh_path = scene_mesh_path
    else:
        raise ValueError(f"Unknown retarget_mode: {args.retarget_mode}")

    joint_mapping = get_default_joint_mapping(args.robot)
    from omniretargeting import OmniRetargeter

    collision_proxy_boxes = None
    collision_proxy_npz = None
    if (
        args.retarget_mode == "climbing_scene"
        and bool(getattr(args, "enable_penetration_constraints", True))
        and str(getattr(args, "collision_proxy_mode", "none")) == "voxel_boxes"
    ):
        proxy_dir = paths.artifacts_dir / "collision_proxy"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        collision_proxy_npz = proxy_dir / f"scene_proxy_boxes_{args.seq}_{args.robot}.npz"

        centers = None
        half_sizes = None
        if collision_proxy_npz.exists():
            data = np.load(collision_proxy_npz, allow_pickle=True)
            try:
                cached_centers = np.asarray(data["centers"], dtype=np.float64)
                cached_half_sizes = np.asarray(data["half_sizes"], dtype=np.float64)
            except Exception:
                cached_centers = None
                cached_half_sizes = None

            meta = None
            if "meta" in data.files:
                try:
                    meta = data["meta"].item()
                except Exception:
                    meta = None

            expected_cfg = {
                "voxel_pitch_m": float(getattr(args, "collision_proxy_voxel_pitch_m", 0.2)),
                "max_boxes": int(getattr(args, "collision_proxy_max_boxes", 512)),
                "halfsize_margin_m": float(getattr(args, "collision_proxy_halfsize_margin_m", 0.0)),
                "hollow": bool(getattr(args, "collision_proxy_hollow", True)),
                "roi_margin_m": float(getattr(args, "collision_proxy_roi_margin_m", 1.0)),
            }

            cache_ok = cached_centers is not None and cached_half_sizes is not None
            if meta is not None:
                try:
                    if str(meta.get("scene_mesh_path")) != str(scene_mesh_path):
                        cache_ok = False
                    if abs(float(meta.get("terrain_scale", 0.0)) - float(scale_factor)) > 1e-6:
                        cache_ok = False
                    meta_cfg = meta.get("cfg", {}) if isinstance(meta.get("cfg", {}), dict) else {}
                    for k, v in expected_cfg.items():
                        if k not in meta_cfg:
                            cache_ok = False
                            break
                        if isinstance(v, float):
                            if abs(float(meta_cfg[k]) - float(v)) > 1e-9:
                                cache_ok = False
                                break
                        else:
                            if meta_cfg[k] != v:
                                cache_ok = False
                                break
                except Exception:
                    cache_ok = False

            # If the scene mesh was regenerated after the proxy, rebuild.
            try:
                if scene_mesh_path.exists() and collision_proxy_npz.exists():
                    if scene_mesh_path.stat().st_mtime > collision_proxy_npz.stat().st_mtime + 1e-6:
                        cache_ok = False
            except Exception:
                pass

            if cache_ok:
                centers = cached_centers
                half_sizes = cached_half_sizes
            else:
                print(f"[her-pipeline] Collision proxy cache stale; rebuilding: {collision_proxy_npz}")

        if centers is None or half_sizes is None:
            # Build ROI from scaled joints, to match solver's world.
            joints_scaled = joints_22_corrected * float(scale_factor)
            roi_min, roi_max = compute_roi_from_joints(joints_scaled, margin_m=0.0)
            cfg = CollisionProxyArgs(
                voxel_pitch_m=float(getattr(args, "collision_proxy_voxel_pitch_m", 0.2)),
                max_boxes=int(getattr(args, "collision_proxy_max_boxes", 512)),
                halfsize_margin_m=float(getattr(args, "collision_proxy_halfsize_margin_m", 0.0)),
                hollow=bool(getattr(args, "collision_proxy_hollow", True)),
                roi_margin_m=float(getattr(args, "collision_proxy_roi_margin_m", 1.0)),
            )
            # Expand ROI by cfg.roi_margin_m.
            roi_min = roi_min - float(cfg.roi_margin_m)
            roi_max = roi_max + float(cfg.roi_margin_m)
            centers, half_sizes = build_voxel_box_proxy(
                scene_mesh_path=scene_mesh_path,
                terrain_scale=float(scale_factor),
                roi_min=roi_min,
                roi_max=roi_max,
                cfg=cfg,
            )
            meta = {
                "scene_mesh_path": str(scene_mesh_path),
                "terrain_scale": float(scale_factor),
                "roi_min": roi_min.tolist(),
                "roi_max": roi_max.tolist(),
                "cfg": {
                    "voxel_pitch_m": float(cfg.voxel_pitch_m),
                    "max_boxes": int(cfg.max_boxes),
                    "halfsize_margin_m": float(cfg.halfsize_margin_m),
                    "hollow": bool(cfg.hollow),
                    "roi_margin_m": float(cfg.roi_margin_m),
                },
                "num_boxes": int(centers.shape[0]),
            }
            save_proxy_npz(collision_proxy_npz, centers=centers, half_sizes=half_sizes, meta=meta)
            print(f"[her-pipeline] Collision proxy built: {collision_proxy_npz} boxes={centers.shape[0]}")

        collision_proxy_boxes = (centers, half_sizes)

    retargeter = OmniRetargeter(
        robot_urdf_path=robot_urdf_path,
        robot_mujoco_xml_path=robot_mujoco_xml_path,
        terrain_mesh_path=terrain_mesh_path,
        joint_mapping=joint_mapping,
        robot_height=ROBOT_HEIGHTS_M[args.robot],
        smplx_joint_names=list(OMNI_SMPLX22_JOINTS),
        debug_frames=int(getattr(args, "debug_frames", 0)),
        terrain_scale_override=float(scale_factor),
        collision_proxy_boxes=collision_proxy_boxes,
        enable_penetration_constraints=bool(getattr(args, "enable_penetration_constraints", True)),
        collision_detection_threshold=float(getattr(args, "collision_detection_threshold", 0.1)),
        penetration_tolerance=float(getattr(args, "penetration_tolerance", 1e-3)),
        max_penetration_constraints=int(getattr(args, "max_penetration_constraints", 64)),
        penetration_constraint_mode=str(getattr(args, "penetration_constraint_mode", "soft")),
        penetration_slack_weight=float(getattr(args, "penetration_slack_weight", 1e4)),
    )

    terrain_scale, retargeted_motion = retargeter.retarget_motion(
        joints_22_corrected,
        visualize_trajectory=False,
    )

    out_npz = paths.retarget_save_dir / f"{args.seq}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_npz),
        qpos=retargeted_motion.astype(np.float32),
        fps=np.asarray(30, dtype=np.int32),
        terrain_scale=np.asarray(terrain_scale, dtype=np.float32),
        z_bias_per_frame=np.asarray(z_bias_per_frame if z_bias_per_frame is not None else np.zeros((0,)), dtype=np.float32),
    )

    _write_pipeline_config(
        path=paths.pipeline_config_json,
        args=args,
        paths=paths,
        scale_factor=scale_factor,
        stage="full",
        robot_urdf_path=robot_urdf_path,
        terrain_mesh_path=terrain_mesh_path,
        terrain_scale=terrain_scale,
        collision_proxy_npz=collision_proxy_npz,
    )
    print(f"[her-pipeline] Retarget done: {out_npz}")


if __name__ == "__main__":
    main(tyro.cli(PipelineArgs))
