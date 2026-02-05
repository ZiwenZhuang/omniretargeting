"""
Path derivation helpers for HER workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ManualPaths, RobotType, SeqName

SEQUENCE_NAME_MAPPING: dict[SeqName, str] = {
    "smooth": "parkour_somersault",
    "wall_smooth": "parkour_wall",
    "rooftop_smooth": "parkour_rooftop",
}


ROBOT_HEIGHTS_M: dict[RobotType, float] = {
    "g1": 1.32,
    "t1": 1.20,
}


@dataclass(frozen=True)
class SequencePaths:
    """All derived paths for a given run (seq + robot)."""

    # Inputs
    results_dir: Path
    all_results_video: Path
    depth_recovered: Path
    scene_dir: Path
    predicted_dir: Path
    fused_scene_ply: Path
    aligned_scene_ply: Path

    # Workspace (outputs)
    run_dir: Path
    artifacts_dir: Path
    ground_alignment_dir: Path
    ground_transform_json: Path

    prepared_data_dir: Path
    prepared_pt_path: Path
    prepared_smplx22_path: Path

    retarget_save_dir: Path
    pipeline_config_json: Path


def get_scale_factor(robot_type: RobotType, human_height_m: float) -> float:
    """Compute robot/human scale factor."""
    if robot_type not in ROBOT_HEIGHTS_M:
        raise ValueError(f"Unknown robot_type: {robot_type}")
    return ROBOT_HEIGHTS_M[robot_type] / float(human_height_m)


def get_sequence_paths(*, seq: SeqName, robot: RobotType, manual: ManualPaths) -> SequencePaths:
    results_dir = manual.results_data_dir / seq
    scene_name = SEQUENCE_NAME_MAPPING[seq]
    scene_dir = manual.her_parkour_dir / scene_name

    all_results_video = results_dir / "all_results_video.pt"
    depth_recovered = results_dir / "depth_recovered"
    predicted_dir = scene_dir / "predicted"
    fused_scene_ply = scene_dir / "fused_scene.ply"
    aligned_scene_ply = scene_dir / f"aligned_scene_manual_{seq}.ply"

    run_dir = manual.runs_root / seq / robot
    artifacts_dir = run_dir / "artifacts"
    ground_alignment_dir = artifacts_dir / "ground_alignment"
    ground_transform_json = ground_alignment_dir / "transform.json"

    prepared_data_dir = run_dir / "inputs" / "retarget_data"
    prepared_pt_path = prepared_data_dir / f"{seq}.pt"
    prepared_smplx22_path = prepared_data_dir / f"{seq}.npz"

    retarget_save_dir = run_dir / "outputs" / "retarget"
    pipeline_config_json = run_dir / "pipeline_config.json"

    return SequencePaths(
        results_dir=results_dir,
        all_results_video=all_results_video,
        depth_recovered=depth_recovered,
        scene_dir=scene_dir,
        predicted_dir=predicted_dir,
        fused_scene_ply=fused_scene_ply,
        aligned_scene_ply=aligned_scene_ply,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        ground_alignment_dir=ground_alignment_dir,
        ground_transform_json=ground_transform_json,
        prepared_data_dir=prepared_data_dir,
        prepared_pt_path=prepared_pt_path,
        prepared_smplx22_path=prepared_smplx22_path,
        retarget_save_dir=retarget_save_dir,
        pipeline_config_json=pipeline_config_json,
    )


def ensure_run_dirs(paths: SequencePaths) -> None:
    for directory in (
        paths.run_dir,
        paths.artifacts_dir,
        paths.ground_alignment_dir,
        paths.prepared_data_dir,
        paths.retarget_save_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

