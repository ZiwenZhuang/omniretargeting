"""
HER workflow configuration (ported from holosoma_retargeting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SeqName = Literal["smooth", "wall_smooth", "rooftop_smooth"]
RobotType = Literal["g1", "t1"]


@dataclass(frozen=True)
class ManualPaths:
    """Edit these when deploying on a new machine."""

    results_data_dir: Path = Path("/home/juyiang/data/results")
    """Directory containing per-sequence result folders (e.g. smooth/wall_smooth/...)."""

    her_parkour_dir: Path = Path("/home/juyiang/data/Her_data/Parkour")
    """Directory containing Parkour scene assets (e.g. fused_scene.ply, predicted/)."""

    smpl_model_path: Path = Path("/home/juyiang/data/SMPL_models/models")
    """SMPL/SMPL-X model directory for `smplx.create()`."""

    runs_root: Path = Path("/home/juyiang/data/holosoma_runs")
    """Workspace for pipeline outputs/artifacts (stable across sessions)."""


@dataclass(frozen=True)
class GeoCalibAlignmentArgs:
    """Config for automatic gravity+ground alignment."""

    use_rgbd_scene: bool = True
    num_geocalib_frames: int = 12
    geocalib_frame_stride: int = 5
    geocalib_frame_indices: str | None = None
    geocalib_weights: str = "pinhole"
    geocalib_camera_y_up: bool = False
    geocalib_device: str | None = None
    geocalib_angle_outlier_deg: float = 15.0

    max_points: int = 200_000
    voxel_size: float = 0.05
    plane_distance_threshold: float = 0.03
    plane_ransac_n: int = 3
    plane_num_iterations: int = 2000
    plane_angle_deg: float = 10.0
    plane_max_candidates: int = 10
    recenter_xy: bool = True

    vis: bool = False
    debug: bool = False


@dataclass(frozen=True)
class PipelineArgs:
    seq: SeqName
    robot: RobotType = "g1"
    human_height_m: float = 1.7
    stage: Literal["prepare", "full"] = "full"
    """Pipeline stage: 'prepare' writes transform+scale artifacts without requiring a scene mesh; 'full' runs retarget."""
    retarget_mode: Literal["robot_only", "climbing_scene"] = "climbing_scene"
    """Select retarget pipeline mode."""
    build_scene_if_missing: bool = True
    """If True and retarget_mode='climbing_scene', automatically run ply2scene when scene mesh is missing."""
    robot_urdf_file: Path | None = None
    """Optional override for robot URDF (useful for climbing)."""
    debug_frames: int = 1
    """Print solver debug output for the first N frames (0 disables)."""
    enable_penetration_constraints: bool = True
    """If True, enforce non-penetration constraints against the terrain mesh in the solver."""
    collision_detection_threshold: float = 0.1
    """Distance threshold (meters) for MuJoCo collision candidate pairs."""
    penetration_tolerance: float = 1e-3
    """Allowed penetration slack (meters)."""
    max_penetration_constraints: int = 0
    """Cap number of penetration constraints per SQP step (0 disables cap)."""
    penetration_constraint_mode: Literal["soft", "hard"] = "hard"
    """Penetration constraint mode. 'soft' uses slack (more robust); 'hard' enforces strict feasibility."""
    penetration_slack_weight: float = 1e4
    """Slack penalty weight used when penetration_constraint_mode='soft'."""

    # Collision proxy for climbing_scene
    collision_proxy_mode: Literal["none", "voxel_boxes"] = "voxel_boxes"
    """Collision proxy type for climbing_scene. Recommended: voxel_boxes."""
    collision_proxy_voxel_pitch_m: float = 0.12
    """Voxel pitch for proxy boxes (meters in *robot-scaled* world)."""
    collision_proxy_max_boxes: int = 3000
    """Max proxy boxes (subsample if exceeded)."""
    collision_proxy_halfsize_margin_m: float = 0.02
    """Extra margin added to each box half-size (meters)."""
    collision_proxy_hollow: bool = True
    """If True, use only surface voxels (fewer boxes)."""
    collision_proxy_roi_margin_m: float = 1.0
    """Expand motion ROI by this margin (meters) when building proxy."""

    # Ground alignment behavior
    ground_alignment_mode: Literal["cached", "manual", "geocalib"] = "geocalib"
    """How to obtain the scene alignment transform in step (1)."""

    geocalib_alignment: GeoCalibAlignmentArgs = field(default_factory=GeoCalibAlignmentArgs)
    """Config for ground_alignment_mode="geocalib"."""

    # Manual path overrides (deploy-time)
    manual: ManualPaths = ManualPaths()
