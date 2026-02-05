"""
HER result loading and conversion helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

from .joints_map import remap_smplx_to_omni_smplx22

ScaleMode = Literal["constant", "per_frame", "average"]


def load_all_results_video(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"HER results file not found: {path}")
    results = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(results, list):
        raise ValueError(f"Expected list in {path}, got {type(results)}")
    return results


def opencv_to_z_up(xyz: np.ndarray) -> np.ndarray:
    """OpenCV (x right, y down, z forward) -> Z-up (x forward, y left, z up)."""
    out = xyz.copy()
    out[..., 0] = xyz[..., 0]
    out[..., 1] = xyz[..., 2]
    out[..., 2] = -xyz[..., 1]
    return out


def apply_transform_matrix(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points (row-major)."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.concatenate([points, ones], axis=1)
    return (points_h @ transform.T)[:, :3]


def transform_cam_to_world(
    points_cam: np.ndarray,
    camera_pose_c2w: np.ndarray,
    *,
    scale_translation: float = 1.0,
) -> np.ndarray:
    """
    Transform points from camera frame to world frame.

    Note: follows legacy behavior where only camera translation is scaled.
    """
    rotation = camera_pose_c2w[:3, :3]
    translation = camera_pose_c2w[:3, 3] * float(scale_translation)
    return (rotation @ points_cam.T).T + translation


def _scale_factor_path(scale_factors_path: Path, frame_idx: int) -> Path:
    return scale_factors_path / f"{frame_idx:04d}_scale_factor.txt"


def load_scale_factor_for_frame(scale_factors_path: str | Path, frame_idx: int) -> float:
    file_path = _scale_factor_path(Path(scale_factors_path), frame_idx)
    if not file_path.exists():
        raise FileNotFoundError(f"Scale factor file not found for frame {frame_idx}: {file_path}")
    with open(file_path, "r") as file:
        return float(file.read().strip())


def load_scale_factor_average(scale_factors_path: str | Path, *, max_frames: int = 10000) -> float:
    scale_dir = Path(scale_factors_path)
    values: list[float] = []
    for frame_idx in range(max_frames):
        file_path = _scale_factor_path(scale_dir, frame_idx)
        if not file_path.exists():
            continue
        try:
            with open(file_path, "r") as file:
                values.append(float(file.read().strip()))
        except Exception:
            continue
    if not values:
        raise FileNotFoundError(f"No valid scale factor files found under: {scale_dir}")
    return float(np.mean(values))


def compute_smplx_joints(
    video_results: list[dict],
    *,
    smpl_model_path: str,
    device: str = "cpu",
    gender: str = "male",
    use_face_contour: bool = True,
) -> list[dict]:
    """
    Fill missing frame["joints"] by running SMPL-X FK from pose/betas/transl.
    Frames that already contain joints are kept as-is.
    """
    missing_any = any(isinstance(frame, dict) and frame.get("joints") is None for frame in video_results)
    if not missing_any:
        return video_results

    import smplx
    from tqdm import tqdm

    smplx_model = smplx.create(
        model_path=str(smpl_model_path),
        model_type="smplx",
        gender=gender,
        use_face_contour=use_face_contour,
    ).to(device)

    output_results: list[dict] = []
    for frame_data in tqdm(video_results, desc="SMPL-X joints"):
        if not isinstance(frame_data, dict):
            output_results.append(frame_data)
            continue
        if frame_data.get("joints") is not None:
            output_results.append(frame_data)
            continue

        pose = frame_data.get("pose")
        betas = frame_data.get("betas")
        transl = frame_data.get("transl")
        if pose is None or betas is None or transl is None:
            raise KeyError("Missing pose/betas/transl when trying to reconstruct SMPL-X joints.")

        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device).unsqueeze(0)
        betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)
        transl_tensor = torch.tensor(transl, dtype=torch.float32, device=device).unsqueeze(0)

        global_orient = pose_tensor[:, :3]
        body_pose = pose_tensor[:, 3:66]
        jaw_pose = pose_tensor[:, 66:69]
        leye_pose = pose_tensor[:, 69:72]
        reye_pose = pose_tensor[:, 72:75]

        with torch.no_grad():
            smplx_out = smplx_model(
                global_orient=global_orient,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                betas=betas_tensor,
                transl=transl_tensor,
                return_verts=False,
            )
            joints = smplx_out.joints[0].detach().cpu().numpy()

        new_frame = dict(frame_data)
        new_frame["joints"] = joints
        output_results.append(new_frame)
    return output_results


def convert_results_to_omni_smplx22(
    video_results: list[dict],
    *,
    scale_factors_path: Optional[str | Path],
    scale_mode: ScaleMode = "average",
    constant_scale_factor: float = 1.0,
    transform_matrix: Optional[np.ndarray] = None,
    assume_input_is_omni_smplx22: bool = False,
) -> np.ndarray:
    """
    Convert HER per-frame results to omniretargeting canonical 22-joint world trajectory.

    Returns:
        (T, 22, 3) joints in Z-up world coordinates.
    """
    if transform_matrix is None:
        transform = np.eye(4, dtype=np.float64)
    else:
        transform = np.asarray(transform_matrix, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"transform_matrix must be 4x4, got {transform.shape}")

    if scale_mode in {"per_frame", "average"} and scale_factors_path is None:
        raise ValueError("scale_factors_path is required for scale_mode='per_frame' or 'average'")

    avg_scale = load_scale_factor_average(scale_factors_path) if scale_mode == "average" else None

    joints_out: list[np.ndarray] = []
    for frame_idx, frame in enumerate(video_results):
        if not isinstance(frame, dict):
            continue
        if frame.get("camera_pose") is None:
            continue
        if frame.get("joints") is None:
            continue

        camera_pose = np.asarray(frame["camera_pose"], dtype=np.float64)
        joints_cam = np.asarray(frame["joints"], dtype=np.float64)

        if scale_mode == "constant":
            scale = float(constant_scale_factor)
        elif scale_mode == "average":
            scale = float(avg_scale)
        else:
            scale = float(load_scale_factor_for_frame(scale_factors_path, frame_idx))

        joints_world = transform_cam_to_world(joints_cam, camera_pose, scale_translation=scale)
        joints_world = opencv_to_z_up(joints_world)
        joints_world = apply_transform_matrix(joints_world, transform)
        joints_22 = remap_smplx_to_omni_smplx22(
            joints_world[None, ...],
            assume_input_is_omni_smplx22=assume_input_is_omni_smplx22,
        )[0]
        joints_out.append(joints_22.astype(np.float32))

    if not joints_out:
        raise ValueError("No valid frames found when converting HER results to omni SMPLX22 trajectory.")
    return np.stack(joints_out, axis=0)


def save_omni_smplx22_npz(joints_22: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), global_joint_positions=joints_22)

