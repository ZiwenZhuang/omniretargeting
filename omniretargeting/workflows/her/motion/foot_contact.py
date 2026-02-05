"""
Foot contact sequence builders for retarget constraints.
"""

from __future__ import annotations

import numpy as np

from .height_correction import split_left_right_contact


def contact_flags_to_sequences(
    contact_flags: np.ndarray,
    *,
    toe_names: tuple[str, str] = ("L_Foot", "R_Foot"),
    num_frames: int | None = None,
) -> list[dict[str, bool]]:
    left, right = split_left_right_contact(contact_flags)
    total_frames = int(num_frames if num_frames is not None else left.shape[0])
    if left.shape[0] < total_frames:
        left = np.concatenate([left, np.full(total_frames - left.shape[0], bool(left[-1]) if left.size else False)])
    else:
        left = left[:total_frames]
    if right.shape[0] < total_frames:
        right = np.concatenate([right, np.full(total_frames - right.shape[0], bool(right[-1]) if right.size else False)])
    else:
        right = right[:total_frames]

    return [{toe_names[0]: bool(left[i]), toe_names[1]: bool(right[i])} for i in range(total_frames)]


def velocity_fallback_sequences(
    joints: np.ndarray,
    *,
    left_foot_idx: int = 10,
    right_foot_idx: int = 11,
    toe_names: tuple[str, str] = ("L_Foot", "R_Foot"),
    velocity_threshold: float = 0.01,
) -> list[dict[str, bool]]:
    """
    Fallback contact estimator using XY foot velocity threshold.
    """
    left_positions = joints[:, left_foot_idx, :2]
    right_positions = joints[:, right_foot_idx, :2]
    left_vel = np.linalg.norm(np.diff(left_positions, axis=0), axis=1)
    right_vel = np.linalg.norm(np.diff(right_positions, axis=0), axis=1)

    left_vel = np.concatenate([[velocity_threshold + 1], left_vel])
    right_vel = np.concatenate([[velocity_threshold + 1], right_vel])
    return [
        {
            toe_names[0]: bool(left_vel[idx] <= velocity_threshold),
            toe_names[1]: bool(right_vel[idx] <= velocity_threshold),
        }
        for idx in range(int(joints.shape[0]))
    ]

