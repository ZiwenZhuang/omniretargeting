"""
Contact-driven vertical correction (replaces global z_min shift).
"""

from __future__ import annotations

import numpy as np


def split_left_right_contact(contact_flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpret contact flags as:
    - [0,1] -> left foot
    - [2,3] -> right foot
    """
    flags = np.asarray(contact_flags, dtype=bool)
    if flags.ndim != 2 or flags.shape[1] < 1:
        raise ValueError(f"Expected contact flags shape (T, C>=1), got {flags.shape}")
    left = flags[:, 0] | flags[:, 1] if flags.shape[1] >= 2 else flags[:, 0]
    right = flags[:, 2] | flags[:, 3] if flags.shape[1] >= 4 else np.zeros(flags.shape[0], dtype=bool)
    return left, right


def apply_contact_height_correction(
    joints: np.ndarray,
    *,
    contact_flags: np.ndarray | None,
    left_foot_idx: int = 10,
    right_foot_idx: int = 11,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Shift each frame along Z so contacting foot(s) stay close to z=0, then interpolate
    per-frame bias across non-contact gaps for smoothness.

    Returns:
    - corrected_joints: same shape as input
    - z_bias_per_frame: (T,) or None when no correction is applied
    """
    corrected = np.asarray(joints, dtype=np.float64).copy()
    if corrected.ndim != 3 or corrected.shape[2] != 3:
        raise ValueError(f"Expected joints shape (T, J, 3), got {corrected.shape}")
    if contact_flags is None:
        return corrected.astype(joints.dtype, copy=False), None

    left, right = split_left_right_contact(contact_flags)
    frame_count = int(corrected.shape[0])
    if left.shape[0] < frame_count:
        left = np.concatenate([left, np.full(frame_count - left.shape[0], bool(left[-1]) if left.size else False)])
    else:
        left = left[:frame_count]
    if right.shape[0] < frame_count:
        right = np.concatenate([right, np.full(frame_count - right.shape[0], bool(right[-1]) if right.size else False)])
    else:
        right = right[:frame_count]

    in_contact = left | right
    contact_indices = np.where(in_contact)[0]
    if contact_indices.size == 0:
        return corrected.astype(joints.dtype, copy=False), None

    observed_bias = np.full((frame_count,), np.nan, dtype=np.float64)
    for frame_idx in contact_indices.tolist():
        z_values: list[float] = []
        if bool(left[frame_idx]):
            z_values.append(float(corrected[frame_idx, left_foot_idx, 2]))
        if bool(right[frame_idx]):
            z_values.append(float(corrected[frame_idx, right_foot_idx, 2]))
        if z_values:
            observed_bias[frame_idx] = float(np.mean(z_values))

    valid = np.isfinite(observed_bias)
    if not np.any(valid):
        return corrected.astype(joints.dtype, copy=False), None

    x = np.where(valid)[0].astype(np.float64)
    y = observed_bias[valid].astype(np.float64)
    bias = np.interp(np.arange(frame_count, dtype=np.float64), x, y).astype(np.float64)
    corrected[:, :, 2] -= bias[:, None]
    return corrected.astype(joints.dtype, copy=False), bias

