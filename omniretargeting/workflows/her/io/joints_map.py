"""
Joint naming/order conversion for HER workflow.

Primary standard in this workflow:
- 22-joint SMPL-X body order used by omniretargeting README/core:
  Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2, L_Ankle, R_Ankle,
  Spine3, L_Foot, R_Foot, Neck, L_Collar, R_Collar, Head,
  L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist
"""

from __future__ import annotations

import numpy as np

# Full SMPL-X joint names used by Her/Holosoma pipeline (stable name->index mapping).
SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",
    "left_mouth_4",
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

OMNI_SMPLX22_JOINTS = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

_OMNI_SMPLX22_NAME_MAP = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip",
    "R_Hip": "right_hip",
    "Spine1": "spine1",
    "L_Knee": "left_knee",
    "R_Knee": "right_knee",
    "Spine2": "spine2",
    "L_Ankle": "left_ankle",
    "R_Ankle": "right_ankle",
    "Spine3": "spine3",
    "L_Foot": "left_foot",
    "R_Foot": "right_foot",
    "Neck": "neck",
    "L_Collar": "left_collar",
    "R_Collar": "right_collar",
    "Head": "head",
    "L_Shoulder": "left_shoulder",
    "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow",
    "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist",
    "R_Wrist": "right_wrist",
}

_SMPLX_NAME_TO_INDEX = {name: idx for idx, name in enumerate(SMPLX_JOINT_NAMES)}
_OMNI_SMPLX22_INDICES = np.asarray(
    [_SMPLX_NAME_TO_INDEX[_OMNI_SMPLX22_NAME_MAP[name]] for name in OMNI_SMPLX22_JOINTS],
    dtype=np.int64,
)


def remap_smplx_to_omni_smplx22(
    joints_smplx: np.ndarray,
    *,
    assume_input_is_omni_smplx22: bool = False,
) -> np.ndarray:
    """
    Remap SMPL-X joints to omniretargeting's canonical 22-joint order.

    Args:
        joints_smplx: (..., J, 3) SMPL-X joints. Expected J=len(SMPLX_JOINT_NAMES).
        assume_input_is_omni_smplx22: if True and J==22, treat input as already in canonical order.
    """
    joint_count = int(joints_smplx.shape[-2])
    if joint_count == len(SMPLX_JOINT_NAMES):
        return joints_smplx[..., _OMNI_SMPLX22_INDICES, :]
    if joint_count == len(OMNI_SMPLX22_JOINTS) and assume_input_is_omni_smplx22:
        return joints_smplx
    raise ValueError(
        f"Cannot safely remap joints with J={joint_count}. "
        f"Expected J={len(SMPLX_JOINT_NAMES)} (full SMPL-X). "
        "If your input is already in omni 22-joint order, set "
        "`assume_input_is_omni_smplx22=True` explicitly."
    )

