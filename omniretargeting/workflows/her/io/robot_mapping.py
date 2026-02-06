"""
Default robot URDF paths and SMPLX22->robot link mappings for HER workflow.
"""

from __future__ import annotations

from pathlib import Path

from omniretargeting.workflows.her.config import RobotType

DEFAULT_ROBOT_URDF_PATHS: dict[RobotType, Path] = {
    "g1": Path("/home/juyiang/data/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"),
    "t1": Path("/home/juyiang/data/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/t1/t1_23dof.urdf"),
}

DEFAULT_SMPLX22_JOINT_MAPPINGS: dict[RobotType, dict[str, str]] = {
    "g1": {
        "Pelvis": "pelvis_contour_link",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_roll_link",
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_intermediate_1_link",
        "R_Ankle": "right_ankle_intermediate_1_link",
        "L_Foot": "left_ankle_roll_sphere_5_link",
        "R_Foot": "right_ankle_roll_sphere_5_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    "t1": {
        "Pelvis": "Trunk",
        "L_Hip": "Hip_Pitch_Left",
        "R_Hip": "Hip_Pitch_Right",
        "L_Knee": "Shank_Left",
        "R_Knee": "Shank_Right",
        "L_Shoulder": "AL1",
        "R_Shoulder": "AR1",
        "L_Elbow": "left_hand_link",
        "R_Elbow": "right_hand_link",
        "L_Ankle": "Ankle_Cross_Left",
        "R_Ankle": "Ankle_Cross_Right",
        "L_Foot": "left_foot_sphere_5_link",
        "R_Foot": "right_foot_sphere_5_link",
        "L_Wrist": "left_hand_sphere_link",
        "R_Wrist": "right_hand_sphere_link",
    },
}

# Foot sticking links (used for contact-driven foot sticking constraints).
# These match the holosoma_retargeting convention: multiple small sphere links under each foot.
DEFAULT_FOOT_STICKING_LINKS: dict[RobotType, list[str]] = {
    "g1": [
        "left_ankle_roll_sphere_1_link",
        "right_ankle_roll_sphere_1_link",
        "left_ankle_roll_sphere_2_link",
        "right_ankle_roll_sphere_2_link",
        "left_ankle_roll_sphere_3_link",
        "right_ankle_roll_sphere_3_link",
        "left_ankle_roll_sphere_4_link",
        "right_ankle_roll_sphere_4_link",
    ],
    "t1": [
        "left_foot_sphere_1_link",
        "right_foot_sphere_1_link",
        "left_foot_sphere_2_link",
        "right_foot_sphere_2_link",
        "left_foot_sphere_3_link",
        "right_foot_sphere_3_link",
        "left_foot_sphere_4_link",
        "right_foot_sphere_4_link",
        "left_foot_sphere_5_link",
        "right_foot_sphere_5_link",
    ],
}


def get_default_joint_mapping(robot: RobotType) -> dict[str, str]:
    if robot not in DEFAULT_SMPLX22_JOINT_MAPPINGS:
        raise ValueError(f"No default joint mapping for robot: {robot}")
    return dict(DEFAULT_SMPLX22_JOINT_MAPPINGS[robot])


def get_default_foot_sticking_links(robot: RobotType) -> list[str]:
    if robot not in DEFAULT_FOOT_STICKING_LINKS:
        raise ValueError(f"No default foot sticking links for robot: {robot}")
    return list(DEFAULT_FOOT_STICKING_LINKS[robot])
