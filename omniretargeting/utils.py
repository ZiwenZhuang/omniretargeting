"""Utility functions for OmniRetargeting."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation


def load_terrain_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load terrain mesh from various formats."""
    supported_formats = ['.obj', '.stl', '.ply', '.gltf', '.glb']

    if mesh_path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported mesh format: {mesh_path.suffix}. "
                        f"Supported formats: {supported_formats}")

    try:
        mesh = trimesh.load(str(mesh_path))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid mesh: {type(mesh)}")
        return mesh
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {mesh_path}: {e}")


def compute_mesh_bounding_box(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the bounding box of a mesh."""
    return mesh.bounds[0], mesh.bounds[1]  # min_point, max_point


def scale_mesh(mesh: trimesh.Trimesh, scale_factor: float) -> trimesh.Trimesh:
    """Scale a mesh by a given factor."""
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)
    return scaled_mesh


def transform_mesh(mesh: trimesh.Trimesh,
                  translation: np.ndarray,
                  rotation: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """Transform a mesh with translation and optional rotation."""
    transformed_mesh = mesh.copy()

    if rotation is not None:
        # Apply rotation first
        rot_matrix = Rotation.from_quat(rotation).as_matrix()
        transformed_mesh.apply_transform(rot_matrix)

    # Apply translation
    transformed_mesh.apply_translation(translation)

    return transformed_mesh


def sample_points_on_mesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample points uniformly on the surface of a mesh."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def compute_mesh_height_at_point(mesh: trimesh.Trimesh, x: float, y: float) -> float:
    """Compute the height (z) of the mesh at a given (x, y) position."""
    # Create a ray from above the point downward
    ray_origin = np.array([x, y, 100.0])  # High z value
    ray_direction = np.array([0, 0, -1])  # Downward

    # Find intersections with the mesh
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=[ray_origin],
        ray_directions=[ray_direction]
    )

    if len(locations) == 0:
        # No intersection found, return a default height
        return 0.0

    # Return the highest intersection point (closest to the ray origin)
    return locations[0][2]


def align_terrain_to_coordinates(mesh: trimesh.Trimesh,
                               reference_points: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Align terrain mesh to match reference coordinate system.

    Args:
        mesh: Input terrain mesh
        reference_points: Reference points defining the coordinate system

    Returns:
        Tuple of (aligned_mesh, transformation_matrix)
    """
    # Simple alignment: translate mesh so that its center matches the origin
    mesh_center = mesh.centroid
    translation = -mesh_center

    aligned_mesh = mesh.copy()
    aligned_mesh.apply_translation(translation)

    # For now, return identity transformation
    # TODO: Implement proper coordinate system alignment
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    return aligned_mesh, transformation


def validate_smplx_trajectory(trajectory: np.ndarray) -> bool:
    """Validate SMPLX trajectory format."""
    if len(trajectory.shape) != 3:
        return False

    num_frames, num_joints, num_coords = trajectory.shape

    if num_coords != 3:
        return False

    if num_frames == 0 or num_joints == 0:
        return False

    # Check for NaN or infinite values
    if not np.isfinite(trajectory).all():
        return False

    return True


def extract_smplx_joint_positions(trajectory: np.ndarray,
                                joint_indices: list) -> np.ndarray:
    """Extract specific joint positions from SMPLX trajectory."""
    return trajectory[:, joint_indices, :]


def convert_quaternion_format(quaternions: np.ndarray,
                            input_format: str = 'wxyz',
                            output_format: str = 'xyzw') -> np.ndarray:
    """Convert between quaternion formats."""
    if input_format == output_format:
        return quaternions.copy()

    if input_format == 'wxyz' and output_format == 'xyzw':
        return quaternions[:, [1, 2, 3, 0]]
    elif input_format == 'xyzw' and output_format == 'wxyz':
        return quaternions[:, [3, 0, 1, 2]]
    else:
        raise ValueError(f"Unsupported conversion: {input_format} -> {output_format}")


def transform_points_local_to_world(quat, trans, points_local):
    """Transform points from local frame to world frame."""
    transform_matrix = trimesh.transformations.quaternion_matrix(quat)
    transform_matrix[:3, 3] = trans
    hom_points = np.hstack([points_local, np.ones((points_local.shape[0], 1))])
    transformed_points_hom = (transform_matrix @ hom_points.T).T
    return transformed_points_hom[:, :3]


def get_adjacency_list(tetrahedra, num_vertices):
    """Creates an adjacency list from the tetrahedra."""
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_laplacian_coordinates(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian coordinates for each vertex in the mesh.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): Whether to use uniform weights.

    Returns:
        np.ndarray: (N, 3) array of Laplacian coordinates.
    """
    laplacian = np.zeros_like(vertices)

    for i in range(len(vertices)):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            vi = vertices[i]
            neighbor_positions = vertices[neighbors_indices]
            distances = np.linalg.norm(vi - neighbor_positions, axis=1)

            if uniform_weight:
                weights = np.ones_like(distances)
            else:
                weights = 1.0 / (1.5 * distances + epsilon)

            sum_of_weights = np.sum(weights)
            weighted_sum_of_neighbors = np.sum(weights[:, np.newaxis] * neighbor_positions, axis=0)
            center_of_neighbors = weighted_sum_of_neighbors / sum_of_weights
            laplacian[i] = vi - center_of_neighbors

    return laplacian


def calculate_laplacian_matrix(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian matrix for the mesh with optional weight schemes.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): If True, use uniform weights; if False, use distance-based weights.

    Returns:
        np.ndarray: (N, N) Laplacian matrix.
    """
    N = len(vertices)
    laplacian_matrix = np.zeros((N, N))

    for i in range(N):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            if uniform_weight:
                weights = np.ones(len(neighbors_indices)) / len(neighbors_indices)
            else:
                vi = vertices[i]
                neighbor_positions = vertices[neighbors_indices]
                distances = np.linalg.norm(vi - neighbor_positions, axis=1)
                weights = 1.0 / (distances + epsilon)
                sum_weights = np.sum(weights)
                weights = weights / sum_weights

            laplacian_matrix[i, i] = 1.0

            for j, neighbor_idx in enumerate(neighbors_indices):
                laplacian_matrix[i, neighbor_idx] = -weights[j]

    return laplacian_matrix


def compute_world_joint_orientations(
    global_orient: np.ndarray,
    full_pose: np.ndarray,
    parents: np.ndarray,
    num_body_joints: int = 22,
) -> np.ndarray:
    """
    Compute world-frame joint orientations from SMPLX pose parameters.
    
    Args:
        global_orient: Root orientation in axis-angle format, shape (T, 3)
        full_pose: Full pose parameters in axis-angle format, shape (T, J_total, 3)
        parents: Parent indices for kinematic tree, shape (J_total,)
        num_body_joints: Number of body joints to return (default 22)
    
    Returns:
        Joint orientations as quaternions (wxyz format), shape (T, J, 4)
    """
    num_frames = global_orient.shape[0]
    num_joints = min(full_pose.shape[1], num_body_joints)
    
    # Output: quaternions in wxyz format
    joint_orientations = np.zeros((num_frames, num_joints, 4))
    
    for frame_idx in range(num_frames):
        # Store rotations for this frame
        frame_rotations = []
        
        for joint_idx in range(num_joints):
            if joint_idx == 0:
                # Root joint: use global_orient directly
                rot = Rotation.from_rotvec(global_orient[frame_idx])
            else:
                # Other joints: multiply parent's world rotation by local rotation
                parent_idx = parents[joint_idx]
                if parent_idx >= 0 and parent_idx < len(frame_rotations):
                    parent_rot = frame_rotations[parent_idx]
                    local_rot = Rotation.from_rotvec(full_pose[frame_idx, joint_idx])
                    rot = parent_rot * local_rot
                else:
                    # Fallback: use local rotation as world rotation
                    rot = Rotation.from_rotvec(full_pose[frame_idx, joint_idx])
            
            frame_rotations.append(rot)
            # Store as quaternion in wxyz format (scalar_first=True)
            joint_orientations[frame_idx, joint_idx] = rot.as_quat(scalar_first=True)
    
    return joint_orientations


def load_smplx_trajectory(
    smplx_file: Path,
    smplx_model_directory: Optional[str] = None,
    gender: str = "neutral",
    return_meta: bool = False,
) -> tuple[np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Load SMPLX trajectory from file.
    
    Handles:
    - Standard .npy files with pre-computed joint positions
    - Pre-processed .npz files with 'global_joint_positions' key
    - Raw SMPLX-NG .npz files (e.g., stageii.npz) with SMPLX parameters
    
    For raw SMPLX-NG files, requires smplx_model_directory to perform forward kinematics.
    
    Args:
        smplx_file: Path to trajectory file (.npy or .npz)
        smplx_model_directory: Path to SMPLX model files (required for raw SMPLX-NG files and orientation computation)
        gender: Gender for SMPLX model ('neutral', 'male', 'female')
        return_meta: If True, return (positions, orientations, root_orient, trans) instead of (positions, orientations)
    
    Returns:
        Tuple of (positions, orientations) or (positions, orientations, root_orient, trans):
        
        If return_meta=False (default):
            - positions: Joint positions array of shape (T, J, 3)
            - orientations: Joint orientations as quaternions (wxyz), shape (T, J, 4)
                           Returns None if orientations cannot be computed (e.g., .npy files)
        
        If return_meta=True:
            - positions: Joint positions array of shape (T, J, 3)
            - orientations: Joint orientations as quaternions (wxyz), shape (T, J, 4)
                           Returns None if orientations cannot be computed
            - root_orient: Root orientation parameters in axis-angle format, shape (T, 3) or None
            - trans: Translation parameters, shape (T, 3) or None
    """
    if smplx_file.suffix == ".npy":
        joints = np.load(smplx_file, allow_pickle=True)
        # Cannot compute orientations from positions only
        print("Warning: Cannot compute orientations from .npy file (positions only). Returning None for orientations.")
        if return_meta:
            return joints, None, None, None
        return joints, None

    smplx_data = np.load(smplx_file, allow_pickle=True)

    if isinstance(smplx_data, np.lib.npyio.NpzFile) and "global_joint_positions" in smplx_data:
        joints = smplx_data["global_joint_positions"]
        root_orient = smplx_data["root_orient"] if "root_orient" in smplx_data else None
        trans = smplx_data["trans"] if "trans" in smplx_data else None
        
        # Try to compute orientations if we have the necessary data
        orientations = None
        if "full_pose" in smplx_data and smplx_model_directory is not None and root_orient is not None:
            # Load body model to get parent structure
            body_model = smplx.create(
                smplx_model_directory,
                "smplx",
                gender=gender,
                use_pca=False,
            )
            full_pose = smplx_data["full_pose"]
            if isinstance(full_pose, np.ndarray) and len(full_pose.shape) == 2:
                # Reshape from (T, J*3) to (T, J, 3)
                full_pose = full_pose.reshape(full_pose.shape[0], -1, 3)
            
            orientations = compute_world_joint_orientations(
                root_orient,
                full_pose,
                body_model.parents.cpu().numpy(),
                num_body_joints=22,
            )
        else:
            print("Warning: Cannot compute orientations from .npz file (missing full_pose, root_orient, or model directory). Returning None for orientations.")
        
        if return_meta:
            return joints, orientations, root_orient, trans
        return joints, orientations

    # Raw SMPLX-NG file - need to run forward kinematics
    body_model = smplx.create(
        smplx_model_directory,
        "smplx",
        gender=str(smplx_data["gender"]),
        use_pca=False,
    )

    # Extract and adjust betas
    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    if betas.shape[1] > 10:
        betas = betas[:, :10]

    num_frames = smplx_data["pose_body"].shape[0]
    root_orient = smplx_data["root_orient"]
    trans = smplx_data["trans"]
    smplx_output = body_model(
        betas=betas,  # Shape parameters (1, 10)
        global_orient=torch.tensor(root_orient).float(),  # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),  # (N, 63)
        transl=torch.tensor(trans).float(),  # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),  # Pose parameters per frame
        right_hand_pose=torch.zeros(num_frames, 45).float(),  # Pose parameters per frame
        jaw_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        leye_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        reye_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        expression=torch.zeros(num_frames, 10).float(),  # Expression parameters expanded to num_frames
        return_full_pose=True,
    )

    # Extract joint positions
    joints = smplx_output.joints.detach().cpu().numpy()

    # Return only body joints (first 22)
    joints = joints[:, :22, :]
    
    # Compute world-frame orientations
    full_pose = smplx_output.full_pose.detach().cpu().numpy()
    # Reshape from (T, J*3) to (T, J, 3)
    full_pose = full_pose.reshape(num_frames, -1, 3)
    
    orientations = compute_world_joint_orientations(
        root_orient,
        full_pose,
        body_model.parents.cpu().numpy(),
        num_body_joints=22,
    )
    
    if return_meta:
        return joints, orientations, root_orient, trans
    return joints, orientations
