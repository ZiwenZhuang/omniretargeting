"""Core OmniRetargeting functionality."""

from __future__ import annotations

import numpy as np
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import trimesh
import mujoco
import yourdfpy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class OmniRetargeter:
    """
    Generic motion retargeting for any humanoid URDF and terrain mesh.

    This class provides functionality to retarget SMPLX trajectories to any humanoid robot
    operating on any terrain mesh by automatically scaling the terrain and computing
    appropriate joint mappings.
    """

    def __init__(
        self,
        robot_urdf_path: Union[str, Path],
        terrain_mesh_path: Union[str, Path],
        joint_mapping: Dict[str, str],
        robot_height: Optional[float] = None,
        smplx_joint_names: Optional[List[str]] = None,
        robot_mujoco_xml_path: Optional[Union[str, Path]] = None,
        debug_frames: int = 5,
        enable_penetration_constraints: bool = True,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        max_penetration_constraints: int = 64,
        penetration_constraint_mode: str = "soft",
        penetration_slack_weight: float = 1e4,
        terrain_scale_override: float | None = None,
        collision_proxy_boxes: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """
        Initialize the OmniRetargeter.

        Args:
            robot_urdf_path: Path to the humanoid robot URDF file
            terrain_mesh_path: Path to the terrain mesh file (any common format)
            joint_mapping: Dictionary mapping SMPLX joint names to robot link names
            robot_height: Height of the robot in meters (auto-detected if None)
            smplx_joint_names: List of SMPLX joint names in order (required for proper joint mapping)
            robot_mujoco_xml_path: Optional MuJoCo MJCF XML path for loading the robot model.
                Recommended when URDF mesh resolution/loading is problematic in MuJoCo.
            debug_frames: Print solver debug output for the first N frames.
            enable_penetration_constraints: If True, add non-penetration constraints against the terrain.
            collision_detection_threshold: MuJoCo distance threshold for collision candidate pairs.
            penetration_tolerance: Allowed penetration slack (meters).
            terrain_scale_override: Optional fixed terrain scale factor. If provided, the retargeter
                will not estimate human height from the trajectory, and will use this scale to scale
                both terrain and human joints. This is recommended when you want consistent scaling
                across pipeline/visualization (e.g., assume human height is 1.7m).
            collision_proxy_boxes: Optional (centers, half_sizes) arrays for primitive collision proxies,
                both in *scaled* world coordinates. When provided, these geoms are injected into the
                collision model and used for non-penetration.
        """
        self.robot_urdf_path = Path(robot_urdf_path)
        self.terrain_mesh_path = Path(terrain_mesh_path)
        self.joint_mapping = joint_mapping
        self.debug_frames = max(0, int(debug_frames))
        self.enable_penetration_constraints = bool(enable_penetration_constraints)
        self.collision_detection_threshold = float(collision_detection_threshold)
        self.penetration_tolerance = float(penetration_tolerance)
        self.max_penetration_constraints = max(0, int(max_penetration_constraints))
        self.penetration_constraint_mode = str(penetration_constraint_mode)
        self.penetration_slack_weight = float(penetration_slack_weight)
        self.terrain_scale_override = None if terrain_scale_override is None else float(terrain_scale_override)
        self.collision_proxy_boxes = collision_proxy_boxes

        # SMPLX joint names (default to standard SMPLX joint ordering)
        if smplx_joint_names is None:
            # Standard SMPLX joint names (first 22 body joints)
            self.smplx_joint_names = [
                "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
                "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
                "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
                "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
            ]
        else:
            self.smplx_joint_names = smplx_joint_names

        # Create mapping from SMPLX joint names to indices
        self.smplx_joint_indices = {}
        for idx, name in enumerate(self.smplx_joint_names):
            self.smplx_joint_indices[name] = idx

        # Get indices for mapped joints
        # CRITICAL: Only include joints that exist in SMPLX AND will be validated in robot
        # Store the valid joint names to ensure consistent ordering
        self.valid_joint_names = []  # Joint names that exist in both SMPLX and robot
        self.mapped_joint_indices = []
        
        for smplx_joint_name in joint_mapping.keys():
            if smplx_joint_name in self.smplx_joint_indices:
                self.mapped_joint_indices.append(self.smplx_joint_indices[smplx_joint_name])
                self.valid_joint_names.append(smplx_joint_name)
            else:
                raise ValueError(
                    f"SMPLX joint '{smplx_joint_name}' not found in SMPLX joint names. "
                    f"Available joints: {list(self.smplx_joint_indices.keys())[:10]}..."
                )

        if len(self.mapped_joint_indices) == 0:
            raise ValueError("No valid joint mappings found. Please check your joint_mapping dictionary.")
        
        # Store the filtered joint mapping (only valid joints) for use in retargeter
        self.valid_joint_mapping = {name: joint_mapping[name] for name in self.valid_joint_names}

        # Load robot description for visualization / metadata (best-effort).
        # Retargeting itself depends on MuJoCo model; URDF parsing is optional.
        self.robot_urdf = None
        if self.robot_urdf_path.suffix.lower() == ".urdf":
            try:
                self.robot_urdf = yourdfpy.URDF.load(str(self.robot_urdf_path), load_meshes=True)
            except Exception as exc:
                print(f"Warning: yourdfpy failed to load meshes from URDF ({self.robot_urdf_path}): {exc}")
                try:
                    self.robot_urdf = yourdfpy.URDF.load(str(self.robot_urdf_path), load_meshes=False)
                except Exception:
                    self.robot_urdf = None

        # Load MuJoCo model.
        # NOTE: MuJoCo's URDF importer can be fragile with mesh path resolution; prefer MJCF if available.
        if robot_mujoco_xml_path is not None:
            mujoco_model_path = Path(robot_mujoco_xml_path)
        elif self.robot_urdf_path.suffix.lower() == ".urdf":
            adjacent_xml = self.robot_urdf_path.with_suffix(".xml")
            mujoco_model_path = adjacent_xml if adjacent_xml.exists() else self.robot_urdf_path
        else:
            mujoco_model_path = self.robot_urdf_path
        try:
            self.robot_model = mujoco.MjModel.from_xml_path(str(mujoco_model_path))
        except Exception as exc:
            # Fallback: if an URDF is provided, try adjacent .xml (common in holosoma_retargeting assets).
            fallback_xml = self.robot_urdf_path.with_suffix(".xml")
            if self.robot_urdf_path.suffix.lower() == ".urdf" and fallback_xml.exists():
                print(
                    f"Warning: MuJoCo failed to load URDF ({mujoco_model_path}); trying MJCF fallback {fallback_xml}."
                )
                self.robot_model = mujoco.MjModel.from_xml_path(str(fallback_xml))
                mujoco_model_path = fallback_xml
            else:
                raise RuntimeError(
                    f"Failed to load MuJoCo model from {mujoco_model_path}. "
                    "If you are providing a URDF, consider passing robot_mujoco_xml_path pointing to a MJCF .xml."
                ) from exc
        self.robot_mujoco_xml_path = mujoco_model_path
        self.robot_data = mujoco.MjData(self.robot_model)

        # Load terrain mesh
        print(f"Loaded terrain: {str(terrain_mesh_path)}")
        self.terrain_mesh = trimesh.load(str(terrain_mesh_path))

        # Detect robot height if not provided
        if robot_height is None:
            self.robot_height = self._detect_robot_height()
        else:
            self.robot_height = robot_height

        # Initialize retargeting components
        self._setup_retargeting_components()

    def _detect_robot_height(self) -> float:
        """
        Detect robot height from URDF by calculating the vertical span of the robot in its default configuration.
        
        Since this is a floating-base robot, we assume the default configuration puts the robot
        in a nominal pose (e.g. standing). We calculate the difference between the highest
        and lowest points of all visual meshes to get the full height.
        """
        # Use MuJoCo to get robot height in default configuration
        # This is more reliable than parsing the URDF scene graph
        
        # Set robot to default configuration (zeros)
        mujoco.mj_resetData(self.robot_model, self.robot_data)
        self.robot_data.qpos[3:7] = [1, 0, 0, 0]  # Set base quaternion to identity
        mujoco.mj_forward(self.robot_model, self.robot_data)
        
        min_z = float('inf')
        max_z = float('-inf')
        
        # Iterate through all bodies and get their positions
        for body_idx in range(self.robot_model.nbody):
            body_pos = self.robot_data.xpos[body_idx]
            z = body_pos[2]
            min_z = min(min_z, z)
            max_z = max(max_z, z)
        
        # Also check geometry positions for more accuracy
        for geom_idx in range(self.robot_model.ngeom):
            geom_pos = self.robot_data.geom_xpos[geom_idx]
            z = geom_pos[2]
            
            # Get geometry size to account for extent
            geom_size = self.robot_model.geom_size[geom_idx]
            geom_type = self.robot_model.geom_type[geom_idx]
            
            # Estimate vertical extent based on geometry type
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = geom_size[0]
                min_z = min(min_z, z - radius)
                max_z = max(max_z, z + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = geom_size[0]
                half_height = geom_size[1]
                min_z = min(min_z, z - half_height - radius)
                max_z = max(max_z, z + half_height + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_size = geom_size[2]  # Z dimension
                min_z = min(min_z, z - half_size)
                max_z = max(max_z, z + half_size)
            else:
                # For other types, just use the position
                min_z = min(min_z, z)
                max_z = max(max_z, z)
        
        height = max_z - min_z
        
        # Sanity check
        if height < 0.3 or height > 3.0:
            print(f"Warning: Detected height {height:.3f}m seems unreasonable. Using default 1.6m")
            return 1.6
        
        print(f"Detected robot height: {height:.4f}m (Min Z: {min_z:.4f}, Max Z: {max_z:.4f})")
        
        return height

    def _setup_retargeting_components(self):
        """Setup internal retargeting components."""
        # Validate joint mapping and filter out invalid joints
        # CRITICAL: Only keep joints that exist in BOTH SMPLX and robot URDF
        # This ensures consistent sizes between human_joints and robot_points
        
        # First, filter out joints missing from robot URDF
        missing_robot_links = self.validate_joint_mapping()
        if missing_robot_links:
            print(f"Warning: The following robot links from joint_mapping were not found in URDF: {missing_robot_links}")
            print("These joints will be removed from the mapping.")
            print(f"\nAvailable robot bodies:")
            for i in range(min(20, self.robot_model.nbody)):
                body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(f"  {body_name}")
            if self.robot_model.nbody > 20:
                print(f"  ... and {self.robot_model.nbody - 20} more")
            
            # Remove joints with missing robot links from valid_joint_names
            self.valid_joint_names = [
                name for name in self.valid_joint_names 
                if self.valid_joint_mapping[name] not in missing_robot_links
            ]
            # Rebuild mapped_joint_indices and valid_joint_mapping
            self.mapped_joint_indices = [
                self.smplx_joint_indices[name] for name in self.valid_joint_names
            ]
            self.valid_joint_mapping = {name: self.joint_mapping[name] for name in self.valid_joint_names}
        
        if len(self.valid_joint_names) == 0:
            raise ValueError(
                "No valid joint mappings found after filtering. "
                "Please check that joint_mapping contains joints that exist in both SMPLX and robot URDF."
            )
        
        print(f"Successfully initialized retargeting with {len(self.valid_joint_names)} mapped joints.")
        print(f"Valid joints: {self.valid_joint_names}")
        print(f"Robot height: {self.robot_height:.3f}m")
        print(f"Robot DOF: {self.get_robot_dof()}")

    def retarget_motion(
        self,
        smplx_trajectory: np.ndarray,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
        visualize_trajectory: bool = True,
    ) -> Tuple[float, np.ndarray]:
        """
        Retarget SMPLX motion to the robot on the terrain.

        Args:
            smplx_trajectory: SMPLX joint positions of shape (T, J, 3) where T is frames, J is joints
            terrain_coordinates: Optional terrain coordinate system reference points

        Returns:
            Tuple of (terrain_scale, retargeted_trajectory)
            - terrain_scale: Scalar factor to scale the terrain mesh
            - retargeted_trajectory: Robot motion of shape (T, 7 + DOF) with [pos, quat, joints]
        """
        # Step 1: Compute terrain scaling factor
        if self.terrain_scale_override is not None:
            terrain_scale = float(self.terrain_scale_override)
            print(f"Using terrain scale override: {terrain_scale:.6f}")
        else:
            terrain_scale = self._compute_terrain_scale(smplx_trajectory)

        # Step 2: Scale terrain mesh
        scaled_terrain = self._scale_terrain_mesh(terrain_scale)

        # Step 3: Process SMPLX trajectory
        processed_trajectory = self._process_smplx_trajectory(smplx_trajectory, terrain_scale)
        processed_base_translations = None
        if base_translations is not None:
            processed_base_translations = base_translations * terrain_scale

        # Option Step:
        if visualize_trajectory:
            # visualize the processed_trajectory
            self._visualize_trajectory(processed_trajectory, scaled_terrain)

        # Step 4: Perform retargeting using the generic retargeting system
        retargeted_motion = self._perform_retargeting(
            processed_trajectory,
            scaled_terrain,
            terrain_scale,
            base_orientations=base_orientations,
            base_translations=processed_base_translations,
        )

        return terrain_scale, retargeted_motion

    def _compute_terrain_scale(
        self,
        smplx_trajectory: np.ndarray,
    ) -> float:
        """
        Compute the appropriate scaling factor for the terrain.
        
        This implementation assumes the terrain provided corresponds to the real-world scale
        relative to the robot's size.
        """
        # Default implementation: Scale based on robot vs human height ratio only
        # This assumes the terrain mesh is already at a "human scale" (e.g. captured or designed for humans)
        # and we just need to adjust it for the robot's size.
        
        # Estimate human height from SMPLX trajectory
        # Height ≈ (head Z - min foot Z) in standing pose
        # Or more robustly: Max height difference in trajectory
        
        # Use simple heuristic if we don't have betas to calculate exact height
        # Assuming typical human height of 1.7m if estimation fails or is unreliable
        estimated_human_height = 1.7
        
        if smplx_trajectory is not None and len(smplx_trajectory) > 0:
            # Estimate height from joints
            # SMPLX/SMPL joints: 0: Pelvis, 10/11: Feet, 15: Head (approx)
            # We can find the max vertical extent across all frames
            
            # Find the frame where the person is most upright (max head height)
            # Assuming Z is up
            head_idx = 15 # Head
            foot_indices = [10, 11] # Left/Right ankle
            
            # Check if we have enough joints
            if smplx_trajectory.shape[1] > 15:
                # Calculate height per frame: Head Z - Average Foot Z
                head_z = smplx_trajectory[:, head_idx, 2]
                
                # Use min foot Z per frame as ground reference relative to body
                feet_z = np.min(smplx_trajectory[:, foot_indices, 2], axis=1)
                
                # Height per frame
                heights = head_z - feet_z
                
                # Use the maximum height observed (standing pose)
                # Add offset for top of head (head joint is in neck/center of head)
                # Approx 10-15cm from head joint to top of head
                HEAD_TOP_OFFSET = 0.12
                estimated_human_height = np.max(heights) + HEAD_TOP_OFFSET
                
                # Sanity check: Constrain to reasonable human range [1.4, 2.2]
                estimated_human_height = np.clip(estimated_human_height, 1.4, 2.2)
                
                print(f"Estimated human height from trajectory: {estimated_human_height:.3f}m")

        scale_factor = self.robot_height / estimated_human_height
        
        print(f"Computed terrain scale factor: {scale_factor:.4f} (Robot: {self.robot_height}m, Human Est: {estimated_human_height:.3f}m)")
        
        return float(scale_factor)

    def _extract_foot_positions(self, smplx_trajectory: np.ndarray) -> np.ndarray:
        """Extract foot positions from SMPLX trajectory."""
        # SMPLX joint indices for feet
        # L_Foot: 10, R_Foot: 11 (standard SMPLX ordering)
        foot_indices = [10, 11]

        foot_positions = []
        for frame in smplx_trajectory:
            for foot_idx in foot_indices:
                if foot_idx < len(frame):
                    foot_positions.append(frame[foot_idx])

        return np.array(foot_positions)

    def _scale_terrain_mesh(self, scale_factor: float) -> trimesh.Trimesh:
        """Scale the terrain mesh by the given factor."""
        scaled_mesh = self.terrain_mesh.copy()
        scaled_mesh.apply_scale(scale_factor)
        return scaled_mesh

    def _process_smplx_trajectory(
        self,
        smplx_trajectory: np.ndarray,
        terrain_scale: float
    ) -> np.ndarray:
        """Process SMPLX trajectory for retargeting."""
        # Apply terrain scaling to trajectory coordinates
        scaled_trajectory = smplx_trajectory * terrain_scale

        # Transform coordinate system if needed (SMPLX uses different convention)
        # TODO: Add coordinate system transformation if needed

        return scaled_trajectory

    def _perform_retargeting(
        self,
        processed_trajectory: np.ndarray,
        scaled_terrain: trimesh.Trimesh,
        terrain_scale: float,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
    ) -> np.ndarray:
        """Perform the actual motion retargeting using generic interaction mesh retargeting."""
        from .retargeting import GenericInteractionRetargeter

        # Create retargeter instance
        # Note: scaled_terrain is already scaled by _scale_terrain_mesh
        # CRITICAL: Pass only valid_joint_mapping to ensure consistent sizes
        # This ensures robot_points and human_joints have the same number of joints
        solver_model = self.robot_model
        solver_data = self.robot_data
        if self.enable_penetration_constraints:
            collision_model = self._compile_collision_model(terrain_scale=float(terrain_scale))
            if collision_model is not None:
                solver_model = collision_model
                solver_data = mujoco.MjData(solver_model)

        retargeter = GenericInteractionRetargeter(
            solver_model,
            solver_data,
            scaled_terrain,
            self.valid_joint_mapping,  # Use filtered mapping, not full joint_mapping
            self.robot_height,
            penetration_tolerance=self.penetration_tolerance,
            collision_detection_threshold=self.collision_detection_threshold,
            valid_joint_names=self.valid_joint_names,  # CRITICAL: Pass ordered joint names for consistency
            debug_frames=self.debug_frames,
            enable_penetration_constraints=self.enable_penetration_constraints,
            max_penetration_constraints=self.max_penetration_constraints,
            penetration_constraint_mode=self.penetration_constraint_mode,
            penetration_slack_weight=self.penetration_slack_weight,
        )

        # Retarget each frame
        retargeted_trajectory = []
        
        # Initialize with a reasonable standing configuration
        q_init = np.zeros(self.robot_model.nq)
        q_init[3:7] = [1, 0, 0, 0]  # Identity quaternion (wxyz)
        
        # Set joint angles to mid-range to avoid limit violations
        # This is safer than all zeros which might violate limits
        for i in range(self.robot_model.njnt):
            qpos_adr = self.robot_model.jnt_qposadr[i]
            if qpos_adr >= 7:  # Skip floating base
                joint_range = self.robot_model.jnt_range[i]
                # Set to middle of range
                q_init[qpos_adr] = (joint_range[0] + joint_range[1]) / 2.0
        
        # Initial height: Use processed (scaled) trajectory root height or default
        # Assuming joint 0 is root
        if len(processed_trajectory) > 0:
            q_init[:3] = processed_trajectory[0, 0] # Start at first frame root pos
            # Maintain initial rotation (identity or from trajectory if available)
        else:
            q_init[2] = self.robot_height * 0.5  # Default height (Z coordinate)

        for frame_idx, human_joints in enumerate(processed_trajectory):
            # Only initialize base from SMPLX for the first frame
            # For subsequent frames, the optimization will handle base movement
            if frame_idx == 0:
                # Initialize base position from SMPLX root
                if base_translations is not None:
                    q_init[:3] = base_translations[frame_idx]
                else:
                    # Use SMPLX pelvis position as initial guess
                    q_init[:3] = human_joints[0]

                # Initialize base orientation
                if base_orientations is not None:
                    root_rotvec = base_orientations[frame_idx]
                    quat_xyzw = Rotation.from_rotvec(root_rotvec).as_quat()
                    q_init[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                else:
                    quat_xyzw = self._estimate_base_orientation_from_joints(human_joints)
                    if quat_xyzw is not None:
                        q_init[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            # For subsequent frames, q_init already contains the previous optimized configuration
            
            # Extract mapped joints using the proper indices
            # CRITICAL: The order MUST match valid_joint_names exactly
            # mapped_joint_indices[i] corresponds to valid_joint_names[i] for all i
            # human_joints shape: (num_smplx_joints, 3)
            
            # Validate order consistency before extraction
            if len(self.mapped_joint_indices) != len(self.valid_joint_names):
                raise ValueError(
                    f"Order mismatch: mapped_joint_indices has {len(self.mapped_joint_indices)} elements, "
                    f"but valid_joint_names has {len(self.valid_joint_names)} elements."
                )
            
            # Verify that each index corresponds to the correct joint name
            for i, (joint_name, smplx_idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                expected_idx = self.smplx_joint_indices.get(joint_name)
                if expected_idx != smplx_idx:
                    raise ValueError(
                        f"Order mismatch at position {i}: joint '{joint_name}' has index {smplx_idx} "
                        f"in mapped_joint_indices, but smplx_joint_indices says it should be {expected_idx}."
                    )
            
            mapped_joints = human_joints[self.mapped_joint_indices]
            # mapped_joints shape: (num_mapped_joints, 3)
            # mapped_joints[i] corresponds to valid_joint_names[i] for all i
            
            # CRITICAL: Validate that we extracted the expected number of joints
            expected_num_joints = len(self.valid_joint_names)
            if len(mapped_joints) != expected_num_joints:
                raise ValueError(
                    f"Size mismatch: extracted {len(mapped_joints)} joints from human_joints, "
                    f"but expected {expected_num_joints} joints from valid_joint_names. "
                    f"This indicates an inconsistency in joint mapping."
                )
            
            # Debug: Print order and actual positions for first frame to verify matching
            import sys
            if not hasattr(sys, '_omni_human_joint_order_printed'):
                print(f"\n=== Human Joint Extraction Order ===")
                print(f"valid_joint_names: {self.valid_joint_names}")
                print(f"mapped_joint_indices: {self.mapped_joint_indices}")
                print(f"Extracted {len(mapped_joints)} joints in order: {self.valid_joint_names}")
                print(f"\n=== First Frame Joint Positions (to verify order) ===")
                for i, (name, idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                    pos = mapped_joints[i] if frame_idx == 0 else mapped_joints[i]
                    if frame_idx == 0:
                        print(f"  {i}: {name} (SMPLX idx {idx}) -> pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                # Specifically check knee vs ankle positions
                if frame_idx == 0:
                    l_knee_idx = self.valid_joint_names.index("L_Knee") if "L_Knee" in self.valid_joint_names else None
                    l_ankle_idx = self.valid_joint_names.index("L_Ankle") if "L_Ankle" in self.valid_joint_names else None
                    if l_knee_idx is not None and l_ankle_idx is not None:
                        l_knee_pos = mapped_joints[l_knee_idx]
                        l_ankle_pos = mapped_joints[l_ankle_idx]
                        print(f"\n=== Knee vs Ankle Position Check ===")
                        print(f"L_Knee (idx {l_knee_idx}, SMPLX {self.mapped_joint_indices[l_knee_idx]}): [{l_knee_pos[0]:.3f}, {l_knee_pos[1]:.3f}, {l_knee_pos[2]:.3f}]")
                        print(f"L_Ankle (idx {l_ankle_idx}, SMPLX {self.mapped_joint_indices[l_ankle_idx]}): [{l_ankle_pos[0]:.3f}, {l_ankle_pos[1]:.3f}, {l_ankle_pos[2]:.3f}]")
                        print(f"Knee should be HIGHER than ankle (knee.z > ankle.z): {l_knee_pos[2] > l_ankle_pos[2]}")
                sys._omni_human_joint_order_printed = True

            # Estimate target base orientation from human joints
            last_target_quat = None
            if frame_idx > 0 and len(retargeted_trajectory) > 0:
                # Get last orientation from previous frame result or previous target estimate
                # We use the result (wxyz) converted back to xyzw for continuity check
                # But wait, retargeted_trajectory has wxyz.
                # _estimate returns xyzw.
                # Better to store the previous estimated xyzw.
                pass
                
            # We need to maintain state of estimated quaternion to ensure continuity
            if not hasattr(self, '_last_estimated_quat'):
                self._last_estimated_quat = None
                
            target_quat_xyzw = self._estimate_base_orientation_from_joints(human_joints, self._last_estimated_quat)
            self._last_estimated_quat = target_quat_xyzw
            
            target_quat_wxyz = None
            if target_quat_xyzw is not None:
                # Convert xyzw to wxyz for MuJoCo
                target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0], 
                                            target_quat_xyzw[1], target_quat_xyzw[2]])

            # Retarget frame with previous frame as reference for smoothness
            q_last = retargeted_trajectory[-1] if len(retargeted_trajectory) > 0 else None
            q_opt = retargeter.retarget_frame(
                mapped_joints, q_init, q_last=q_last, 
                target_base_orientation=target_quat_wxyz
            )
            retargeted_trajectory.append(q_opt)

            # Update initial guess for next frame (smooth trajectory)
            q_init = q_opt

        return np.array(retargeted_trajectory)

    def _compile_collision_model(self, *, terrain_scale: float) -> mujoco.MjModel | None:
        """
        Compile a temporary MuJoCo model that includes the terrain mesh as a collidable 'ground' geom.

        The default robot MJCFs in holosoma assets include a plane ground. For climbing_scene we need
        collision distances against the *scene mesh* instead. We inject a mesh geom named with 'ground'
        into the worldbody and remove the default plane ground so that `_compute_penetration_constraints`
        can filter robot-vs-ground pairs reliably.
        """
        mjcf_path = Path(self.robot_mujoco_xml_path)
        if mjcf_path.suffix.lower() != ".xml" or not mjcf_path.exists():
            return None
        if not self.terrain_mesh_path.exists():
            return None

        xml_text = mjcf_path.read_text()

        # Make meshdir absolute so compiling from a temp file still finds robot mesh assets.
        robot_dir = mjcf_path.parent
        assets_dir = (robot_dir / "assets").resolve()

        def _replace_meshdir(match: re.Match[str]) -> str:
            value = match.group(1)
            if value.startswith("/") or value.startswith("\\"):
                return f'meshdir="{value}"'
            return f'meshdir="{assets_dir.as_posix()}/"'

        xml_text = re.sub(r'meshdir="([^"]*)"', _replace_meshdir, xml_text, count=1)

        # Keep the default ground plane if present (holosoma relies on it).
        #
        # For climbing_scene we *can* add the scene mesh as an additional collidable ground geom.
        # However, MuJoCo mesh collision/distance can be unreliable for concave meshes; when a collision
        # proxy is provided (e.g. voxel boxes), we prefer to rely on the proxy and skip mesh collision
        # entirely to avoid "dist==0" degenerate normals dominating the constraint set.

        terrain_mesh_abs = self.terrain_mesh_path.resolve().as_posix()
        terrain_mesh_name = "terrain_collision_mesh"
        terrain_geom_name = "ground_scene"
        scale_str = f"{float(terrain_scale):.8f} {float(terrain_scale):.8f} {float(terrain_scale):.8f}"

        inject_scene_mesh_collision = self.collision_proxy_boxes is None

        asset_inject = ""
        worldbody_inject = ""
        if inject_scene_mesh_collision:
            asset_inject = f'    <mesh name="{terrain_mesh_name}" file="{terrain_mesh_abs}" scale="{scale_str}"/>\n'
            worldbody_inject = (
                f'    <geom name="{terrain_geom_name}" type="mesh" mesh="{terrain_mesh_name}" '
                'contype="1" conaffinity="1" rgba="0 0 0 0"/>\n'
            )

        proxy_inject = ""
        if self.collision_proxy_boxes is not None:
            centers, half_sizes = self.collision_proxy_boxes
            centers = np.asarray(centers, dtype=float).reshape(-1, 3)
            half_sizes = np.asarray(half_sizes, dtype=float).reshape(-1, 3)
            if centers.shape[0] != half_sizes.shape[0]:
                raise ValueError(
                    f"collision_proxy_boxes centers ({centers.shape[0]}) and half_sizes ({half_sizes.shape[0]}) mismatch"
                )
            for i in range(centers.shape[0]):
                cx, cy, cz = centers[i].tolist()
                sx, sy, sz = half_sizes[i].tolist()
                proxy_inject += (
                    f'    <geom name="scene_proxy_box_{i:04d}" type="box" '
                    f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" size="{sx:.6f} {sy:.6f} {sz:.6f}" '
                    'contype="1" conaffinity="1" rgba="0 0 0 0"/>\n'
                )

        if "</asset>" in xml_text:
            xml_text = xml_text.replace("</asset>", asset_inject + "  </asset>", 1)
        else:
            return None

        if "<worldbody>" in xml_text:
            xml_text = xml_text.replace("<worldbody>", "<worldbody>\n" + worldbody_inject + proxy_inject, 1)
        else:
            return None

        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"omniretargeting_collision_{mjcf_path.stem}.xml"
        tmp_path.write_text(xml_text)

        try:
            return mujoco.MjModel.from_xml_path(str(tmp_path))
        except Exception as exc:
            print(f"Warning: failed to compile collision model MJCF {tmp_path}: {exc}")
            return None

    def _estimate_base_orientation_from_joints(self, joints: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray | None:
        """
        Estimate a base orientation from joint positions.

        Uses pelvis/hips/spine to build an approximate body frame:
        - up: pelvis -> spine1
        - right: left_hip -> right_hip
        - forward: right x up
        Returns quaternion in xyzw order.
        
        Args:
            joints: Joint positions array.
            last_quat: Quaternion from previous frame to ensure continuity (xyzw).
        """
        if joints.shape[0] < 4:
            return None

        pelvis = joints[0]
        left_hip = joints[1]
        right_hip = joints[2]
        spine1 = joints[3]

        up = spine1 - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm
        
        # Build forward using right-hand rule
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        # Re-orthogonalize right to ensure perfect orthogonality
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Rotation matrix with body axes in world coordinates.
        # Convention: X=forward, Y=right, Z=up
        rot = np.column_stack([forward, right, up])
        
        # Verify right-handed
        det = np.linalg.det(rot)
        if det < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])
        
        current_quat = Rotation.from_matrix(rot).as_quat() # xyzw
        
        # Ensure continuity if last quaternion is provided
        if last_quat is not None:
            # Check dot product to see if we need to flip sign
            dot = np.dot(current_quat, last_quat)
            if dot < 0:
                current_quat = -current_quat
                
        return current_quat

    def get_robot_dof(self) -> int:
        """Get the number of degrees of freedom of the robot."""
        return self.robot_model.nq - 7  # Subtract floating base DOF

    def get_joint_names(self) -> List[str]:
        """Get the names of all robot joints."""
        return [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)
                if self.robot_model.joint(i).name]

    def validate_joint_mapping(self) -> List[str]:
        """Validate that the joint mapping is compatible with the robot.
        
        Note: joint_mapping maps SMPLX joint names to robot BODY (link) names, not joint names.
        So we check for body names in the URDF.
        """
        # Get all body names from the robot model
        robot_bodies = set()
        for i in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                robot_bodies.add(body_name)
        
        # Check which mapped body names don't exist
        mapped_bodies = set(self.joint_mapping.values())
        missing_bodies = mapped_bodies - robot_bodies
        return list(missing_bodies)

    def _visualize_trajectory(self, trajectory: np.ndarray, scaled_terrain: trimesh.Trimesh):
        """
        Visualize the SMPLX trajectory using matplotlib 3D animation with terrain mesh.
        
        Args:
            trajectory: Processed trajectory of shape (T, J, 3) where T is frames, J is joints
                       Coordinates are assumed to be in +Z up convention (already transformed)
            scaled_terrain: Scaled terrain mesh
        """
        print(f"Visualizing trajectory with shape: {trajectory.shape}")
        print(f"Terrain mesh: {len(scaled_terrain.vertices)} vertices, {len(scaled_terrain.faces)} faces")
        print("Coordinate system: +Z is up")
        
        num_frames, num_joints, _ = trajectory.shape
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute axis limits based on trajectory AND terrain bounds
        all_points = trajectory.reshape(-1, 3)
        traj_bounds = np.array([all_points.min(axis=0), all_points.max(axis=0)])
        
        # Get terrain bounds
        terrain_bounds = scaled_terrain.bounds  # Shape: (2, 3) - min and max
        
        # Combine bounds
        x_min = min(traj_bounds[0, 0], terrain_bounds[0, 0])
        y_min = min(traj_bounds[0, 1], terrain_bounds[0, 1])
        z_min = min(traj_bounds[0, 2], terrain_bounds[0, 2])
        x_max = max(traj_bounds[1, 0], terrain_bounds[1, 0])
        y_max = max(traj_bounds[1, 1], terrain_bounds[1, 1])
        z_max = max(traj_bounds[1, 2], terrain_bounds[1, 2])
        
        # Add some margin
        margin = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        ax.set_xlim([x_min - margin * x_range, x_max + margin * x_range])
        ax.set_ylim([y_min - margin * y_range, y_max + margin * y_range])
        ax.set_zlim([z_min - margin * z_range, z_max + margin * z_range])
        
        # Set labels (Z is up)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m) - Up')
        ax.set_title('SMPLX Trajectory with Terrain Visualization')
        
        # Set equal aspect ratio
        max_range = max(x_range, y_range, z_range)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
        ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
        ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])
        
        # Plot terrain mesh
        print("Rendering terrain mesh...")
        terrain_vertices = scaled_terrain.vertices
        terrain_faces = scaled_terrain.faces
        
        # Create a simplified mesh for visualization if too complex
        if len(terrain_faces) > 10000:
            print(f"Terrain has {len(terrain_faces)} faces, simplifying for visualization...")
            simplified_terrain = scaled_terrain.simplify_quadric_decimation(10000)
            terrain_vertices = simplified_terrain.vertices
            terrain_faces = simplified_terrain.faces
            print(f"Simplified to {len(terrain_faces)} faces")
        
        # Plot terrain as a triangulated surface
        ax.plot_trisurf(
            terrain_vertices[:, 0], 
            terrain_vertices[:, 1], 
            terrain_vertices[:, 2],
            triangles=terrain_faces,
            color='gray',
            alpha=0.3,
            edgecolor='none',
            shade=True,
            linewidth=0
        )
        
        # Initialize scatter plot for joints
        scatter = ax.scatter([], [], [], c='blue', marker='o', s=50, alpha=0.9, edgecolors='black', linewidths=0.5)
        
        # Add text for frame counter
        frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        # Define SMPLX skeleton connections (approximate body structure)
        # These are indices in the standard SMPLX joint ordering
        skeleton_connections = [
            # Spine
            (0, 3),   # Pelvis -> Spine1
            (3, 6),   # Spine1 -> Spine2
            (6, 9),   # Spine2 -> Spine3
            (9, 12),  # Spine3 -> Neck
            (12, 15), # Neck -> Head
            
            # Left leg
            (0, 1),   # Pelvis -> L_Hip
            (1, 4),   # L_Hip -> L_Knee
            (4, 7),   # L_Knee -> L_Ankle
            (7, 10),  # L_Ankle -> L_Foot
            
            # Right leg
            (0, 2),   # Pelvis -> R_Hip
            (2, 5),   # R_Hip -> R_Knee
            (5, 8),   # R_Knee -> R_Ankle
            (8, 11),  # R_Ankle -> R_Foot
            
            # Left arm
            (9, 13),  # Spine3 -> L_Collar
            (13, 16), # L_Collar -> L_Shoulder
            (16, 18), # L_Shoulder -> L_Elbow
            (18, 20), # L_Elbow -> L_Wrist
            
            # Right arm
            (9, 14),  # Spine3 -> R_Collar
            (14, 17), # R_Collar -> R_Shoulder
            (17, 19), # R_Shoulder -> R_Elbow
            (19, 21), # R_Elbow -> R_Wrist
        ]
        
        # Filter connections to only include joints we have
        valid_connections = [(i, j) for i, j in skeleton_connections if i < num_joints and j < num_joints]
        
        # Initialize line objects for skeleton
        lines = []
        for _ in valid_connections:
            line, = ax.plot([], [], [], 'r-', linewidth=2.5, alpha=0.8)
            lines.append(line)
        
        # Set viewing angle for better visibility
        ax.view_init(elev=20, azim=45)
        
        def init():
            """Initialize animation."""
            scatter._offsets3d = ([], [], [])
            frame_text.set_text('')
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [scatter, frame_text] + lines
        
        def update(frame_idx):
            """Update animation for each frame."""
            # Get joint positions for current frame
            joints = trajectory[frame_idx]  # Shape: (J, 3)
            
            # Update scatter plot
            xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
            scatter._offsets3d = (xs, ys, zs)
            
            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx + 1}/{num_frames}')
            
            # Update skeleton lines
            for line, (i, j) in zip(lines, valid_connections):
                x_data = [joints[i, 0], joints[j, 0]]
                y_data = [joints[i, 1], joints[j, 1]]
                z_data = [joints[i, 2], joints[j, 2]]
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
            
            return [scatter, frame_text] + lines
        
        # Create animation
        print(f"Creating animation for {num_frames} frames...")
        anim = FuncAnimation(
            fig, 
            update, 
            frames=num_frames,
            init_func=init,
            interval=33,  # ~30 FPS
            blit=True,
            repeat=True
        )
        
        print("Displaying animation. Close the window to continue...")
        plt.tight_layout()
        plt.show()
        print("Visualization complete.")
