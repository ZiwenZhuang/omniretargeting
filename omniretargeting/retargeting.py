"""Core retargeting functionality adapted for generic robots and terrains."""

from __future__ import annotations

import numpy as np
import mujoco
import cvxpy as cp
from scipy import sparse as sp
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
import trimesh
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yourdfpy

from .utils import (
    load_terrain_mesh,
    sample_points_on_mesh,
    scale_mesh,
    compute_mesh_height_at_point,
    validate_smplx_trajectory,
    transform_points_local_to_world,
    get_adjacency_list,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
)


class GenericInteractionRetargeter:
    """
    Generic interaction mesh retargeter that works with any robot and terrain.

    This adapts the interaction mesh retargeting approach from holosoma_retargeting
    to work with generic URDF robots and terrain meshes.
    """

    def __init__(
        self,
        robot_model: mujoco.MjModel,
        robot_data: mujoco.MjData,
        terrain_mesh: trimesh.Trimesh,
        joint_mapping: Dict[str, str],
        robot_height: float,
        q_a_init_idx: int = -7,
        step_size: float = 0.2,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        collision_detection_threshold: float = 0.1,
        valid_joint_names: Optional[List[str]] = None,
        debug_frames: int = 5,
        print_joint_order: bool = True,
        enable_penetration_constraints: bool = True,
        max_penetration_constraints: int = 64,
        penetration_constraint_mode: str = "soft",
        penetration_slack_weight: float = 1e4,
    ):
        """Initialize the generic retargeter.
        
        Args:
            robot_model: MuJoCo model of the robot
            robot_data: MuJoCo data for the robot
            terrain_mesh: Terrain mesh (already scaled if needed)
            joint_mapping: Mapping from SMPLX joint names to robot link names
            robot_height: Height of the robot
            q_a_init_idx: Index where optimization variables start
            step_size: Trust region size for SQP
            penetration_tolerance: Tolerance for penetration constraints
            foot_sticking_tolerance: Tolerance for foot sticking
            collision_detection_threshold: Distance threshold for collision detection
            valid_joint_names: Ordered list of joint names to ensure consistent ordering
        """
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.terrain_mesh = terrain_mesh
        self.joint_mapping = joint_mapping  # This should already be filtered to valid joints only
        self.robot_height = robot_height
        
        # CRITICAL: Store ordered joint names to ensure consistent ordering
        # This ensures human_joints[i] matches robot_points[i] for all i
        if valid_joint_names is not None:
            self.valid_joint_names = valid_joint_names
            # Verify that valid_joint_names matches joint_mapping keys
            if set(self.valid_joint_names) != set(joint_mapping.keys()):
                raise ValueError(
                    f"valid_joint_names ({set(self.valid_joint_names)}) "
                    f"does not match joint_mapping keys ({set(joint_mapping.keys())})"
                )
        else:
            # Fallback: use dictionary insertion order (Python 3.7+)
            self.valid_joint_names = list(joint_mapping.keys())
        
        # Validate that all joints in mapping exist in robot
        # This is a final safety check - fail fast if joints are missing
        self._validate_joint_mapping()

        # Retargeting parameters
        self.q_a_init_idx = q_a_init_idx
        self.step_size = step_size
        self.penetration_tolerance = penetration_tolerance
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self.collision_detection_threshold = collision_detection_threshold
        self.debug_frames = max(0, int(debug_frames))
        self.print_joint_order = bool(print_joint_order)
        self.enable_penetration_constraints = bool(enable_penetration_constraints)
        self.max_penetration_constraints = max(0, int(max_penetration_constraints))
        self.penetration_constraint_mode = str(penetration_constraint_mode).lower().strip()
        if self.penetration_constraint_mode not in ("hard", "soft"):
            raise ValueError("penetration_constraint_mode must be 'hard' or 'soft'")
        self.penetration_slack_weight = float(penetration_slack_weight)
        self._frame_count = 0
        self._debug_this_frame = False
        self._joint_order_printed = False

        # Setup robot configuration
        self._setup_robot_config()

        # Setup terrain interaction
        self._setup_terrain_interaction()

    def _setup_robot_config(self):
        """Setup robot configuration parameters."""
        self.nq = self.robot_model.nq
        self.nv = self.robot_model.nv
        # Determine which qpos indices are optimized.
        # q_a_init_idx follows the original convention:
        #   -7: include floating base (0..nq)
        #    0: start at actuated joints (after floating base)
        #   12: start at waist, etc.
        # This assumes standard MuJoCo convention:
        # qpos structure: [floating_base (7), joint1 (1), joint2 (1), ...]
        start_idx = 7 + self.q_a_init_idx
        start_idx = int(np.clip(start_idx, 0, self.nq))
        self.q_a_indices = np.arange(start_idx, self.nq)
        self.nq_a = len(self.q_a_indices)
        
        print(f"Robot config: nq={self.nq}, nv={self.nv}, nq_a={self.nq_a}")
        print(f"q_a_indices range: {self.q_a_indices.min()} to {self.q_a_indices.max()}")

        # Joint limits
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]
        
        large_number = 1e6
        # Construct full limits array matching nq size
        # Start with floating base limits (unbounded)
        full_lower_limits = -large_number * np.ones(self.nq)
        full_upper_limits = large_number * np.ones(self.nq)
        
        # Fill in limits for actuated joints
        # This assumes joint addresses are contiguous after the base
        # Depending on the robot model, we might need to be more careful here
        # But for standard humanoids this usually holds
        
        # Typically self.robot_model.jnt_qposadr gives the index in qpos for each joint
        for i in range(self.robot_model.njnt):
            qpos_adr = self.robot_model.jnt_qposadr[i]
            if qpos_adr >= 7: # Skip root joint(s) if they are part of the base
                # For 1-DOF joints
                full_lower_limits[qpos_adr] = self.robot_model.jnt_range[i, 0]
                full_upper_limits[qpos_adr] = self.robot_model.jnt_range[i, 1]

        self.q_a_lb = full_lower_limits[self.q_a_indices]
        self.q_a_ub = full_upper_limits[self.q_a_indices]

        # Joint cost weights - small regularization to prevent extreme angles
        # Floating base (first 7 DOF) gets very small weight, joints get moderate weight
        self.Q_diag = np.ones(self.nq_a) * 1e-3  # Small default regularization (matching original)
        
        # Reduce weight for floating base to allow free movement
        base_indices_in_qa = []
        for base_idx in range(7):
            if base_idx in self.q_a_indices:
                idx_in_qa = np.where(self.q_a_indices == base_idx)[0]
                if len(idx_in_qa) > 0:
                    base_indices_in_qa.append(idx_in_qa[0])
        
        if len(base_indices_in_qa) > 0:
            self.Q_diag[base_indices_in_qa] = 0.001  # Very small weight for base
        
        # Store smoothness weight (matching original: 0.2)
        self.smooth_weight = 0.2
    
    def _validate_joint_mapping(self):
        """Validate that all joints in mapping exist in robot. Raise error if any are missing."""
        # Get all body names from robot
        robot_bodies = set()
        for i in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                robot_bodies.add(body_name)
        
        # Check all mapped bodies exist
        missing_bodies = []
        for joint_name, link_name in self.joint_mapping.items():
            if link_name not in robot_bodies:
                missing_bodies.append((joint_name, link_name))
        
        if missing_bodies:
            raise ValueError(
                f"The following robot links from joint_mapping were not found in URDF: {missing_bodies}. "
                f"Please check your joint_mapping. Available bodies: {sorted(list(robot_bodies))[:10]}..."
            )

    def _setup_terrain_interaction(self):
        """Setup terrain interaction parameters."""
        # Sample points on terrain for interaction mesh
        self.terrain_points = sample_points_on_mesh(self.terrain_mesh, 100)

        # Setup collision detection
        self.collision_pairs = self._setup_collision_detection()

    def _setup_collision_detection(self) -> List[Tuple[int, int]]:
        """Setup collision detection pairs for robot-terrain interaction."""
        # Get all robot geoms
        robot_geom_names = []
        for i in range(self.robot_model.ngeom):
            geom_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name:
                robot_geom_names.append((i, geom_name))

        # For now, create dummy collision pairs
        # TODO: Implement proper collision detection setup
        collision_pairs = []

        # Add foot-terrain collision pairs
        foot_keywords = ['foot', 'ankle', 'sole']
        for geom_id, geom_name in robot_geom_names:
            if any(keyword in geom_name.lower() for keyword in foot_keywords):
                # Create virtual terrain collision
                collision_pairs.append((geom_id, -1))  # -1 indicates terrain

        return collision_pairs

    def create_interaction_mesh(self, human_joints: np.ndarray, terrain_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create interaction mesh from human joints and terrain points.

        Args:
            human_joints: Human joint positions (N, 3)
            terrain_points: Terrain surface points (M, 3)

        Returns:
            Tuple of (vertices, tetrahedra)
        """
        # Combine human joints and terrain points
        vertices = np.vstack([human_joints, terrain_points])

        # Create Delaunay triangulation
        tri = Delaunay(vertices)

        return vertices, tri.simplices

    def retarget_frame(
        self,
        human_joints: np.ndarray,
        q_init: np.ndarray,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Retarget a single frame of human motion to robot motion.

        Args:
            human_joints: Human joint positions (N, 3)
            q_init: Initial robot configuration
            max_iter: Maximum optimization iterations
            q_last: Configuration at previous time step (for smoothness)

        Returns:
            Optimized robot configuration
        """
        # Note: self.terrain_points are sampled from self.terrain_mesh
        # In core.py, self.terrain_mesh is already the *scaled* terrain mesh passed to __init__
        # So we don't need to apply terrain_scale again here.
        # We use self.terrain_points directly.
        scaled_terrain_points = self.terrain_points 

        # Create interaction mesh
        vertices, tetrahedra = self.create_interaction_mesh(human_joints, scaled_terrain_points)

        # Create adjacency list
        adj_list = get_adjacency_list(tetrahedra, len(vertices))

        # Calculate target Laplacian coordinates
        # CRITICAL: Use uniform_weight=True to match the matrix computation in optimization
        # This ensures target_laplacian and lap0 use the same weighting scheme
        target_laplacian = calculate_laplacian_coordinates(vertices, adj_list, uniform_weight=True)

        # Perform optimization
        q_opt = self._optimize_configuration(
            q_init.copy(),
            target_laplacian,
            adj_list,
            scaled_terrain_points,
            max_iter=max_iter,
            q_last=q_last,
            target_base_orientation=target_base_orientation
        )

        return q_opt

    def _optimize_configuration(
        self,
        q_init: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: List[List[int]],
        terrain_points: np.ndarray,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimize robot configuration using SQP with interaction mesh constraints.

        Args:
            q_init: Initial configuration
            target_laplacian: Target Laplacian coordinates
            adj_list: Mesh adjacency list
            terrain_points: Terrain contact points
            max_iter: Maximum iterations
            q_last: Configuration at previous time step (for smoothness)

        Returns:
            Optimized configuration
        """
        q = q_init.copy()
        last_cost = np.inf

        frame_num = self._frame_count
        self._frame_count += 1

        show_debug = self.debug_frames > 0 and frame_num < self.debug_frames
        self._debug_this_frame = show_debug

        if show_debug:
            print(f"\n=== Frame {frame_num} Optimization ===")
            print(f"q_init[:10]: {q_init[:10]}")
            if q_last is not None:
                print(f"q_last[:10]: {q_last[:10]}")
                print(f"Has q_last: True")
            else:
                print(f"Has q_last: False")

        for iteration in range(max_iter):
            # Single optimization step
            q_new, cost = self._single_optimization_step(
                q, target_laplacian, adj_list, terrain_points, q_last, target_base_orientation
            )
            
            if show_debug and iteration < 3:
                print(f"  Iter {iteration}: cost={cost:.4f}, |dq|={np.linalg.norm(q_new - q):.6f}, status={'OK' if cost < np.inf else 'FAIL'}")

            # Check convergence (but don't stop early if we're still violating non-penetration).
            if abs(cost - last_cost) < 1e-6:
                has_violation = False
                if self.enable_penetration_constraints and np.isfinite(cost):
                    saved_debug = self._debug_this_frame
                    self._debug_this_frame = False
                    try:
                        ineqs_new = self._compute_penetration_inequalities(q_new)
                    finally:
                        self._debug_this_frame = saved_debug
                    has_violation = any((rhs > 0.0) for _J, rhs in ineqs_new)
                if not has_violation:
                    if show_debug:
                        print(f"  Converged at iteration {iteration}")
                    break

            q = q_new
            last_cost = cost
        
        if show_debug:
            print(f"Final cost: {last_cost:.4f}")
            print(f"Final q[:10]: {q[:10]}")
            print(f"Total change: {np.linalg.norm(q - q_init):.6f}")

        # Final feasibility cleanup for collision constraints.
        if self.enable_penetration_constraints and np.isfinite(last_cost):
            q = self._project_penetration_feasible(q, max_steps=3)

        return q

    def _project_penetration_feasible(self, q: np.ndarray, *, max_steps: int = 3) -> np.ndarray:
        """
        Post-SQP feasibility projection for non-penetration constraints.

        Even with hard constraints, SQP-style linearization + quaternion normalization can leave
        small residual violations. This projection solves a small conic program that minimally
        changes q (within the trust region and joint limits) while satisfying the current
        linearized non-penetration inequalities.
        """
        if not self.enable_penetration_constraints:
            return q

        q_proj = q.copy()

        for _ in range(int(max_steps)):
            # Compute inequalities at the current configuration.
            saved_debug = self._debug_this_frame
            self._debug_this_frame = False
            try:
                ineqs = self._compute_penetration_inequalities(q_proj)
            finally:
                self._debug_this_frame = saved_debug

            if not ineqs:
                break

            rhs_vals = np.array([rhs for _J, rhs in ineqs], dtype=float)
            if float(rhs_vals.max(initial=0.0)) <= 0.0:
                break

            dqa = cp.Variable(len(self.q_a_indices), name="dqa_proj")
            constraints: list[Any] = []

            q_a_current = q_proj[self.q_a_indices]
            constraints.extend(
                [
                    dqa >= (self.q_a_lb - q_a_current),
                    dqa <= (self.q_a_ub - q_a_current),
                ]
            )
            constraints.extend([J @ dqa >= float(rhs) for (J, rhs) in ineqs])
            constraints.append(cp.SOC(self.step_size, dqa))

            obj = cp.sum_squares(dqa)
            problem = cp.Problem(cp.Minimize(obj), constraints)
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or dqa.value is None:
                break

            q_new = q_proj.copy()
            q_new[self.q_a_indices] = q_a_current + dqa.value
            quat_new = q_new[3:7]
            q_new[3:7] = quat_new / (np.linalg.norm(quat_new) + 1e-12)
            q_proj = q_new

        return q_proj

    def _single_optimization_step(
        self,
        q: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: List[List[int]],
        terrain_points: np.ndarray,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Single SQP optimization step.

        Args:
            q: Current configuration
            target_laplacian: Target Laplacian coordinates
            adj_list: Mesh adjacency list
            terrain_points: Terrain contact points
            q_last: Configuration at previous time step (for smoothness)

        Returns:
            Tuple of (optimized_config, cost)
        """
        # Update robot state
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # Compute Jacobians for mapped joints
        J_V, p_V, _ = self._compute_robot_jacobians(q)

        # Create Laplacian matrices
        # CRITICAL: Ensure robot_points are in the SAME ORDER as human_joints passed to retarget_frame
        # The order MUST match: human_joints[i] corresponds to robot_points[i] for all i
        # 
        # Order flow:
        #   1. In core.py: valid_joint_names is built by iterating joint_mapping.keys() in order
        #   2. In core.py: mapped_joints = human_joints[mapped_joint_indices] where mapped_joint_indices
        #      corresponds to valid_joint_names[i] for each i
        #   3. In core.py: valid_joint_mapping preserves valid_joint_names order
        #   4. Here: self.joint_mapping IS valid_joint_mapping (passed from core.py)
        #   5. Here: robot_points built by iterating self.joint_mapping.keys() which matches valid_joint_names
        #   6. J_V stacked by iterating self.joint_mapping.keys() in the same order
        #
        # So: human_joints[i] (from valid_joint_names[i]) should match robot_points[i] (from joint_mapping.keys()[i])
        # CRITICAL: Use self.valid_joint_names to ensure consistent ordering
        joint_names_ordered = self.valid_joint_names
        
        # CRITICAL: Verify that p_V has all joints in the correct order
        if set(joint_names_ordered) != set(p_V.keys()):
            missing = set(joint_names_ordered) - set(p_V.keys())
            extra = set(p_V.keys()) - set(joint_names_ordered)
            raise RuntimeError(
                f"Joint mismatch: p_V has different joints than joint_mapping. "
                f"Missing from p_V: {missing}, Extra in p_V: {extra}"
            )
        
        robot_points = []
        for joint_name in joint_names_ordered:
            robot_points.append(p_V[joint_name])
        
        robot_points = np.array(robot_points)
        
        # Debug: Print order for first frame to verify
        if self.print_joint_order and not self._joint_order_printed:
            print(f"\n=== Joint Order Verification ===")
            print(f"Joint mapping order: {joint_names_ordered}")
            print(f"p_V keys order: {list(p_V.keys())}")
            print(f"First 3 robot points correspond to: {joint_names_ordered[:3]}")
            self._joint_order_printed = True
        
        # CRITICAL: Validate that sizes match exactly
        expected_num_joints = len(joint_names_ordered)
        if len(robot_points) != expected_num_joints:
            raise ValueError(
                f"Size mismatch: robot_points has {len(robot_points)} joints, "
                f"but expected {expected_num_joints} joints from joint_mapping.keys()."
            )
        if J_V.shape[0] != 3 * expected_num_joints:
            raise ValueError(
                f"Jacobian dimension mismatch: J_V has {J_V.shape[0]//3} joints, "
                f"but expected {expected_num_joints} joints from joint_mapping. "
                f"J_V shape: {J_V.shape}, expected rows: {3 * expected_num_joints}"
            )
        if len(robot_points) != J_V.shape[0] // 3:
            raise ValueError(
                f"Size mismatch between robot_points ({len(robot_points)}) and J_V ({J_V.shape[0]//3} joints)."
            )
        if len(robot_points) == 0:
            # Handle empty robot points to avoid vstack error
            if len(terrain_points) == 0:
                # Both empty, raise error as we can't do anything
                raise ValueError("Both robot_points and terrain_points are empty")
            vertices = terrain_points
            print("WARNING: No robot points found! Only using terrain points.")
        else:
            vertices = np.vstack([robot_points, terrain_points])

        # CRITICAL: Use uniform_weight=True to match target_laplacian computation
        # This ensures consistent Laplacian computation between target and current
        L = calculate_laplacian_matrix(vertices, adj_list, uniform_weight=True)
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        # Kron shape: (3*num_vertices, 3*num_vertices)
        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        
        # J_V shape: (3*num_joints, nq_a) - stacked Jacobians for each mapped joint
        # BUT Kron expects input vector of size 3*num_vertices (where vertices = mapped_joints + terrain_points)
        # J_V maps joint velocities (dqa) to velocities of mapped_joints (and 0 for terrain points)
        
        # We need to construct a full Jacobian J_full of shape (3*num_vertices, nq_a)
        # The top part corresponds to robot_points (mapped joints), bottom part (terrain) is zeros
        
        num_robot_points = len(robot_points)
        num_terrain_points = len(terrain_points)
        num_vertices = num_robot_points + num_terrain_points
        
        # Verify sizes match
        if J_V.shape[0] != 3 * num_robot_points:
             # This can happen if p_V has different number of points than J_V's stack
             # But they come from the same loop, so they should match
             print(f"Warning: J_V rows ({J_V.shape[0]}) != 3 * num_robot_points ({3*num_robot_points})")
        
        # Construct full Jacobian for all vertices
        # Top part: J_V (robot points), Bottom part: 0 (terrain points, static)
        J_full_vertices = sp.vstack([
            sp.csr_matrix(J_V),  # Jacobians for robot points
            sp.csr_matrix((3 * num_terrain_points, self.nq_a)) # Zeros for terrain points
        ])
        
        # Now J_L = Kron @ J_full_vertices
        # Kron: (3*V, 3*V)
        # J_full_vertices: (3*V, nq_a)
        # Result J_L: (3*V, nq_a)
        J_L = Kron @ J_full_vertices

        # Setup optimization problem
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * len(vertices), name="laplacian")

        # Constraints
        constraints = []
        
        # CRITICAL: Linear equality constraint matching original implementation
        # This defines: lap_var = lap0_vec + J_L @ dqa
        # Original uses: J_L[:, self.q_a_indices] @ dqa - lap_var == -lap0_vec
        # Rearranged: lap_var == lap0_vec + J_L @ dqa
        lap0_vec = (L @ vertices).reshape(-1)
        target_lap_vec = target_laplacian.reshape(-1)
        
        # Note: J_L already has columns only for q_a_indices (from J_V construction)
        # But original slices again, so we match that exactly
        constraints.append(cp.Constant(J_L) @ dqa - lap_var == -lap0_vec)

        # Joint limits
        q_a_current = q[self.q_a_indices]
        constraints.extend([
            dqa >= (self.q_a_lb - q_a_current),
            dqa <= (self.q_a_ub - q_a_current),
        ])

        # Terrain penetration constraints (MuJoCo distance + contact Jacobian linearization).
        penetration_slack = None
        penetration_ineqs: list[tuple[np.ndarray, float]] = []
        if self.enable_penetration_constraints:
            penetration_ineqs = self._compute_penetration_inequalities(q)
            if penetration_ineqs:
                if self.penetration_constraint_mode == "hard":
                    constraints.extend([J @ dqa >= rhs for (J, rhs) in penetration_ineqs])
                else:
                    penetration_slack = cp.Variable(len(penetration_ineqs), nonneg=True, name="penetration_slack")
                    for i, (J, rhs) in enumerate(penetration_ineqs):
                        constraints.append(J @ dqa + penetration_slack[i] >= rhs)

        # Trust region
        constraints.append(cp.SOC(self.step_size, dqa))

        # Objective - matching original implementation exactly
        weights = np.ones(len(vertices)) * 10  # Laplacian weights (matching original laplacian_weights = 10)
        sqrt_w3 = np.sqrt(np.repeat(weights, 3))
        
        # Minimize: ||lap_var - target_lap_vec||^2
        # where lap_var = lap0_vec + J_L @ dqa (from constraint)
        obj = cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec))
        
        # Joint regularization cost (keep joints near zero/neutral)
        # Matching original: Q_diag cost uses q_a_n_last (last optimized at current time step)
        # In our case, q_a_current = q[self.q_a_indices] which is the current guess
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        
        # Modify Q_diag for specific joints (matching original MANUAL_COST logic)
        Q_diag_modified = Qd.copy()
        for i in range(self.robot_model.njnt):
            joint_name = self.robot_model.joint(i).name
            if joint_name:
                joint_name_lower = joint_name.lower()
                # Strong regularization for joints prone to 180° flips
                if ('waist' in joint_name_lower or 'torso' in joint_name_lower or 
                    ('hip' in joint_name_lower and 'yaw' in joint_name_lower)):
                    qpos_adr = self.robot_model.jnt_qposadr[i]
                    if qpos_adr in self.q_a_indices:
                        idx_in_qa = np.where(self.q_a_indices == qpos_adr)[0]
                        if len(idx_in_qa) > 0:
                            Q_diag_modified[idx_in_qa[0]] = 0.2  # Strong regularization (matching original MANUAL_COST)
        
        # Q_diag cost: ||sqrt(Q_diag) * (dqa + q_a_current)||^2
        # This matches original: cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last))
        obj += cp.sum_squares(cp.multiply(np.sqrt(Q_diag_modified), dqa + q_a_current))

        # Soft non-penetration: penalize slack if enabled.
        if penetration_slack is not None:
            obj += float(self.penetration_slack_weight) * cp.sum_squares(penetration_slack)

        # Smoothness cost (matching original implementation exactly)
        # CRITICAL FIX: Use previous frame's velocity, not current guess
        if q_last is not None:
            q_a_current = q[self.q_a_indices]  # Current guess at this SQP iteration
            dqa_smooth = q_last[self.q_a_indices] - q_a_current  # Velocity from prev frame to current guess
            obj += self.smooth_weight * cp.sum_squares(dqa - dqa_smooth)
        
        # Base orientation tracking cost
        # Keep the base orientation close to the target (estimated from human joints)
        if target_base_orientation is not None and 3 in self.q_a_indices:
            # Find quaternion indices in q_a_indices
            quat_indices_in_qa = []
            for quat_idx in [3, 4, 5, 6]:  # wxyz quaternion
                if quat_idx in self.q_a_indices:
                    idx_in_qa = np.where(self.q_a_indices == quat_idx)[0]
                    if len(idx_in_qa) > 0:
                        quat_indices_in_qa.append(idx_in_qa[0])
            
            if len(quat_indices_in_qa) == 4:
                # Target quaternion in wxyz (MuJoCo convention)
                quat_target_wxyz = target_base_orientation
                quat_current = q[3:7]  # Current quaternion
                
                # Add cost to keep quaternion close to target
                orientation_weight = 5.0  # Strong preference to maintain orientation
                for i, qa_idx in enumerate(quat_indices_in_qa):
                    target_val = quat_target_wxyz[i]
                    current_val = quat_current[i]
                    # Penalize deviation: (q_new - target)^2 = (q_current + dqa - target)^2
                    obj += orientation_weight * cp.square(dqa[qa_idx] + current_val - target_val)

        # Solve
        problem = cp.Problem(cp.Minimize(obj), constraints)

        show_solver_debug = bool(self._debug_this_frame)
        
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                if show_solver_debug:
                    print(f"    First solve failed: {problem.status}, trying without SOC...")
                # Fallback to simpler problem without trust region
                constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
                problem = cp.Problem(cp.Minimize(obj), constraints)
                problem.solve(solver=cp.CLARABEL, verbose=False)
                if show_solver_debug:
                    print(f"    Second solve status: {problem.status}")

            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                dqa_opt = dqa.value
                cost = problem.value

                q_opt = q.copy()
                q_opt[self.q_a_indices] = dqa_opt + q_a_current
                
                # CRITICAL FIX: Normalize quaternion with sign continuity to prevent frame-to-frame jumps
                quat_new = q_opt[3:7]
                quat_new = quat_new / (np.linalg.norm(quat_new) + 1e-12)
                
                # Ensure quaternion sign continuity with previous frame (if available)
                if q_last is not None:
                    quat_prev = q_last[3:7]
                    # If dot product is negative, quaternions are in opposite hemispheres
                    # Flip sign to ensure continuity
                    if np.dot(quat_new, quat_prev) < 0:
                        quat_new = -quat_new
                
                q_opt[3:7] = quat_new

                return q_opt, cost
            else:
                if show_solver_debug:
                    print(f"    SOLVER FAILED: {problem.status}")
                return q, np.inf

        except Exception as e:
            if show_solver_debug:
                print(f"    EXCEPTION: {e}")
            return q, np.inf

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        if self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = self.robot_model.jnt_qposadr[j0]  # 0
            dadr = self.robot_model.jnt_dofadr[j0]  # 0

            # Linear block: v_lin = xyz_dot
            T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

            # Angular block: ω_* = 2 * E_*(q) * quat_dot
            w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

            def get_e_world(qw, qx, qy, qz):
                return np.array(
                    [
                        [-qx, qw, qz, -qy],
                        [-qy, -qz, qw, qx],
                        [-qz, qy, -qx, qw],
                    ]
                )

            def get_e_body(qw, qx, qy, qz):
                return np.array(
                    [
                        [-qx, qw, -qz, qy],
                        [-qy, qz, qw, -qx],
                        [-qz, -qy, qx, qw],
                    ]
                )

            E_fn = get_e_world if use_world_omega else get_e_body
            E1 = 2.0 * E_fn(w, x, y, z)
            
            # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
            # T[dadr + 0 : dadr + 3, qadr + 0 : qadr + 3] = np.eye(3) # Already set
            T[dadr + 3 : dadr + 6, qadr + 3 : qadr + 7] = E1  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1 if self.robot_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray = None, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """
        if p_body is None:
            p_body = np.zeros(3)
            
        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        # mujoco.mj_forward(self.robot_model, self.robot_data) # Assumed called before

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _compute_robot_jacobians(self, q: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], None]:
        """Compute Jacobians for robot joints in world frame.
        
        Args:
            q: Robot configuration
            
        Returns:
            Tuple of (J_V, p_dict, None):
                - J_V: Stacked Jacobians (3*num_joints, nq_a)
                - p_dict: Dictionary of positions for each joint
                - None: Placeholder for compatibility
        """
        J_dict = {}
        p_dict = {}

        for joint_name, link_name in self.joint_mapping.items():
            try:
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

                # Get position in world frame
                pos = self.robot_data.xpos[body_id].copy()

                # Compute Jacobian in world frame
                J_full = self._calc_contact_jacobian_from_point(body_id)

                # Extract optimized part (J_full is already in qpos coordinates)
                valid_indices = self.q_a_indices[self.q_a_indices < J_full.shape[1]]
                if len(valid_indices) < len(self.q_a_indices):
                    print(
                        f"Warning: Truncating indices for joint {joint_name}. "
                        f"J width: {J_full.shape[1]}, Max idx: {self.q_a_indices.max()}"
                    )

                J_reduced = J_full[:, valid_indices]
                
                # Pad if needed
                if J_reduced.shape[1] < self.nq_a:
                    J_pad = np.zeros((3, self.nq_a))
                    J_pad[:, :J_reduced.shape[1]] = J_reduced
                    J_reduced = J_pad
                    
                J_dict[joint_name] = J_reduced
                p_dict[joint_name] = pos

            except Exception as e:
                # CRITICAL: All joints should exist (validated in __init__), so this is unexpected
                # Raise error instead of skipping to ensure size consistency
                raise RuntimeError(
                    f"Failed to compute Jacobian for joint '{joint_name}' -> link '{link_name}'. "
                    f"This should not happen if joint_mapping was validated. Error: {e}"
                ) from e

        # Stack Jacobians in the SAME ORDER as valid_joint_names to match human_joints order
        # This is critical for correct Laplacian matching!
        # CRITICAL: Use self.valid_joint_names to ensure consistent ordering
        joint_names_ordered = self.valid_joint_names
        num_joints = len(joint_names_ordered)
        
        if num_joints > 0:
            J_V = np.zeros((3 * num_joints, self.nq_a))
            for i, joint_name in enumerate(joint_names_ordered):
                if joint_name in J_dict:
                    J = J_dict[joint_name]
                    # Ensure J has the correct shape (3, nq_a)
                    if J.shape != (3, self.nq_a):
                        if J.shape[1] > self.nq_a:
                            J = J[:, :self.nq_a]
                        elif J.shape[1] < self.nq_a:
                            J_pad = np.zeros((3, self.nq_a))
                            J_pad[:, :J.shape[1]] = J
                            J = J_pad
                    J_V[3 * i:3 * (i + 1), :] = J
                else:
                    # CRITICAL: All joints should exist (validated in __init__), so this is unexpected
                    raise RuntimeError(
                        f"Jacobian for joint '{joint_name}' not found in J_dict. "
                        f"This should not happen if joint_mapping was validated. "
                        f"Available joints in J_dict: {list(J_dict.keys())}"
                    )
        else:
            J_V = np.zeros((0, self.nq_a))

        return J_V, p_dict, None

    def _prefilter_pairs_with_mj_collision(self, threshold: float) -> set:
        """
        Use MuJoCo collision detection to find candidate geometry pairs.
        
        Args:
            threshold: Distance threshold for collision detection
            
        Returns:
            Set of (geom1_id, geom2_id) tuples for candidate collision pairs
        """
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        # Cache geometry names
        if not hasattr(self, '_geom_names'):
            self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        # Save original margins
        if not hasattr(self, '_saved_margins'):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        # Temporarily set margins to threshold
        m.geom_margin[:] = threshold

        # Run collision detection
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore original margins
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _compute_jacobian_for_contact_relative(
        self, 
        geom1_id: int, 
        geom2_id: int, 
        geom1_name: str,
        geom2_name: str,
        fromto: np.ndarray, 
        dist: float,
        contact_normal_world_ba: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute relative contact Jacobian for a geometry pair.
        
        Args:
            geom1_id: First geometry ID
            geom2_id: Second geometry ID
            geom1_name: First geometry name
            geom2_name: Second geometry name
            fromto: Contact points [pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z]
            dist: Signed distance between geometries
            
        Returns:
            Contact Jacobian (1D array of length nq)
        """
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            # IMPORTANT:
            # - For some geom pairs (notably mesh), MuJoCo's `mj_geomDistance` may clamp penetration to dist==0,
            #   i.e. it never returns negative signed distance.
            # - Using np.sign(dist) would then produce a zero normal at dist==0, disabling constraints entirely.
            # We therefore treat dist>=0 as the same "side" (s=+1), and only flip when dist<0.
            s = -1.0 if dist < 0.0 else 1.0
            nhat_BA_W = s * (v / norm_v)
        # Degenerate case: points coincide
        elif contact_normal_world_ba is not None and np.linalg.norm(contact_normal_world_ba) > 1e-12:
            nhat_BA_W = np.asarray(contact_normal_world_ba, dtype=float).reshape(3)
        elif any(token in (geom2_name or "").lower() for token in ("ground", "scene", "terrain")):
            s = -1.0 if dist < 0.0 else 1.0
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * s
        elif any(token in (geom1_name or "").lower() for token in ("ground", "scene", "terrain")):
            s = -1.0 if dist < 0.0 else 1.0
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * s
        else:
            # Last resort: fall back to center-to-center direction.
            v_center = self.robot_data.geom_xpos[geom1_id] - self.robot_data.geom_xpos[geom2_id]
            norm_center = float(np.linalg.norm(v_center))
            if norm_center > 1e-12:
                nhat_BA_W = v_center / norm_center
            else:
                nhat_BA_W = np.array([0.0, 0.0, 0.0])

        # Get body IDs for the geometries
        body1_id = self.robot_model.geom_bodyid[geom1_id]
        body2_id = self.robot_model.geom_bodyid[geom2_id]

        # Compute Jacobians for both contact points (in world frame)
        J_bodyA = self._calc_contact_jacobian_from_point(body1_id, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(body2_id, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        # Project onto contact normal
        return nhat_BA_W @ Jc

    def _prefilter_contact_normals_with_mj_collision(self, threshold: float) -> dict[tuple[int, int], np.ndarray]:
        """
        Use MuJoCo collision to collect candidate pair normals.

        Returns mapping (min_geom_id, max_geom_id) -> normal_world(min->max).

        Notes:
        - This is used as a robustness fallback when `mj_geomDistance` returns degenerate closest points
          (e.g. dist==0 and fromto points coincide).
        - The normal direction follows MuJoCo's contact convention: it points from `contact.geom1` to
          `contact.geom2`. We canonicalize the key to (min,max) and flip the normal accordingly.
        """
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin
        m.geom_margin[:] = float(threshold)

        mujoco.mj_collision(m, d)

        normals: dict[tuple[int, int], np.ndarray] = {}
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0 or g1 >= ngeom or g2 >= ngeom:
                continue
            n12 = np.asarray(c.frame[:3], dtype=float).reshape(3)
            if np.linalg.norm(n12) < 1e-12:
                continue
            a, b = (g1, g2) if g1 < g2 else (g2, g1)
            n_ab = n12 if (a == g1 and b == g2) else -n12
            # Keep the first one; contacts can be many, but any normal is better than none for degeneracy.
            normals.setdefault((a, b), n_ab)

        m.geom_margin[:] = self._saved_margins
        return normals

    def _compute_penetration_inequalities(self, q: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """
        Compute linearized non-penetration inequalities using MuJoCo collision detection.

        Returns a list of (J_actuated, rhs) such that:
            J_actuated @ dqa >= rhs
        where rhs is defined so that (in the local linearization) enforcing the inequality helps maintain
        a minimum separation of `penetration_tolerance` along the contact normal direction.
        """
        inequalities: list[tuple[np.ndarray, float]] = []

        # Update robot state (should already be done, but ensure it's current)
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # Direct robot-vs-ground scan.
        # We intentionally do NOT rely on `mj_collision` prefilter because mesh interactions can be missed
        # (and `mj_geomDistance` for mesh can clamp penetrations to dist==0). Scanning all robot geoms
        # against a small set of ground/scene geoms is cheap (O(ngeom)).
        if not hasattr(self, "_geom_names") or len(getattr(self, "_geom_names", [])) != m.ngeom:
            self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]

        contype, conaff = m.geom_contype, m.geom_conaffinity
        fromto = np.zeros(6, dtype=float)

        def _is_ground(name: str) -> bool:
            name_lower = (name or "").lower()
            return ("ground" in name_lower) or ("scene" in name_lower) or ("terrain" in name_lower)

        contact_normals = self._prefilter_contact_normals_with_mj_collision(threshold)

        ground_geom_ids: list[int] = []
        robot_geom_ids: list[int] = []
        for gid, name in enumerate(self._geom_names):
            if contype[gid] == 0 and conaff[gid] == 0:
                continue
            if _is_ground(name):
                ground_geom_ids.append(gid)
            else:
                robot_geom_ids.append(gid)

        checked = len(ground_geom_ids) * len(robot_geom_ids)
        use_prefilter = checked > 4000

        scored: list[tuple[float, int, int, np.ndarray]] = []

        if use_prefilter:
            # Holosoma-style prefilter: use mj_collision broadphase with temporary margins.
            candidates = self._prefilter_pairs_with_mj_collision(threshold)
            for g1, g2 in candidates:
                n1 = self._geom_names[g1]
                n2 = self._geom_names[g2]
                g1_ground = _is_ground(n1)
                g2_ground = _is_ground(n2)
                if g1_ground == g2_ground:
                    continue
                # Ensure order is (robot, ground).
                robot_id, ground_id = (g1, g2) if (not g1_ground) else (g2, g1)

                fromto[:] = 0.0
                dist = float(mujoco.mj_geomDistance(m, d, robot_id, ground_id, threshold, fromto))
                if dist < (threshold - 1e-9):
                    scored.append((dist, robot_id, ground_id, fromto.copy()))
        else:
            # Direct scan: cheap when number of grounds is small.
            for ground_id in ground_geom_ids:
                for robot_id in robot_geom_ids:
                    fromto[:] = 0.0
                    dist = float(mujoco.mj_geomDistance(m, d, robot_id, ground_id, threshold, fromto))
                    # IMPORTANT: `mj_geomDistance(..., distmax=threshold, ...)` can return exactly `threshold`
                    # as a sentinel for "distance >= threshold". Only keep strictly-below-threshold pairs.
                    if dist < (threshold - 1e-9):
                        scored.append((dist, robot_id, ground_id, fromto.copy()))

        # Prioritize closest pairs.
        scored.sort(key=lambda item: item[0])
        if self.max_penetration_constraints > 0:
            scored = scored[: self.max_penetration_constraints]

        if self._debug_this_frame:
            if scored:
                min_dist = float(scored[0][0])
                max_dist = float(scored[-1][0])
            else:
                min_dist = float("nan")
                max_dist = float("nan")
            print(
                f"[collision] grounds={len(ground_geom_ids)} robots={len(robot_geom_ids)} checked={checked} kept={len(scored)} "
                f"dist=[{min_dist:.4f},{max_dist:.4f}] thr={threshold:.4f} tol={float(self.penetration_tolerance):.4e} "
                f"mode={self.penetration_constraint_mode}"
            )

        for dist, g1, g2, fromto_i in scored:
            contact_normal_ba = None
            key = (min(g1, g2), max(g1, g2))
            if key in contact_normals:
                # contact_normals stores normal from min->max. We need normal from geom2->geom1.
                n_min_to_max = contact_normals[key]
                if g2 == key[0] and g1 == key[1]:
                    contact_normal_ba = n_min_to_max
                elif g2 == key[1] and g1 == key[0]:
                    contact_normal_ba = -n_min_to_max

            J_rel = self._compute_jacobian_for_contact_relative(
                g1,
                g2,
                self._geom_names[g1],
                self._geom_names[g2],
                fromto_i,
                dist,
                contact_normal_world_ba=contact_normal_ba,
            )

            # Extract optimized part (q_a_indices are qpos indices, J_rel is length nq).
            valid_indices = self.q_a_indices[self.q_a_indices < J_rel.shape[0]]
            J_rel_actuated = J_rel[valid_indices]

            if len(J_rel_actuated) < self.nq_a:
                J_pad = np.zeros(self.nq_a)
                J_pad[: len(J_rel_actuated)] = J_rel_actuated
                J_rel_actuated = J_pad

            # Enforce a minimum separation.
            # For mesh geoms, MuJoCo's mj_geomDistance may clamp penetrations to dist==0 (never negative),
            # so holosoma's rhs=(-dist - tol) would never activate. We instead enforce dist >= tol.
            rhs = float(self.penetration_tolerance) - float(dist)
            inequalities.append((J_rel_actuated, rhs))

        if self._debug_this_frame and scored:
            if inequalities:
                rhs_vals = np.array([rhs for _J, rhs in inequalities], dtype=float)
                n_viol = int(np.sum(rhs_vals > 0.0))
                print(
                    f"[collision] ineqs={len(inequalities)} violating={n_viol} rhs=[{rhs_vals.min():.6f},{rhs_vals.max():.6f}]"
                )
            else:
                print("[collision] ineqs=0")

        return inequalities


def retarget_smplx_to_robot(
    smplx_trajectory: np.ndarray,
    robot_urdf_path: Path,
    terrain_mesh_path: Path,
    joint_mapping: Dict[str, str],
    robot_height: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    High-level function to retarget SMPLX trajectory to any robot on any terrain.

    Args:
        smplx_trajectory: SMPLX joint positions (T, J, 3)
        robot_urdf_path: Path to robot URDF
        terrain_mesh_path: Path to terrain mesh
        joint_mapping: Mapping from SMPLX joints to robot links
        robot_height: Robot height override

    Returns:
        Tuple of (terrain_scale, retargeted_trajectory)
    """
    # Validate inputs
    if not validate_smplx_trajectory(smplx_trajectory):
        raise ValueError("Invalid SMPLX trajectory format")

    # Load robot
    _ = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True)

    robot_model_path = robot_urdf_path
    if robot_urdf_path.suffix.lower() == ".urdf":
        adjacent_xml = robot_urdf_path.with_suffix(".xml")
        if adjacent_xml.exists():
            robot_model_path = adjacent_xml
    robot_model = mujoco.MjModel.from_xml_path(str(robot_model_path))
    robot_data = mujoco.MjData(robot_model)

    # Detect robot height if not provided
    if robot_height is None:
        robot_height = 1.6  # Default humanoid height

    # Load terrain
    terrain_mesh = load_terrain_mesh(terrain_mesh_path)

    # Compute terrain scaling
    from .core import OmniRetargeter
    temp_retargeter = OmniRetargeter(
        robot_urdf_path=robot_urdf_path,
        robot_mujoco_xml_path=robot_model_path if robot_model_path.suffix.lower() == ".xml" else None,
        terrain_mesh_path=terrain_mesh_path,
        joint_mapping=joint_mapping,
        robot_height=robot_height
    )
    terrain_scale = temp_retargeter._compute_terrain_scale(smplx_trajectory)

    # Scale terrain
    scaled_terrain = scale_mesh(terrain_mesh, terrain_scale)

    # Initialize retargeter with scaled terrain
    retargeter = GenericInteractionRetargeter(
        robot_model, robot_data, scaled_terrain, joint_mapping, robot_height
    )

    # Retarget each frame
    retargeted_trajectory = []
    q_init = np.zeros(robot_model.nq)
    q_init[3:7] = [1, 0, 0, 0]  # Identity quaternion
    q_init[6] = robot_height * 0.5  # Initial height
    q_last = None

    for frame_idx, human_joints in enumerate(smplx_trajectory):
        # Extract mapped joints
        mapped_indices = []  # TODO: Map joint names to indices
        if len(mapped_indices) == 0:
            # Fallback: use first few joints
            mapped_joints = human_joints[:len(joint_mapping)]
        else:
            mapped_joints = human_joints[mapped_indices]

        # Retarget frame
        q_opt = retargeter.retarget_frame(mapped_joints, q_init, terrain_scale, q_last=q_last)
        retargeted_trajectory.append(q_opt)

        # Update initial guess for next frame
        q_last = q_opt.copy()
        q_init = q_opt

    return terrain_scale, np.array(retargeted_trajectory)
