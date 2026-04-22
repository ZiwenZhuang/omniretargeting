# OmniRetargeting

**Generic motion retargeting for any humanoid URDF and terrain mesh.**

This is a re-implementation of the [OmniRetarget](https://arxiv.org/abs/2509.26633) method. OmniRetargeting is a flexible motion retargeting system that can convert human motion trajectories (in SMPLX format) to any humanoid robot operating on any terrain mesh. Unlike specialized retargeting systems, OmniRetargeting automatically adapts to different robot morphologies and terrain types.

## Installation

```bash
pip install omniretargeting
```

Or install from source:

```bash
git clone <repository-url>
cd omniretargeting
pip install -e .
```

For development with testing:

```bash
pip install -e ".[dev,test]"
```

## Quick Start

```python
from omniretargeting import OmniRetargeter
from omniretargeting.utils import load_smplx_trajectory
from pathlib import Path

# Define your inputs
robot_urdf = "path/to/your/robot.urdf"
terrain_mesh = "path/to/your/terrain.obj"

# Load SMPLX trajectory (positions and orientations)
# For raw SMPLX-NG files (e.g., stageii.npz with poses/trans/betas)
smplx_trajectory, smplx_orientations = load_smplx_trajectory(
    Path("path/to/trajectory_stageii.npz"),
    smplx_model_directory="path/to/smplx/models",  # Required for raw files and orientation computation
    gender="neutral",
)

# OR for pre-processed files (.npy or .npz with global_joint_positions)
# smplx_trajectory, smplx_orientations = load_smplx_trajectory(
#     Path("path/to/trajectory.npy")
# )

print(f"Loaded trajectory shape: {smplx_trajectory.shape}")  # (T, J, 3)
if smplx_orientations is not None:
    print(f"Loaded orientations shape: {smplx_orientations.shape}")  # (T, J, 4) - quaternions (wxyz)
else:
    print("Orientations not available for this file format")

# Define joint mapping from SMPLX joint names to robot BODY (link) names.
# Keys must be SMPLX joint names in the capitalized convention used by
# omniretargeting (see the full list under "Joint Mapping" below).
joint_mapping = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip_roll_link",
    "R_Hip": "right_hip_roll_link",
    "L_Knee": "left_knee_link",
    "R_Knee": "right_knee_link",
    "L_Ankle": "left_ankle_roll_link",
    "R_Ankle": "right_ankle_roll_link",
    # ... add more mappings as needed
}

# Create retargeter
retargeter = OmniRetargeter(
    robot_urdf_path=robot_urdf,
    terrain_mesh_path=terrain_mesh,
    joint_mapping=joint_mapping,
)

# Perform retargeting
terrain_scale, retargeted_motion = retargeter.retarget_motion(
    smplx_trajectory,
    framerate=30.0,
    enable_terrain_scaling=True,
    visualize_trajectory=False,
)

print(f"Terrain scale factor: {terrain_scale}")
print(f"Retargeted motion shape: {retargeted_motion.shape}")  # (T, 7 + DOF)
```

For a ready-to-run setup, omniretargeting ships with a Unitree G1 profile
(`robot_models/unitree_g1/unitree_g1.json`) that already contains a full joint
mapping, height/orientation helpers, and foot-stabilization settings. Use it
via the CLI (see below) or by loading the profile directly:

```python
from omniretargeting import OmniRetargeter, load_robot_config

cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")
retargeter = OmniRetargeter(
    robot_urdf_path=cfg["urdf_path"],
    terrain_mesh_path="path/to/terrain.obj",
    joint_mapping=cfg["joint_mapping"],
    robot_height=cfg.get("robot_height"),
    smplx_joint_names=cfg.get("smplx_joint_names"),
    height_estimation=cfg.get("height_estimation"),
    base_orientation=cfg.get("base_orientation"),
    retargeting=cfg.get("retargeting"),
)
```

## Input Format

### SMPLX Trajectory

The trajectory must be provided as a numpy array of shape `(T, J, 3)` where:
- **T**: Number of frames
- **J**: Number of joints (typically 22 for body joints)
- **3**: (x, y, z) coordinates in world frame

#### Loading Trajectory Files

OmniRetargeting supports multiple trajectory file formats through the `load_smplx_trajectory` utility:

**1. Pre-processed files (.npy)**:
```python
from omniretargeting.utils import load_smplx_trajectory
# Returns (positions, None) since orientations cannot be computed from positions alone
trajectory, orientations = load_smplx_trajectory(Path("trajectory.npy"))
# orientations will be None
```

**2. Pre-processed files (.npz with 'global_joint_positions')**:
```python
# Returns (positions, orientations) if full_pose data is available
trajectory, orientations = load_smplx_trajectory(
    Path("trajectory.npz"),
    smplx_model_directory="/path/to/smplx/models",  # Needed for parent structure
)
# Looks for 'global_joint_positions' key for positions
# Looks for 'full_pose' and 'root_orient' keys for orientations
```

**3. Raw SMPLX-NG files (stageii.npz)**:

Raw SMPLX-NG files contain SMPLX parameters, not joint positions. Keys include:
- `'gender'`, `'surface_model_type'`, `'mocap_frame_rate'`
- `'trans'`, `'poses'`, `'betas'`
- `'root_orient'`, `'pose_body'`, `'pose_hand'`, `'pose_jaw'`, `'pose_eye'`

To load these, you **must** provide the SMPLX model path:

```python
trajectory, orientations = load_smplx_trajectory(
    Path("HumanEva_S3_Jog_1_stageii.npz"),
    smplx_model_directory="/path/to/smplx/models",  # Required!
    gender="neutral",  # Auto-detected from file if present
)
```

The function **always returns a tuple** `(positions, orientations)`:
1. Load SMPLX parameters from the file
2. Run forward kinematics using the SMPLX model
3. Return joint positions in shape `(T, 22, 3)` and orientations in shape `(T, 22, 4)`

**Orientation Format:**
- Orientations are returned as quaternions in **wxyz** format (scalar-first)
- Shape: `(T, J, 4)` where the last dimension is `[w, x, y, z]`
- Represents the world-frame orientation of each joint
- Will be `None` if orientations cannot be computed (e.g., .npy files with positions only)

### Joint Mapping
A dictionary mapping SMPLX joint names (keys) to robot **body/link** names (values) as they appear in the URDF:

```python
joint_mapping = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip_roll_link",
    "R_Hip": "right_hip_roll_link",
    "Spine1": "waist_yaw_link",
    "L_Knee": "left_knee_link",
    "R_Knee": "right_knee_link",
    "L_Ankle": "left_ankle_roll_link",
    "R_Ankle": "right_ankle_roll_link",
    "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link",
    "R_Elbow": "right_elbow_link",
    "L_Wrist": "left_wrist_yaw_link",
    "R_Wrist": "right_wrist_yaw_link",
}
```

The default SMPLX joint ordering used by `OmniRetargeter` (first 22 body joints) is:

```
Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2, L_Ankle, R_Ankle,
Spine3, L_Foot, R_Foot, Neck, L_Collar, R_Collar, Head, L_Shoulder,
R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist
```

You can override this list via the `smplx_joint_names` argument (or the
`smplx_joint_names` field in a robot profile JSON). Any SMPLX key in
`joint_mapping` must be present in `smplx_joint_names`, and the mapped value
must match a body name in the robot URDF; unresolved entries are filtered out
with a warning at initialization.

### Terrain Mesh
Supports common mesh formats:
- `.obj` (Wavefront OBJ)
- `.stl` (STL mesh)
- `.ply` (Polygon File Format)
- `.gltf`/`.glb` (glTF)

**Optional Terrain Scaling**: the terrain mesh is unscaled by default. If `enable_terrain_scaling=True` is passed to `retarget_motion()` (or `--output-scaled-terrain` is set on the CLI), OmniRetargeting computes a terrain scale factor from the robot/human height ratio and retargets against the scaled mesh.

### Robot URDF
Standard URDF format for humanoid robots. The system automatically:
- Detects robot height from the default pose (overridable via `robot_height`)
- Reads joint limits and types from the URDF
- Loads visual meshes for (optional) visualization

## Output Format

### `retarget_motion()` return value

```python
terrain_scale, retargeted_motion = retargeter.retarget_motion(
    smplx_trajectory,
    framerate=30.0,
    enable_terrain_scaling=True,
)
```

- **`terrain_scale`**: `1.0` by default, or the computed terrain scaling factor when `enable_terrain_scaling=True`.
- **`retargeted_motion`**: Numpy array of shape `(T, 7 + DOF)` containing:
  - `[0:3]`: Root position (x, y, z)
  - `[3:7]`: Root quaternion in **wxyz** order (MuJoCo convention)
  - `[7:]`: Joint angles in radians

### CLI `.npz` schema

`python -m omniretargeting.main --output my_motion.npz ...` writes a `.npz`
containing the following keys (the output filename is also normalized to end
with `_retargeted.npz` if it doesn't already):

| Key            | Shape      | Description                                     |
|----------------|------------|-------------------------------------------------|
| `framerate`    | scalar     | Motion framerate (from file or `--framerate`).  |
| `joint_names`  | `(DOF,)`   | Robot joint names (excluding the floating base). |
| `joint_pos`    | `(T, DOF)` | Joint angles in radians.                         |
| `base_pos_w`   | `(T, 3)`   | Root position in world frame.                    |
| `base_quat_w`  | `(T, 4)`   | Root quaternion in world frame (wxyz).           |

If `--output-scaled-terrain` is provided, the scaled terrain mesh used for
retargeting is exported to that path as well.

## Advanced Usage

### Custom Robot Height

```python
retargeter = OmniRetargeter(
    robot_urdf_path=robot_urdf,
    terrain_mesh_path=terrain_mesh,
    joint_mapping=joint_mapping,
    robot_height=1.8  # Override auto-detected height
)
```

### CLI

The CLI is driven by a per-robot JSON profile. The URDF path, joint mapping,
and retargeting settings all come from the profile — the CLI does **not**
accept a separate URDF argument.

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --smplx_model_dir /path/to/smplx/models \
  --smplx_motion /path/to/motion_stageii.npz \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz \
  --output-scaled-terrain /path/to/scaled-terrain.obj \
  --framerate 30 \
  --penetration-resolver xyz_nudge
```

Main arguments:

| Flag | Default | Description |
|---|---|---|
| `--robot-config` | `robot_models/unitree_g1/unitree_g1.json` | Path to robot profile JSON. |
| `--smplx_model_dir` | *(required)* | Directory containing SMPLX model files. |
| `--smplx_motion` | *(required)* | Path to SMPLX motion file (`.npz`). |
| `--output` | *(required)* | Output `.npz` path (normalized to end in `_retargeted.npz`). |
| `--terrain` | flat ground | Path to terrain mesh; a default flat terrain is generated if omitted. |
| `--output-scaled-terrain` | `None` | If set, enables terrain scaling and exports the scaled mesh. |
| `--mapping` | profile | Overrides the profile's `joint_mapping` with an external JSON file. |
| `--framerate` | auto / 30 | Motion framerate; auto-detected from the SMPLX file when possible. |
| `--vis` | off | Launch a MuJoCo viewer on the retargeted motion. |
| `--save-video PATH` | off | Render the retargeted motion to video (requires `imageio[ffmpeg]`, and `MUJOCO_GL=egl`/`osmesa` for headless). |
| `--replace-cylinders-with-capsules` | off | Swap cylinder collision geoms for capsules (IsaacLab/PhysX convention). |
| `--penetration-resolver {hard_constraint,xyz_nudge}` | `xyz_nudge` | Contact handling mode; overrides the value in the profile. |

### Robot Profile Config (Per-Humanoid)

Keep one JSON profile per humanoid robot (for example under
`robot_models/<robot_name>/`). Relative `urdf_path` values are resolved against
the profile file's directory.

Supported top-level fields:

- `name` – optional profile name, used in log output
- `urdf_path` – **required**, path to the robot URDF (relative to the profile file)
- `joint_mapping` – **required**, SMPLX joint name → robot body name
- `robot_height` – optional override for auto-detected robot height
- `smplx_joint_names` – optional custom SMPLX joint ordering
- `height_estimation` – head/foot joint names and `head_top_offset` used to estimate human height
- `base_orientation` – SMPLX joint names used to estimate root orientation (`pelvis`, `left_hip`, `right_hip`, `spine`)
- `retargeting` – solver settings forwarded to `GenericInteractionRetargeter`:
  - `collision_detection_threshold`
  - `terrain_sample_points`
  - `replace_cylinders_with_capsules`
  - `penetration_resolver`: `"hard_constraint"` or `"xyz_nudge"`
  - `foot_stabilization`: nested block (see `robot_models/unitree_g1/unitree_g1.json`) that controls the post-processing XYZ-nudge pass (`enabled`, `clearance`, `surface_clearance`, `contact_clearance`, `xy_correction_gain`, smoothing windows, wall-contact thresholds, etc.)

### Validation

```python
# Check if joint mapping is valid
missing_joints = retargeter.validate_joint_mapping()
if missing_joints:
    print(f"Warning: Missing joints: {missing_joints}")

# Get robot information
print(f"Robot DOF: {retargeter.get_robot_dof()}")
print(f"Joint names: {retargeter.get_joint_names()}")
```

## Running Tests

```bash
pytest tests/
```

## API Reference

### `OmniRetargeter`

Main class for motion retargeting (defined in `omniretargeting/core.py`).

#### Constructor
```python
OmniRetargeter(
    robot_urdf_path,
    terrain_mesh_path,
    joint_mapping,
    robot_height=None,
    smplx_joint_names=None,
    height_estimation=None,
    base_orientation=None,
    retargeting=None,
)
```

#### Methods

- `retarget_motion(smplx_trajectory, base_orientations=None, base_translations=None, framerate=None, visualize_trajectory=True, enable_terrain_scaling=False)` → `(terrain_scale, retargeted_motion)`
- `get_robot_dof()` → `int`
- `get_joint_names()` → `List[str]`
- `validate_joint_mapping()` → `List[str]` (robot body names from `joint_mapping` that are missing from the URDF)

### `load_robot_config`

```python
from omniretargeting import load_robot_config
cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")
```

Loads a robot profile JSON and resolves `urdf_path` relative to the profile
file. Raises if `joint_mapping` is missing or empty.

### `load_smplx_trajectory`

```python
from omniretargeting.utils import load_smplx_trajectory
positions, orientations = load_smplx_trajectory(
    smplx_file=Path("motion_stageii.npz"),
    smplx_model_directory="/path/to/smplx/models",
    gender="neutral",
)
```

Always returns `(positions, orientations)` (orientations may be `None` for
`.npy` files). Pass `return_meta=True` to additionally receive the raw
`root_orient` and `trans` arrays.

## Dependencies

Declared in `pyproject.toml` / `setup.py`:

- numpy, scipy, matplotlib, tqdm
- torch
- trimesh, smplx, jinja2
- mujoco (≥3.7 for URDF `strippath=false` default)
- viser, yourdfpy, robot_descriptions
- cvxpy, libigl, tyro
- open3d, pyvista

## Architecture

OmniRetargeting adapts the interaction-mesh retargeting approach from the
holosoma_retargeting project to work with generic robots and terrains:

1. **Terrain Scaling** (optional): Scales the terrain mesh by the robot-to-human height ratio before retargeting (enabled by `enable_terrain_scaling=True` or `--output-scaled-terrain`).
2. **Generic Robot Support**: Works with any URDF through automatic model loading, body-name validation, and auto-detected height.
3. **Interaction Mesh**: Builds a tetrahedral interaction mesh from mapped human joints and terrain sample points.
4. **Optimization**: Per-frame SQP optimization with Laplacian-deformation objective, joint limits, and a target base-orientation term for smoothness.
5. **Collision / Penetration Handling**: Two modes selectable via `retargeting.penetration_resolver`:
   - `hard_constraint` – penetration inequalities inside the SQP.
   - `xyz_nudge` – post-optimization foot stabilization that projects probe points out of the terrain and smooths XY drift (see `foot_stabilization` in the profile).
6. **Joint Limits**: Respects robot joint limits throughout.

## Limitations

- **Coordinate-system alignment**: A TODO remains in `utils.py` for a more
  principled SMPLX → world coordinate transformation; the current pipeline
  assumes the SMPLX trajectory is already in a +Z-up world frame.
- **Foot stabilization tuning**: The `xyz_nudge` resolver is effective on flat
  and mildly uneven terrain but may need per-robot tuning
  (`foot_stabilization` block in the profile) for complex scenes with walls.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## Citation

This repository is a re-implementation of the OmniRetarget method. If you use this code in your research, please cite the original paper:

```
@article{yang2025omniretarget,
  title={OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction},
  author={Yang, Lujie and Huang, Xiaoyu and Wu, Zhen and Kanazawa, Angjoo and Abbeel, Pieter and Sferrazza, Carmelo and Liu, C. Karen and Duan, Rocky and Shi, Guanya},
  journal={arXiv preprint arXiv:2509.26633},
  year={2025},
  url={https://arxiv.org/abs/2509.26633}
}
```
