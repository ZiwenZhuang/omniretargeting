# Progress Summary

Date: 2026-03-26

## Completed

- Added per-humanoid robot profile support via JSON config loader:
  - `omniretargeting/robot_config.py`
  - validates config shape and required `joint_mapping`
  - resolves relative `urdf_path` from profile directory
- Integrated robot profile into CLI:
  - `--robot-config` added and now defaults to `robot_models/unitree_g1/unitree_g1.json`
  - profile values are wired into `OmniRetargeter` (`joint_mapping`, `urdf_path`, `robot_height`, `smplx_joint_names`, `height_estimation`, `base_orientation`, `retargeting`)
  - `--mapping` still supported and overrides profile mapping
- Removed redundant hardcoded default mapping in `main.py`:
  - deleted `get_default_joint_mapping()`
  - default behavior now comes from default CLI profile
- Added configurable retargeting parameters per robot profile:
  - `collision_detection_threshold`
  - `terrain_sample_points`
- Added configurable SMPLX-based heuristics per profile:
  - height estimation joint names/offset
  - base orientation reference joints
- Added/updated robot profiles and docs:
  - `robot_models/unitree_g1/unitree_g1.json`
  - `robot_models/unitree_h1/unitree_h1.json` (starter profile)
  - `robot_models/README.md`
- Set default G1 URDF in profile:
  - `robot_models/unitree_g1/unitree_g1.json` includes `"urdf_path": "g1_29dof_popsicle.urdf"`
  - resolves to `robot_models/unitree_g1/g1_29dof_popsicle.urdf`
- Updated package exports and packaging:
  - `omniretargeting/__init__.py` exports `load_robot_config`
  - `MANIFEST.in` includes config files
- Added tests:
  - `tests/test_robot_config.py` for config loading/validation
- Reorganized robot assets at repo root under `robot_models/` (per-robot subdirectories):
  - URDF, meshes, and JSON profile are colocated under `robot_models/<robot_name>/`
  - e.g. `robot_models/unitree_g1/` contains `unitree_g1.json`, `g1_29dof_popsicle.urdf`, and `meshes/`

## Validation Performed

- AST/syntax checks passed for modified Python files.
- Path check confirmed the default G1 URDF path exists.
- Could not run full pytest in current shell because `pytest` is not installed.

## Current Status

- Added configurable penetration resolver modes (`hard_constraint` and `xyz_nudge`) through robot profile + CLI plumbing.
- Ported xyz-nudge foot stabilization post-processing and robust terrain-height fallback into the current branch.

- Removed unused `foot_geom_keywords` config plumbing and dead `collision_pairs` setup; terrain penetration remains driven by active trimesh/MuJoCo constraint code.

- Individual robot configuration flow is functional and now the default path.
- Redundant default mapping logic has been removed.
- Default G1 profile supplies `urdf_path` in JSON; the CLI no longer exposes `--urdf`.

## Next Suggested Work

- Add profile-level validation command to check that all mapped link names exist in a given URDF before retargeting.
- Add more robot profiles (with verified link names) for each target humanoid.
- Install test dependencies and run full test suite.

- Added omniretargeting.visualize_offsets to render default SMPL-X joints against robot links from a robot config.
