# OmniRetargeting Progress

## Architecture Summary

- Source-agnostic adapter architecture with per-source `target_mapping` and `base_orientation`
- Data sources: SMPL-X, OMOMO, LAFAN1 (BVH), Nokov (BVH)
- HOI support: `object_points` in `MotionData`/`MotionFrame`, scene scaling, OMOMO adapter
- CLI: YAML `--source-config` mode + legacy CLI compatibility
- Visualization: MuJoCo offscreen rendering, object mesh injection, `--save-video`

## Recent Changes (2026-06-22)

### LAFAN1 Batch Retargeting
- Added `omniretargeting/batch.py` — parallel batch processing with memory-based worker sizing
- Removed ground-ensuring (foot Z=0 shift) from LAFAN1 and Nokov data sources

### MuJoCo `mj_collision` FatalError Fix
- **Symptom:** `mujoco.FatalError: mj_narrowphase: collision function returned 9 contacts for geom pair, expected at most 8 from mj_maxContact`
- **Root cause:** `mjMAXCONPAIR=8` is a compile-time constant in MuJoCo. Box-box face-face overlap can produce up to 12 contacts.
- **MuJoCo upstream (v3.9.0):** Not fixed. The limit is still hardcoded.
- **Fix in `retargeting.py`:** `_prefilter_pairs_with_mj_collision()` catches `mujoco.FatalError` and falls back to `_brute_force_candidate_pairs()` (pairwise `mj_geomDistance`, negligible cost).
- **Alternative long-term fix:** Change fist geoms from boxes to capsules/spheres in the URDF.

## Key Files

### Core
- `omniretargeting/core.py` — OmniRetargeter, scene scaling, frame dispatch
- `omniretargeting/retargeting.py` — GenericInteractionRetargeter, collision constraints
- `omniretargeting/main.py` — CLI entry point (YAML + legacy)
- `omniretargeting/batch.py` — parallel batch processing

### Data Sources
- `omniretargeting/data_sources/base.py` — MotionData/MotionFrame containers
- `omniretargeting/data_sources/smplx.py` — SMPL-X adapter
- `omniretargeting/data_sources/omomo.py` — OMOMO object interaction adapter
- `omniretargeting/data_sources/lafan1.py` — LAFAN1 BVH adapter
- `omniretargeting/data_sources/nokov.py` — Nokov BVH adapter

### Robot Configs
- `robot_models/unitree_g1/unitree_g1.json`
- `robot_models/unitree_h1/unitree_h1.json`
- `robot_models/booster_k1/booster_k1.json`
- `robot_models/hightorque_mini_pi_plus/hightorque_mini_pi_plus.json`

### Tests
- `tests/test_basic.py` — CLI and regression coverage
- `tests/test_objects.py` — object-point unit coverage
- `tests/test_omomo_integration.py` — OMOMO integration
- `tests/data_sources/test_lafan1.py` — 15 tests
- `tests/data_sources/test_nokov.py` — 16 tests
- `tests/data_sources/test_smplx.py` — 6 tests

## Remaining Work

1. Validate HOI retargeting quality with real OMOMO end-to-end runs
2. Decide whether to add regression tests for YAML config loading and scaled-object export
3. Complete LAFAN1 batch retargeting and validate output quality
