# Robot Model Config Profiles

These JSON files define per-robot configuration for the OmniRetargeting CLI.

Use with:

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --smplx_model_dir /path/to/smplx/models \
  --smplx_motion /path/to/motion.npz \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz
```

Notes:
- Set `urdf_path` in the profile JSON; the CLI does not take a `--urdf` flag.
- `urdf_path` supports two formats:
  - **Relative path** — a bare filename or relative path referencing a URDF co-located with the JSON config (e.g., `"h1.urdf"` resolves to the path relative to the `unitree_h1.json` file). This is the preferred format when the URDF lives in the `robot_models/` directory.
  - **`package://` URI** — e.g., `"package://company_robots/company_robot_type/urdf/company_robot_type.urdf"`. The package name must be an importable Python package; the subpath is resolved relative to the package's directory.
- Link names in each source entry's target_mapping must match body names in your URDF/MuJoCo model.
- `unitree_h1/unitree_h1.json` is a starter profile and may need link-name adjustments for your specific URDF variant.
