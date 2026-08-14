"""Tests for the batch entry point's command construction."""

from pathlib import Path

from omniretargeting.utils.batch_processing import _build_command


def test_build_command_uniform_scale_skips_per_motion_terrain(tmp_path):
    cmd = _build_command(
        tmp_path / "c.yaml",
        "robot.json",
        tmp_path,
        "stem",
        save_video=False,
        scale_factor=0.85,
    )

    assert "--scale-factor" in cmd
    assert cmd[cmd.index("--scale-factor") + 1] == "0.85"
    # The scaled terrain is exported once at the batch level, not per motion.
    assert "--output-scaled-terrain" not in cmd


def test_build_command_default_exports_per_motion_terrain(tmp_path):
    cmd = _build_command(
        tmp_path / "c.yaml",
        "robot.json",
        tmp_path,
        "stem",
        save_video=False,
    )

    assert "--output-scaled-terrain" in cmd
    assert cmd[cmd.index("--output-scaled-terrain") + 1].endswith("stem_scaled_terrain.obj")
    assert "--scale-factor" not in cmd
