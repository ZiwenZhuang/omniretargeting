import json
from pathlib import Path

import pytest

from omniretargeting.robot_config import load_robot_config


def test_load_robot_config_resolves_relative_urdf(tmp_path: Path):
    config_path = tmp_path / "robot.json"
    rel_urdf = "robots/test.urdf"
    (tmp_path / "robots").mkdir()

    config = {
        "name": "test_robot",
        "urdf_path": rel_urdf,
        "joint_mapping": {"Pelvis": "pelvis"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_robot_config(config_path)

    assert loaded["joint_mapping"] == {"Pelvis": "pelvis"}
    assert loaded["urdf_path"] == str((tmp_path / rel_urdf).resolve())


def test_load_robot_config_requires_joint_mapping(tmp_path: Path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps({"name": "broken"}), encoding="utf-8")

    with pytest.raises(ValueError, match="joint_mapping"):
        load_robot_config(config_path)


def test_load_robot_config_requires_object(tmp_path: Path):
    config_path = tmp_path / "invalid_shape.json"
    config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_robot_config(config_path)
