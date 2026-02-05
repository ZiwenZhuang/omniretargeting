"""
Helpers for building a robot+scene MuJoCo XML using ply2scene assets.
"""

from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    _ensure_dir(dst.parent)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _link_dir_contents(src_dir: Path, dst_dir: Path) -> None:
    _ensure_dir(dst_dir)
    for src in src_dir.iterdir():
        if src.is_file():
            _symlink_or_copy(src, dst_dir / src.name)


def create_scaled_scene_urdf(scene_urdf_path: Path, scale: float, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = scene_urdf_path.with_name(
            scene_urdf_path.stem + f"_scaled_{scale:.4f}" + scene_urdf_path.suffix
        )
    # If writing to a separate file and it already exists, treat it as a cache.
    # But if overwriting the source file (output_path == scene_urdf_path), we must rewrite
    # to ensure the scale matches the current pipeline scale_factor.
    if output_path.exists() and output_path.resolve() != scene_urdf_path.resolve():
        return output_path

    content = scene_urdf_path.read_text()
    desired = f'scale="{scale} {scale} {scale}"'
    if output_path.exists() and desired in content:
        # Overwrite target already has the expected scale; keep as-is.
        return output_path
    content = re.sub(r'scale="[^"]*"', f'scale="{scale} {scale} {scale}"', content)
    output_path.write_text(content)
    return output_path


def build_robot_scene_xml(
    base_robot_xml: Path,
    scene_dir: Path,
    output_dir: Path,
    *,
    scale: float,
    output_name: str = "robot_scene.xml",
    disable_plane_ground: bool = True,
) -> Path:
    """
    Build a MuJoCo XML that merges robot model and ply2scene meshes.

    The resulting XML is written under output_dir, and required robot/scene
    mesh assets are linked into output_dir based on the robot XML meshdir.
    """
    _ensure_dir(output_dir)

    tree = ET.parse(base_robot_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    meshdir_path = output_dir / meshdir if meshdir else output_dir
    _ensure_dir(meshdir_path)

    robot_dir = base_robot_xml.parent
    robot_mesh_dir = robot_dir / "meshes"
    robot_assets_dir = robot_dir / "assets"

    if "assets" in meshdir:
        _link_dir_contents(robot_assets_dir, meshdir_path)
    else:
        _link_dir_contents(robot_mesh_dir, meshdir_path)

    _link_dir_contents(robot_assets_dir, output_dir / "assets")

    scene_mesh_dir = scene_dir / "meshes"
    _symlink_or_copy(scene_mesh_dir / "scene_visual.obj", meshdir_path / "scene_visual.obj")
    _symlink_or_copy(scene_mesh_dir / "scene_collision.obj", meshdir_path / "scene_collision.obj")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    for child in list(asset):
        if child.tag == "mesh" and child.attrib.get("name") in {"scene_visual", "scene_collision"}:
            asset.remove(child)

    ET.SubElement(
        asset,
        "mesh",
        name="scene_visual",
        file="scene_visual.obj",
        scale=f"{scale} {scale} {scale}",
    )
    ET.SubElement(
        asset,
        "mesh",
        name="scene_collision",
        file="scene_collision.obj",
        scale=f"{scale} {scale} {scale}",
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    if disable_plane_ground:
        for child in list(worldbody):
            if child.tag == "geom" and child.attrib.get("type") == "plane":
                worldbody.remove(child)

    for child in list(worldbody):
        if child.tag == "body" and child.attrib.get("name") == "scene":
            worldbody.remove(child)

    scene_body = ET.SubElement(worldbody, "body", name="scene")
    ET.SubElement(
        scene_body,
        "geom",
        name="scene_collision",
        type="mesh",
        mesh="scene_collision",
        contype="1",
        conaffinity="1",
        rgba="0 0 0 0",
    )
    ET.SubElement(
        scene_body,
        "geom",
        name="scene_visual",
        type="mesh",
        mesh="scene_visual",
        contype="0",
        conaffinity="0",
        rgba="0.7 0.7 0.7 1",
    )

    output_path = output_dir / output_name
    tree.write(output_path)
    return output_path
