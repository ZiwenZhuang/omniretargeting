import argparse
import numpy as np
import trimesh
from pathlib import Path
import tempfile
import os
import json
import time

from omniretargeting import OmniRetargeter
from omniretargeting.utils import load_smplx_trajectory

def visualize_trajectory(urdf_path, trajectory, smplx_trajectory=None):
    """Visualize the retargeted trajectory and optional SMPLX joints in MuJoCo viewer."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("Error: mujoco package not found. Cannot visualize.")
        return

    print("Launching viewer...")
    print("Controls: Space to pause/resume, [ and ] to step frames.")
    
    # Load model
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    
    # Brighten the scene by increasing ambient light
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]  # Increase ambient light
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]  # Increase diffuse light
    model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]  # Add specular highlights
    
    # Set map values for better visibility
    model.vis.map.znear = 0.001  # Better near clipping
    model.vis.map.zfar = 50.0    # Better far clipping
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Configure viewer for better visibility
        # Try to access scene for background color and rendering settings
        scene = None
        if hasattr(viewer, 'user_scn'):
            scene = viewer.user_scn
            print("Using viewer.user_scn")
        elif hasattr(viewer, 'scn'):
            scene = viewer.scn
            print("Using viewer.scn")
        else:
            print(f"Viewer attributes: {dir(viewer)}")
            print("Note: Could not find scene object (scn or user_scn)")
        
        if scene is not None:
            try:
                # Disable skybox and fog - just use solid background
                scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
                scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0  # Fog was making it darker!
                
                # Set background color to bright white/light gray (RGBA)
                if hasattr(scene, 'rgba_background'):
                    scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]
                    print(f"Background color set to: {scene.rgba_background}")
                    
                print("Scene rendering customized successfully")
            except (AttributeError, TypeError) as e:
                print(f"Could not customize scene: {e}")
        
        # Enable coordinate frame visualization
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # Show world frame
        
        # Show visual geometries (meshes) instead of collision shapes
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0  # Hide convex hulls
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1      # Show static bodies
        
        num_frames = len(trajectory)
        frame_idx = 0
        smplx_frame_idx = 0
        
        # Playback speed control
        fps = 30.0
        dt = 1.0 / fps
        
        # Setup SMPLX joint visualization (spheres) if provided
        smplx_geoms_base = None
        smplx_num_joints = 0
        if smplx_trajectory is not None and scene is not None:
            smplx_num_joints = smplx_trajectory.shape[1]
            smplx_geoms_base = scene.ngeom
            for i in range(smplx_num_joints):
                geom = scene.geoms[smplx_geoms_base + i]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.02, 0.0, 0.0]),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.1, 0.9, 0.1, 0.9]),
                )
            scene.ngeom = smplx_geoms_base + smplx_num_joints

        while viewer.is_running():
            step_start = time.time()
            
            # Update background color every frame (some viewers need this)
            if scene is not None and hasattr(scene, 'rgba_background'):
                scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]
            
            # Update state
            data.qpos[:] = trajectory[frame_idx]
            mujoco.mj_forward(model, data)

            # Update SMPLX joint markers
            if smplx_geoms_base is not None:
                smplx_joints = smplx_trajectory[smplx_frame_idx]
                for i in range(smplx_num_joints):
                    scene.geoms[smplx_geoms_base + i].pos = smplx_joints[i]
            
            # Advance frame
            frame_idx = (frame_idx + 1) % num_frames
            if smplx_trajectory is not None:
                smplx_frame_idx = (smplx_frame_idx + 1) % len(smplx_trajectory)
            
            # Sync viewer
            viewer.sync()
            
            # Sleep to maintain frame rate
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def create_flat_terrain(size=10.0):
    """Create a simple flat terrain mesh."""
    mesh = trimesh.creation.box(extents=[size, size, 0.1])
    mesh.apply_translation([0, 0, -0.05])
    return mesh

def get_default_joint_mapping():
    """Return a default joint mapping for Unitree G1 robot.
    
    Keys are standard SMPLX joint names.
    Values are Unitree G1 body names from the URDF.
    """
    return {
        # SMPLX joint name -> Unitree G1 body name (link names from URDF)
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_roll_link",  # Hip joints needed for proper leg tracking
        "R_Hip": "right_hip_roll_link",
        "Spine1": "waist_yaw_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Ankle": "left_ankle_roll_link",
        "R_Ankle": "right_ankle_roll_link",
        "L_Shoulder": "left_shoulder_roll_link",  # Shoulder joints needed for proper arm chain
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Wrist": "left_wrist_yaw_link",
        "R_Wrist": "right_wrist_yaw_link",
    }

def main():
    parser = argparse.ArgumentParser(description="OmniRetargeting CLI")
    parser.add_argument("--urdf", required=True, help="Path to robot URDF file")
    parser.add_argument("--smplx_model_dir", required=True, help="Directory containing SMPLX model files")
    parser.add_argument("--smplx_motion", required=True, help="Path to SMPLX motion file (.npz)")
    parser.add_argument("--output", required=True, help="Path to save output motion (.npy)")
    parser.add_argument("--terrain", help="Path to terrain mesh file (optional, defaults to flat ground)")
    parser.add_argument("--mapping", help="Path to joint mapping JSON file (optional, uses default if not provided)")
    parser.add_argument("--vis", action="store_true", help="Visualize the retargeted motion")
    
    args = parser.parse_args()
    
    # Load joint mapping
    if args.mapping:
        with open(args.mapping, 'r') as f:
            joint_mapping = json.load(f)
    else:
        print("Using default joint mapping.")
        joint_mapping = get_default_joint_mapping()

    # Handle terrain
    temp_terrain_path = None
    if args.terrain:
        terrain_path = args.terrain
    else:
        print("No terrain provided, creating default flat terrain.")
        flat_terrain = create_flat_terrain()
        fd, temp_terrain_path = tempfile.mkstemp(suffix=".obj")
        os.close(fd)
        flat_terrain.export(temp_terrain_path)
        terrain_path = temp_terrain_path

    try:
        # Load SMPLX trajectory
        print(f"Loading SMPLX trajectory from {args.smplx_motion}...")
        smplx_trajectory, smplx_orientations = load_smplx_trajectory(
            smplx_file=Path(args.smplx_motion),
            smplx_model_directory=args.smplx_model_dir,
        )
        print(f"Loaded trajectory with shape: {smplx_trajectory.shape}")
        if smplx_orientations is not None:
            print(f"Loaded orientations with shape: {smplx_orientations.shape}")
        else:
            print("Warning: Orientations not available for this file format.")

        # Initialize Retargeter
        print("Initializing OmniRetargeter...")
        retargeter = OmniRetargeter(
            robot_urdf_path=args.urdf,
            terrain_mesh_path=terrain_path,
            joint_mapping=joint_mapping,
        )

        # Perform retargeting
        print("Retargeting motion...")
        terrain_scale, retargeted_motion = retargeter.retarget_motion(
            smplx_trajectory,
            visualize_trajectory=args.vis,
        )
        
        # Save output
        print(f"Saving output to {args.output}...")
        np.save(args.output, retargeted_motion)
        print(f"Done! Terrain scale used: {terrain_scale}")

        if args.vis:
            visualize_trajectory(args.urdf, retargeted_motion, smplx_trajectory * terrain_scale)

    finally:
        # Cleanup temp file
        if temp_terrain_path and os.path.exists(temp_terrain_path):
            os.remove(temp_terrain_path)

if __name__ == "__main__":
    main()

