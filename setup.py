from setuptools import find_packages, setup  # type: ignore[import-untyped]

setup(
    name="omniretargeting",
    version="0.1.0",
    description="Generic motion retargeting for any humanoid URDF and terrain mesh.",
    author="OmniRetargeting Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies from holosoma_retargeting
        "numpy==2.3.5",
        "torch",
        "tqdm",
        "scipy",
        "matplotlib",
        "trimesh",
        "smplx",
        "jinja2",
        "mujoco",
        "viser",
        "robot_descriptions",
        "yourdfpy",
        "cvxpy",
        "libigl",
        "tyro",
        "imageio[ffmpeg]",
        # Additional dependencies for generic mesh processing
        "open3d",
        "pyvista",
    ],
)
