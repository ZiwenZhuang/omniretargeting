"""
Batch processing script for OmniRetargeting.

Scans a folder of motion files, writes per-motion source configs,
and executes main.py for each file with scaled terrain and video export.

Usage:
    python -m omniretargeting.batch \
        --source-folder /path/to/motions \
        --source-type smplx \
        --robot-config robot_models/unitree_g1/unitree_g1.json \
        --output-dir /tmp/batch_output \
        --model-dir /path/to/smplx/models
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

_SOURCE_EXTENSIONS = {
    "smplx": [".npz"],
    "lafan1": [".bvh"],
    "nokov": [".bvh"],
    "omomo": [".npz"],
}


def detect_resources() -> dict:
    """Detect available CPU cores and total system memory."""
    cpu_count = os.cpu_count() or 1
    memory_gb = None
    try:
        import psutil
        mem = psutil.virtual_memory()
        memory_gb = mem.total / (1024**3)
    except ImportError:
        pass
    return {"cpu_count": cpu_count, "memory_gb": memory_gb}


def determine_batch_size(resources: dict) -> int:
    """Choose a conservative worker count from available resources."""
    cpu_count = resources["cpu_count"]
    workers = max(1, cpu_count - 1)
    memory_gb = resources.get("memory_gb")
    if memory_gb is not None and memory_gb < 4:
        workers = min(workers, 2)
    return workers


def scan_source_folder(folder: Path, source_type: str) -> list[Path]:
    """Return motion files in *folder* matching *source_type*, sorted by name."""
    exts = _SOURCE_EXTENSIONS.get(source_type, [".npz", ".bvh"])
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(folder.glob(f"*{ext}")))
    return files


def write_source_config(
    motion_path: Path,
    source_type: str,
    output_dir: Path,
    terrain_path: str | None = None,
    extra_options: dict | None = None,
) -> Path:
    """Write a per-motion YAML source config and return its path."""
    config: dict = {
        "type": source_type,
        "motion": str(motion_path),
    }
    if terrain_path:
        config["terrain"] = terrain_path
    if extra_options:
        config.update(extra_options)

    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{motion_path.stem}_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def get_activation_prefix() -> str:
    """Return a shell activation command for the current conda/virtualenv, or ''."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_base = conda_prefix.rsplit("/envs", 1)[0] if "/envs" in conda_prefix else ""
        if not conda_base:
            for candidate in [
                os.path.expanduser("~/anaconda3"),
                os.path.expanduser("~/miniconda3"),
            ]:
                if os.path.isdir(candidate):
                    conda_base = candidate
                    break
        if conda_base:
            return f"source {conda_base}/etc/profile.d/conda.sh && conda activate {conda_env}"
        return f"conda activate {conda_env}"

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return f"source {virtual_env}/bin/activate"
    return ""


def _find_git_root(path: Path) -> str | None:
    """Walk upward from *path* to find the nearest .git directory."""
    path = path.resolve()
    while path != path.parent:
        if (path / ".git").exists():
            return str(path)
        path = path.parent
    return None


def _git_status(repo_path: str) -> str:
    """Return branch, commit, and dirty status for a git repository."""
    lines: list[str] = []
    try:
        for args, label in [
            (["rev-parse", "--abbrev-ref", "HEAD"], "Branch"),
            (["rev-parse", "HEAD"], "Commit"),
        ]:
            r = subprocess.run(
                ["git", "-C", repo_path] + args,
                capture_output=True, text=True, timeout=10,
            )
            lines.append(f"{label}: {r.stdout.strip()}")

        r = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = r.stdout.strip()
        lines.append(f"Dirty: {'yes' if dirty else 'no'}")
        if dirty:
            lines.append("Changes:")
            for line in dirty.split("\n")[:50]:
                lines.append(f"  {line}")
    except Exception as exc:
        lines.append(f"Error: {exc}")
    return "\n".join(lines)


def log_repo_status(output_dir: Path, robot_config_path: str | None = None) -> Path:
    """Write git status of this repo (and URDF package repo if applicable) to a log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_file = output_dir / "repo_status.log"

    parts: list[str] = []
    parts.append("=== Repository Status Log ===")
    parts.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("")

    parts.append(f"--- OmniRetargeting ({REPO_ROOT}) ---")
    parts.append(_git_status(str(REPO_ROOT)))
    parts.append("")

    if robot_config_path:
        rc_path = Path(robot_config_path)
        if rc_path.exists():
            with open(rc_path) as f:
                robot_config = json.load(f)
            urdf_path = robot_config.get("urdf_path") or robot_config.get("robot", {}).get("urdf_path", "")
            if isinstance(urdf_path, str) and urdf_path.startswith("package://"):
                pkg_name = urdf_path[len("package://"):].split("/")[0]
                try:
                    import importlib
                    mod = importlib.import_module(pkg_name)
                    if mod.__file__:
                        pkg_dir = Path(mod.__file__).parent
                        git_root = _find_git_root(pkg_dir)
                        if git_root:
                            parts.append(f"--- URDF Package '{pkg_name}' ({git_root}) ---")
                            parts.append(_git_status(git_root))
                            parts.append("")
                except Exception:
                    pass

    with open(status_file, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"Repository status logged to {status_file}")
    return status_file


def _build_command(
    source_config_path: Path,
    robot_config_path: str,
    output_dir: Path,
    motion_stem: str,
    framerate: float | None = None,
) -> list[str]:
    """Build the main.py argument list for one motion file."""
    motion_dir = output_dir / "motions" / motion_stem
    motion_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "omniretargeting.main",
        "--robot-config", robot_config_path,
        "--source-config", str(source_config_path),
        "--output", str(motion_dir / f"{motion_stem}_retargeted.npz"),
        "--output-scaled-terrain", str(motion_dir / f"{motion_stem}_scaled_terrain.obj"),
        "--save-video", str(motion_dir / f"{motion_stem}_retargeted.mp4"),
        "--scaled-objects", str(motion_dir / f"{motion_stem}_scaled_objects"),
    ]
    if framerate is not None:
        cmd.extend(["--framerate", str(framerate)])
    return cmd


def run_single_job(cmd: list[str], activation_prefix: str, log_file: Path, timeout: float) -> dict:
    """Launch one retargeting subprocess and return timing/status."""
    if activation_prefix:
        full_cmd = f"{activation_prefix} && {' '.join(cmd)}"
        shell_cmd: list[str] = ["bash", "-lc", full_cmd]
    else:
        shell_cmd = cmd

    start = time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_file, "w") as f:
            proc = subprocess.run(shell_cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        elapsed = time.time() - start
        return {"returncode": proc.returncode, "elapsed": elapsed, "log_file": str(log_file)}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        with open(log_file, "a") as f:
            f.write(f"\n\n*** Killed after timeout ({timeout:.0f}s, elapsed {elapsed:.0f}s) ***\n")
        return {"returncode": -1, "elapsed": elapsed, "log_file": str(log_file), "timed_out": True}


def _run_test_job(
    motion_files: list[Path],
    source_type: str,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None,
    extra_options: dict,
    framerate: float | None,
    timeout: float,
) -> dict:
    """Run the first motion as a probe job.  Returns timing info for batch-size tuning."""
    first = motion_files[0]
    print(f"\n--- Running test job: {first.name} ---")
    config_path = write_source_config(first, source_type, output_dir, terrain_path, extra_options)
    cmd = _build_command(config_path, robot_config_path, output_dir, first.stem, framerate)
    log_file = output_dir / "logs" / f"{first.stem}.log"

    print(f"  Command: {' '.join(cmd)}")
    result = run_single_job(cmd, activation_prefix, log_file, timeout)
    result["motion_file"] = str(first)
    result["motion_stem"] = first.stem

    if result.get("timed_out"):
        print(f"  Test job TIMED OUT ({result['elapsed']:.0f}s > {timeout:.0f}s limit)")
    elif result["returncode"] == 0:
        print(f"  Test job OK ({result['elapsed']:.1f}s)")
    else:
        print(f"  Test job FAILED (rc={result['returncode']}, {result['elapsed']:.1f}s)")
        print(f"  Log: {result['log_file']}")
    return result


def process_batch(
    motion_files: list[Path],
    source_type: str,
    robot_config_path: str,
    output_dir: Path,
    activation_prefix: str,
    terrain_path: str | None = None,
    extra_options: dict | None = None,
    _max_workers: int = 1,
    framerate: float | None = None,
    skip_test_job: bool = False,
    timeout: float = 3600,
) -> list[dict]:
    """Process every motion file, returning per-file results.

    Note: *_max_workers* is reserved for future parallel-execution support.
    The current implementation processes jobs sequentially for safety.
    """
    if extra_options is None:
        extra_options = {}

    results: list[dict] = []
    remaining = list(motion_files)

    # Run first motion as a probe to validate the pipeline (unless skipped)
    if remaining and not skip_test_job:
        test_result = _run_test_job(
            remaining, source_type, robot_config_path, output_dir,
            activation_prefix, terrain_path, extra_options, framerate, timeout,
        )
        results.append(test_result)
        if test_result["returncode"] != 0:
            print("\nTest job failed. Aborting batch. Check the log above.")
            return results
        remaining = remaining[1:]

    # Process remaining motions
    for i, motion_file in enumerate(remaining):
        motion_stem = motion_file.stem
        print(f"\n[{i + len(results) + 1}/{len(motion_files)}] Processing: {motion_file.name}")

        config_path = write_source_config(motion_file, source_type, output_dir, terrain_path, extra_options)
        cmd = _build_command(config_path, robot_config_path, output_dir, motion_stem, framerate)
        log_file = output_dir / "logs" / f"{motion_stem}.log"

        print(f"  Command: {' '.join(cmd)}")
        result = run_single_job(cmd, activation_prefix, log_file, timeout)
        result["motion_file"] = str(motion_file)
        result["motion_stem"] = motion_stem
        results.append(result)

        if result.get("timed_out"):
            print(f"  TIMED OUT ({result['elapsed']:.0f}s > {timeout:.0f}s limit)")
        elif result["returncode"] == 0:
            print(f"  OK ({result['elapsed']:.1f}s)")
        else:
            print(f"  FAILED (rc={result['returncode']}, {result['elapsed']:.1f}s)")
            print(f"  Log: {result['log_file']}")

    return results


def _summarize(results: list[dict]) -> int:
    total = len(results)
    ok = sum(1 for r in results if r["returncode"] == 0)
    timed_out = sum(1 for r in results if r.get("timed_out"))
    failed = total - ok
    print(f"\n{'=' * 60}")
    print(f"Batch complete: {ok}/{total} succeeded, {failed} failed")
    if timed_out:
        print(f"  ({timed_out} timed out)")
    if failed:
        print("Failures:")
        for r in results:
            if r.get("timed_out"):
                print(f"  - {r['motion_stem']}: TIMED OUT after {r['elapsed']:.0f}s (log: {r['log_file']})")
            elif r["returncode"] != 0:
                print(f"  - {r['motion_stem']}: rc={r['returncode']} (log: {r['log_file']})")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniRetargeting Batch Processor")
    parser.add_argument("--source-folder", required=True,
                        help="Folder containing motion files to process")
    parser.add_argument("--source-type", required=True,
                        choices=["smplx", "lafan1", "nokov", "omomo"],
                        help="Type of motion data source")
    parser.add_argument("--robot-config", required=True,
                        help="Path to robot configuration JSON file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save batch outputs")
    parser.add_argument("--model-dir", default=None,
                        help="Model directory (required for SMPL-X source type)")
    parser.add_argument("--terrain", default=None,
                        help="Path to terrain mesh file applied to all motions")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum parallel workers (default: auto-detect)")
    parser.add_argument("--framerate", type=float, default=None,
                        help="Override framerate for all motions")
    parser.add_argument("--source-options", default=None,
                        help="JSON string of extra source options for all motions")
    parser.add_argument("--skip-test-job", action="store_true",
                        help="Skip the initial probe job and process all files directly")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Per-file timeout in seconds (default: 3600)")

    args = parser.parse_args()

    source_folder = Path(args.source_folder)
    if not source_folder.is_dir():
        print(f"Error: source folder not found: {args.source_folder}")
        sys.exit(1)

    robot_config_path = str(Path(args.robot_config).expanduser())
    if not Path(robot_config_path).exists():
        print(f"Error: robot config not found: {args.robot_config}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Environment detection
    resources = detect_resources()
    print(f"CPU cores: {resources['cpu_count']}")
    if resources["memory_gb"] is not None:
        print(f"Total memory: {resources['memory_gb']:.1f} GB")

    max_workers = args.max_workers or determine_batch_size(resources)
    print(f"Max workers: {max_workers}")

    activation_prefix = get_activation_prefix()
    if activation_prefix:
        print(f"Environment activation: detected")
    else:
        print("Warning: no conda/virtualenv detected, using bare Python")

    # Log repository status before processing
    log_repo_status(output_dir, args.robot_config)

    # Scan for motion files
    motion_files = scan_source_folder(source_folder, args.source_type)
    if not motion_files:
        print(f"Error: no motion files found in {args.source_folder} for type '{args.source_type}'")
        sys.exit(1)
    print(f"Found {len(motion_files)} motion file(s)")

    # Build shared extra options
    extra_options: dict = {}
    if args.model_dir:
        extra_options["model_directory"] = args.model_dir
    if args.source_options:
        extra_options.update(json.loads(args.source_options))

    # Process batch
    results = process_batch(
        motion_files=motion_files,
        source_type=args.source_type,
        robot_config_path=robot_config_path,
        output_dir=output_dir,
        activation_prefix=activation_prefix,
        terrain_path=args.terrain,
        extra_options=extra_options,
        _max_workers=max_workers,
        framerate=args.framerate,
        skip_test_job=args.skip_test_job,
        timeout=args.timeout,
    )

    failed = _summarize(results)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
