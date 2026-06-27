"""Heuristic motion-file filters for dataset curation.

Each filter is a callable class: ``__init__`` takes hyperparameters,
``__call__(path)`` returns ``True`` to keep the file.
Filters can be composed via :class:`FilterPipeline`.

Use :func:`filter_dataset` to apply filters to a directory tree and
output selected files as symlinks or copies, preserving relative structure.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def _git_status() -> dict | None:
    """Capture git commit, branch, and per-file status of this repository."""
    repo_dir = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, text=True,
        ).strip()
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain", "-u"], cwd=repo_dir, text=True,
        ).strip()
        files = {}
        for line in status_output.splitlines():
            code = line[:2].strip()
            path = line[3:]
            files[path] = code
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(files),
            "files": files,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class FilterPipeline:
    """Chain multiple filters with AND semantics."""

    def __init__(self, filters: Sequence[MotionFilter] | None = None):
        self._filters: list[MotionFilter] = list(filters or [])

    def add(self, f: MotionFilter) -> FilterPipeline:
        self._filters.append(f)
        return self

    def __call__(self, path: str | Path) -> bool:
        path = Path(path)
        return all(f(path) for f in self._filters)

    def to_dict(self) -> dict:
        return {"filters": [f.to_dict() for f in self._filters]}

    def __repr__(self) -> str:
        return f"FilterPipeline({self._filters})"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MotionFilter:
    """Abstract base for motion-file filters."""

    def __call__(self, path: str | Path) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Filename-based filters
# ---------------------------------------------------------------------------

class ExcludeMirrored(MotionFilter):
    """Reject mirrored takes (filenames ending with ``_M`` before extension)."""

    def __init__(self, suffix: str = "_M"):
        self._suffix = suffix

    def __call__(self, path: str | Path) -> bool:
        stem = Path(path).stem
        return not stem.endswith(self._suffix)

    def to_dict(self) -> dict:
        return {"type": "ExcludeMirrored", "suffix": self._suffix}

    def __repr__(self) -> str:
        return f"ExcludeMirrored(suffix={self._suffix!r})"


class ExcludeByKeyword(MotionFilter):
    """Reject files whose stem contains any of the given keywords (case-insensitive)."""

    def __init__(self, keywords: Sequence[str]):
        self._keywords = [k.lower() for k in keywords]

    def __call__(self, path: str | Path) -> bool:
        stem = Path(path).stem.lower()
        return not any(kw in stem for kw in self._keywords)

    def to_dict(self) -> dict:
        return {"type": "ExcludeByKeyword", "keywords": self._keywords}

    def __repr__(self) -> str:
        return f"ExcludeByKeyword({self._keywords})"


class IncludeByKeyword(MotionFilter):
    """Keep only files whose stem contains at least one of the given keywords."""

    def __init__(self, keywords: Sequence[str]):
        self._keywords = [k.lower() for k in keywords]

    def __call__(self, path: str | Path) -> bool:
        stem = Path(path).stem.lower()
        return any(kw in stem for kw in self._keywords)

    def to_dict(self) -> dict:
        return {"type": "IncludeByKeyword", "keywords": self._keywords}

    def __repr__(self) -> str:
        return f"IncludeByKeyword({self._keywords})"


class ExcludeByRegex(MotionFilter):
    """Reject files whose stem matches the given regex pattern."""

    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        self._rx = re.compile(pattern, flags)

    def __call__(self, path: str | Path) -> bool:
        return self._rx.search(Path(path).stem) is None

    def to_dict(self) -> dict:
        return {"type": "ExcludeByRegex", "pattern": self._rx.pattern}

    def __repr__(self) -> str:
        return f"ExcludeByRegex({self._rx.pattern!r})"


class IncludeByRegex(MotionFilter):
    """Keep only files whose stem matches the given regex pattern."""

    def __init__(self, pattern: str, flags: int = re.IGNORECASE):
        self._rx = re.compile(pattern, flags)

    def __call__(self, path: str | Path) -> bool:
        return self._rx.search(Path(path).stem) is not None

    def to_dict(self) -> dict:
        return {"type": "IncludeByRegex", "pattern": self._rx.pattern}

    def __repr__(self) -> str:
        return f"IncludeByRegex({self._rx.pattern!r})"


# ---------------------------------------------------------------------------
# Humanoid planar-motion filter
# ---------------------------------------------------------------------------

# Motions to EXCLUDE: those with no meaningful ground interaction.
# Ground-contact motions (crawling, kneeling, handstands, floor exercises,
# rolling, falling/fainting to ground, etc.) are all KEPT.

_AIRBORNE_KEYWORDS: list[str] = [
    # Flying / supernatural — no ground contact
    "flying", "levitat", "float_air", "hover",
    # Hanging / swinging — suspended, no ground plane
    "hanging", "monkey_bar", "trapez",
    # Swimming — aquatic, no ground plane
    "swim",
]

_FURNITURE_SEATED_KEYWORDS: list[str] = [
    # Sitting on furniture (chair, bench, stool) — not ground-planar
    "sit_on_chair", "sit_on_bench", "sit_on_stool",
    "sitting_on_chair", "sitting_on_bench", "sitting_on_stool",
    "having_a_sit", "seated_chair", "chair_sit",
]

_AIRBORNE_REGEX_PARTS: list[str] = [
    r"\bhang\b",         # bare "hang" but not "change", "hanger" (tool)
    r"(?<!\w)flying\b",  # "flying_superman" etc., not "outflying"
]


class HumanGroundMotionFilter(MotionFilter):
    """Keep motions with meaningful ground-plane interaction.

    Excludes only:
      - Furniture-seated motions (sitting on chair/bench/stool)
      - Airborne / flying / levitating (no ground contact)
      - Suspended motions (hanging, trapeze, monkey bars)
      - Swimming (aquatic, no ground plane)

    Keeps all ground-contact motions including crawling, kneeling,
    lying on floor, handstands, cartwheels, rolling, falling, etc.

    Parameters
    ----------
    extra_exclude_keywords:
        Additional keywords to reject on top of the built-in list.
    exclude_mirrored:
        Also reject mirrored ``_M`` takes.
    """

    def __init__(
        self,
        extra_exclude_keywords: Sequence[str] | None = None,
        exclude_mirrored: bool = False,
    ):
        kws = list(_AIRBORNE_KEYWORDS) + list(_FURNITURE_SEATED_KEYWORDS)
        if extra_exclude_keywords:
            kws.extend(extra_exclude_keywords)
        self._keywords = [k.lower() for k in kws]

        parts = list(_AIRBORNE_REGEX_PARTS)
        self._rx = re.compile("|".join(parts), re.IGNORECASE) if parts else None
        self._exclude_mirrored = exclude_mirrored

    def __call__(self, path: str | Path) -> bool:
        stem = Path(path).stem.lower()

        if self._exclude_mirrored and stem.endswith("_m"):
            return False

        if any(kw in stem for kw in self._keywords):
            return False

        if self._rx and self._rx.search(stem):
            return False

        return True

    def to_dict(self) -> dict:
        extra = [k for k in self._keywords
                 if k not in [x.lower() for x in _AIRBORNE_KEYWORDS + _FURNITURE_SEATED_KEYWORDS]]
        d: dict = {"type": "HumanGroundMotionFilter"}
        if extra:
            d["extra_exclude_keywords"] = extra
        if self._exclude_mirrored:
            d["exclude_mirrored"] = True
        return d

    def __repr__(self) -> str:
        return (
            f"HumanGroundMotionFilter("
            f"n_keywords={len(self._keywords)}, "
            f"exclude_mirrored={self._exclude_mirrored})"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_ground_locomotion_filter(
    exclude_mirrored: bool = False,
    extra_exclude: Sequence[str] | None = None,
) -> FilterPipeline:
    """Return a pipeline that keeps human motions with ground-plane interaction."""
    return FilterPipeline([
        HumanGroundMotionFilter(
            extra_exclude_keywords=extra_exclude,
            exclude_mirrored=exclude_mirrored,
        ),
    ])


# ---------------------------------------------------------------------------
# Dataset filtering
# ---------------------------------------------------------------------------

def filter_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    filter_fn: MotionFilter | FilterPipeline,
    *,
    mode: str = "symlink",
    extensions: Sequence[str] | None = None,
    recursive: bool = True,
) -> dict:
    """Apply *filter_fn* to every motion file under *input_dir* and output
    selected files into ``{output_dir}/results/``, preserving the relative
    directory structure.

    Parameters
    ----------
    input_dir:
        Root directory to scan for motion files.
    output_dir:
        Destination root.  Selected files land in ``{output_dir}/results/``
        and the filtering config is saved to ``{output_dir}/filtering_config.json``.
    filter_fn:
        A :class:`MotionFilter` or :class:`FilterPipeline` instance.
    mode:
        ``"symlink"`` — create symbolic links to the original files.
        ``"copy"`` — copy files into the output tree.
    extensions:
        File extensions to consider (e.g. ``[".bvh", ".npz"]``).
        ``None`` means all files.
    recursive:
        Scan subdirectories recursively.

    Returns
    -------
    dict
        Summary with keys ``kept``, ``rejected``, ``total``.
    """
    if mode not in ("symlink", "copy"):
        raise ValueError(f"mode must be 'symlink' or 'copy', got {mode!r}")

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    results_dir = output_dir / "results"

    if recursive:
        candidates = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in input_dir.iterdir() if p.is_file()]

    if extensions:
        ext_set = {e if e.startswith(".") else f".{e}" for e in extensions}
        candidates = [p for p in candidates if p.suffix.lower() in ext_set]

    candidates.sort()

    kept_paths: list[Path] = []
    rejected_paths: list[Path] = []

    for src in candidates:
        if filter_fn(src):
            kept_paths.append(src)
            rel = src.relative_to(input_dir)
            dst = results_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "symlink":
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
        else:
            rejected_paths.append(src)

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "extensions": list(extensions) if extensions else None,
        "recursive": recursive,
        "filter": (filter_fn.to_dict() if hasattr(filter_fn, "to_dict")
                   else {"type": filter_fn.__class__.__name__}),
        "git": _git_status(),
        "summary": {
            "total": len(candidates),
            "kept": len(kept_paths),
            "rejected": len(rejected_paths),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "filtering_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    return config["summary"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter a motion dataset by filename heuristics.",
    )
    parser.add_argument("input_dir", help="Source dataset directory")
    parser.add_argument("output_dir", help="Destination directory")
    parser.add_argument(
        "--mode", choices=["symlink", "copy"], default="symlink",
        help="Output mode (default: symlink)",
    )
    parser.add_argument(
        "--extensions", nargs="*", default=None,
        help="File extensions to include (e.g. .bvh .npz)",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Do not scan subdirectories",
    )
    parser.add_argument(
        "--exclude-mirrored", action="store_true",
        help="Also exclude mirrored '_M' takes",
    )
    parser.add_argument(
        "--extra-exclude", nargs="*", default=None,
        help="Additional keywords to exclude",
    )
    parser.add_argument(
        "--filter", choices=["ground", "none"], default="ground",
        help="Which built-in filter to apply (default: ground)",
    )
    args = parser.parse_args()

    if args.filter == "ground":
        filt = make_ground_locomotion_filter(
            exclude_mirrored=args.exclude_mirrored,
            extra_exclude=args.extra_exclude,
        )
    else:
        filt = FilterPipeline()
        if args.exclude_mirrored:
            filt.add(ExcludeMirrored())

    summary = filter_dataset(
        args.input_dir,
        args.output_dir,
        filt,
        mode=args.mode,
        extensions=args.extensions,
        recursive=not args.no_recursive,
    )
    print(
        f"Done: {summary['kept']}/{summary['total']} kept, "
        f"{summary['rejected']} rejected"
    )
    print(f"Config saved to {Path(args.output_dir) / 'filtering_config.json'}")


if __name__ == "__main__":
    _cli()
