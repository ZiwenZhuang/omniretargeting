"""
Contact logits/flags IO for HER workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# [L_Ankle, L_Foot, R_Ankle, R_Foot, L_Wrist, R_Wrist]
DEFAULT_CONTACT_JOINT_IDS = np.asarray([7, 10, 8, 11, 20, 21], dtype=np.int64)


def extract_contact_logits_from_results(video_results: list[dict]) -> np.ndarray:
    logits: list[np.ndarray] = []
    for frame_data in video_results:
        if isinstance(frame_data, dict) and frame_data.get("static_conf_logits") is not None:
            conf = np.asarray(frame_data["static_conf_logits"], dtype=np.float32).reshape(-1)
            logits.append(conf)
        else:
            logits.append(np.zeros(6, dtype=np.float32))
    if not logits:
        raise ValueError("No frames found when extracting contact logits.")
    return np.stack(logits, axis=0)


def save_contact_logits_from_results(
    video_results: list[dict],
    out_path: Path,
    *,
    joint_ids: np.ndarray = DEFAULT_CONTACT_JOINT_IDS,
) -> None:
    logits = extract_contact_logits_from_results(video_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), static_conf_logits=logits, smplx_joint_ids=np.asarray(joint_ids, dtype=np.int64))


def save_contact_logits_from_pt(
    pt_path: Path,
    out_path: Path,
    *,
    joint_ids: np.ndarray = DEFAULT_CONTACT_JOINT_IDS,
) -> None:
    results = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    save_contact_logits_from_results(results, out_path, joint_ids=joint_ids)


def load_contact_flags(
    contact_npz_path: Path,
    *,
    num_frames: Optional[int] = None,
    threshold: float = 0.3,
) -> np.ndarray | None:
    if not contact_npz_path.exists():
        return None
    data = np.load(str(contact_npz_path))
    logits = data.get("static_conf_logits")
    if logits is None:
        return None
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] < 4:
        return None
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    flags = sigmoid > float(threshold)
    if num_frames is None:
        return flags

    num_frames = int(num_frames)
    if flags.shape[0] >= num_frames:
        return flags[:num_frames]
    last = flags[-1] if flags.size else np.zeros((flags.shape[1],), dtype=bool)
    pad = np.repeat(last[None, :], num_frames - flags.shape[0], axis=0)
    return np.concatenate([flags, pad], axis=0)
