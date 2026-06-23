"""Shared helpers for the generated video evaluation scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


EPS = 1e-12
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4"}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return cleaned or "video"


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        fail(f"{description} does not exist: {path}")
    if not path.is_file():
        fail(f"{description} is not a file: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.exists():
        fail(f"{description} does not exist: {path}")
    if not path.is_dir():
        fail(f"{description} is not a directory: {path}")


def list_frame_paths(frame_dir: Path) -> list[Path]:
    require_dir(frame_dir, "Frame directory")
    frames = sorted(
        path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        fail(f"No png/jpg frames found in: {frame_dir}")
    return frames


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    require_file(path, "JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_latent_metrics(embeddings: np.ndarray) -> dict:
    if embeddings.ndim != 2:
        fail(f"Expected embedding array with shape [T, D], got {embeddings.shape}")
    if embeddings.shape[0] < 2:
        fail("Need at least 2 frames to compute trajectory metrics")

    embeddings = embeddings.astype(np.float64, copy=False)
    diffs = embeddings[1:] - embeddings[:-1]
    step_distances = np.linalg.norm(diffs, axis=1)

    path_length = float(np.sum(step_distances))
    endpoint_distance = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
    straightness = endpoint_distance / max(path_length, EPS)

    if diffs.shape[0] >= 2:
        prev = diffs[:-1]
        nxt = diffs[1:]
        prev_norm = np.linalg.norm(prev, axis=1)
        next_norm = np.linalg.norm(nxt, axis=1)
        denom = np.maximum(prev_norm * next_norm, EPS)
        cosine = np.sum(prev * nxt, axis=1) / denom
        cosine = np.clip(cosine, -1.0, 1.0)
        curvature = 1.0 - cosine
    else:
        curvature = np.array([], dtype=np.float64)

    return {
        "num_frames": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "step_distances": step_distances.tolist(),
        "mean_step_distance": float(np.mean(step_distances)),
        "max_step_distance": float(np.max(step_distances)),
        "std_step_distance": float(np.std(step_distances)),
        "curvature": curvature.tolist(),
        "mean_curvature": float(np.mean(curvature)) if curvature.size else 0.0,
        "max_curvature": float(np.max(curvature)) if curvature.size else 0.0,
        "straightness": float(straightness),
        "path_length": path_length,
        "endpoint_distance": endpoint_distance,
    }


def summary_from_metrics(metrics_payload: dict) -> dict:
    summary = metrics_payload.get("summary", metrics_payload)
    return {
        "num_frames": summary.get("num_frames"),
        "embedding_dim": summary.get("embedding_dim"),
        "straightness": summary.get("straightness"),
        "mean_step_distance": summary.get("mean_step_distance"),
        "max_step_distance": summary.get("max_step_distance"),
        "std_step_distance": summary.get("std_step_distance"),
        "mean_curvature": summary.get("mean_curvature"),
        "max_curvature": summary.get("max_curvature"),
    }

