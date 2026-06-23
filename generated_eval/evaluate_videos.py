#!/usr/bin/env python
"""Evaluate physical consistency proxies for generated videos.

This MVP treats frame-wise visual embeddings as a temporal trajectory. Smooth,
physically plausible videos should generally have smaller step discontinuities,
lower curvature, and higher straightness in that embedding space.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
EPS = 1e-12


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: opencv-python. Install with:\n"
            "  python -m pip install -r generated_eval/requirements.txt\n"
            "or:\n"
            "  python -m pip install opencv-python"
        ) from exc
    return cv2


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "video"


def iter_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise SystemExit(f"Input file is not a supported video: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    videos = sorted(
        path for path in input_path.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(f"No videos found under: {input_path}")
    return videos


def unique_video_dir(output_dir: Path, video_path: Path) -> Path:
    base = safe_stem(video_path)
    candidate = output_dir / base
    if not candidate.exists():
        return candidate

    parent_hint = safe_stem(video_path.parent)
    candidate = output_dir / f"{parent_hint}_{base}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        indexed = output_dir / f"{parent_hint}_{base}_{index:03d}"
        if not indexed.exists():
            return indexed
        index += 1


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    frame_stride: int,
    max_frames: int | None,
    image_ext: str,
) -> tuple[list[Path], dict[str, float | int | str]]:
    cv2 = require_cv2()
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_paths: list[Path] = []
    source_index = 0
    saved_index = 0
    ok, frame = cap.read()
    while ok:
        if source_index % frame_stride == 0:
            frame_path = frames_dir / f"frame_{saved_index:06d}{image_ext}"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not write frame: {frame_path}")
            frame_paths.append(frame_path)
            saved_index += 1
            if max_frames is not None and saved_index >= max_frames:
                break
        source_index += 1
        ok, frame = cap.read()

    cap.release()

    if len(frame_paths) < 2:
        raise RuntimeError(
            f"Need at least 2 extracted frames, got {len(frame_paths)} from {video_path}"
        )

    metadata: dict[str, float | int | str] = {
        "video_path": str(video_path),
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "source_width": width,
        "source_height": height,
        "frame_stride": frame_stride,
        "extracted_frames": len(frame_paths),
    }
    return frame_paths, metadata


@dataclass
class EncoderConfig:
    name: str
    weights: str
    image_size: int
    batch_size: int
    device: str
    normalize_embeddings: bool


class RawEncoder:
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.name = "raw"

    def encode(self, frame_paths: list[Path], batch_size: int) -> np.ndarray:
        cv2 = require_cv2()
        vectors: list[np.ndarray] = []
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            vector = image.astype(np.float32).reshape(-1) / 255.0
            vectors.append(vector)
        return np.stack(vectors, axis=0)


class ResNet18Encoder:
    def __init__(self, config: EncoderConfig):
        try:
            import torch
            import torch.nn as nn
            from PIL import Image
            from torchvision import models, transforms
        except ImportError as exc:
            raise SystemExit(
                "Missing torch/torchvision/Pillow for --encoder resnet18. Install with:\n"
                "  python -m pip install -r generated_eval/requirements.txt\n"
                "or use --encoder raw."
            ) from exc

        self.torch = torch
        self.image_cls = Image
        self.name = "resnet18"

        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        if config.weights == "imagenet":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
            self.transform = weights.transforms()
        elif config.weights == "none":
            model = models.resnet18(weights=None)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config.image_size, config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            raise ValueError(f"Unsupported weights mode: {config.weights}")

        model.fc = nn.Identity()
        model.eval()
        model.to(self.device)
        self.model = model

    def encode(self, frame_paths: list[Path], batch_size: int) -> np.ndarray:
        torch = self.torch
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(frame_paths), batch_size):
                batch_paths = frame_paths[start : start + batch_size]
                images = [
                    self.transform(self.image_cls.open(path).convert("RGB"))
                    for path in batch_paths
                ]
                batch = torch.stack(images, dim=0).to(self.device)
                output = self.model(batch).detach().cpu().numpy().astype(np.float32)
                embeddings.append(output)
        return np.concatenate(embeddings, axis=0)


def build_encoder(config: EncoderConfig):
    if config.name == "raw":
        return RawEncoder(config.image_size)
    if config.name == "resnet18":
        return ResNet18Encoder(config)
    raise ValueError(f"Unsupported encoder: {config.name}")


def maybe_normalize_embeddings(embeddings: np.ndarray, enabled: bool) -> np.ndarray:
    embeddings = embeddings.astype(np.float32, copy=False)
    if not enabled:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, EPS)


def summarize_array(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def compute_metrics(embeddings: np.ndarray) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [T, D], got {embeddings.shape}")
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least two embeddings to compute temporal metrics")

    diffs = embeddings[1:] - embeddings[:-1]
    steps = np.linalg.norm(diffs, axis=1)
    path_length = float(np.sum(steps))
    endpoint_distance = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
    straightness = endpoint_distance / max(path_length, EPS)

    metrics: dict[str, float | int] = {
        "num_frames": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "path_length": path_length,
        "endpoint_distance": endpoint_distance,
        "straightness": float(straightness),
    }
    metrics.update(summarize_array(steps, "step_distance"))
    metrics["step_distance_cv"] = float(np.std(steps) / max(np.mean(steps), EPS))
    metrics["discontinuity_ratio_max_to_median"] = float(
        np.max(steps) / max(np.median(steps), EPS)
    )
    metrics["discontinuity_ratio_p95_to_median"] = float(
        np.percentile(steps, 95) / max(np.median(steps), EPS)
    )

    if embeddings.shape[0] >= 3:
        second_diffs = embeddings[2:] - 2.0 * embeddings[1:-1] + embeddings[:-2]
        second_norms = np.linalg.norm(second_diffs, axis=1)

        prev = diffs[:-1]
        nxt = diffs[1:]
        prev_norm = np.linalg.norm(prev, axis=1)
        next_norm = np.linalg.norm(nxt, axis=1)
        cos = np.sum(prev * nxt, axis=1) / np.maximum(prev_norm * next_norm, EPS)
        cos = np.clip(cos, -1.0, 1.0)
        turn_angles = np.arccos(cos)
        local_scale = 0.5 * (prev_norm + next_norm)
        curvature = turn_angles / np.maximum(local_scale, EPS)

        metrics.update(summarize_array(second_norms, "second_diff"))
        metrics.update(summarize_array(turn_angles, "turn_angle_rad"))
        metrics["turn_angle_mean_deg"] = float(np.degrees(np.mean(turn_angles)))
        metrics["turn_angle_max_deg"] = float(np.degrees(np.max(turn_angles)))
        metrics.update(summarize_array(curvature, "curvature"))
    else:
        second_norms = np.zeros((0,), dtype=np.float32)
        turn_angles = np.zeros((0,), dtype=np.float32)
        curvature = np.zeros((0,), dtype=np.float32)
        metrics.update(summarize_array(second_norms, "second_diff"))
        metrics.update(summarize_array(turn_angles, "turn_angle_rad"))
        metrics["turn_angle_mean_deg"] = 0.0
        metrics["turn_angle_max_deg"] = 0.0
        metrics.update(summarize_array(curvature, "curvature"))

    series = {
        "steps": steps,
        "second_diffs": second_norms,
        "turn_angles": turn_angles,
        "curvature": curvature,
    }
    return metrics, series


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    if embeddings.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)

    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(centered).astype(np.float32)
    except ImportError:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].T
        coords = centered @ components
        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros_like(coords)], axis=1)
        return coords[:, :2].astype(np.float32)


def save_plots(
    embeddings: np.ndarray,
    series: dict[str, np.ndarray],
    plots_dir: Path,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install with:\n"
            "  python -m pip install -r generated_eval/requirements.txt"
        ) from exc

    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(series["steps"])), series["steps"], linewidth=1.8)
    plt.xlabel("Frame transition")
    plt.ylabel("Embedding distance")
    plt.title(f"{title}: step distance")
    plt.tight_layout()
    plt.savefig(plots_dir / "step_distance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(series["curvature"])), series["curvature"], linewidth=1.8)
    plt.xlabel("Frame triplet")
    plt.ylabel("Curvature proxy")
    plt.title(f"{title}: curvature")
    plt.tight_layout()
    plt.savefig(plots_dir / "curvature.png", dpi=160)
    plt.close()

    coords = pca_2d(embeddings)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.arange(coords.shape[0]),
        cmap="viridis",
        s=22,
    )
    plt.plot(coords[:, 0], coords[:, 1], linewidth=1.0, alpha=0.7)
    plt.scatter(coords[0, 0], coords[0, 1], marker="o", s=70, label="start")
    plt.scatter(coords[-1, 0], coords[-1, 1], marker="x", s=70, label="end")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{title}: 2D PCA trajectory")
    plt.colorbar(scatter, label="Frame index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_trajectory.png", dpi=160)
    plt.close()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_video(
    video_path: Path,
    video_output_dir: Path,
    encoder,
    config: EncoderConfig,
    frame_stride: int,
    max_frames: int | None,
    image_ext: str,
) -> dict[str, float | int | str]:
    frames_dir = video_output_dir / "frames"
    plots_dir = video_output_dir / "plots"
    frame_paths, video_meta = extract_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        frame_stride=frame_stride,
        max_frames=max_frames,
        image_ext=image_ext,
    )

    embeddings = encoder.encode(frame_paths, batch_size=config.batch_size)
    embeddings = maybe_normalize_embeddings(embeddings, config.normalize_embeddings)
    np.save(video_output_dir / "embeddings.npy", embeddings)

    metrics, series = compute_metrics(embeddings)
    payload = {
        "video": video_meta,
        "encoder": {
            "name": config.name,
            "weights": config.weights,
            "image_size": config.image_size,
            "normalize_embeddings": config.normalize_embeddings,
        },
        "metrics": metrics,
    }
    write_json(video_output_dir / "metrics.json", payload)
    save_plots(embeddings, series, plots_dir, title=video_path.stem)

    summary_row: dict[str, float | int | str] = {
        "video_path": str(video_path),
        "output_dir": str(video_output_dir),
        "encoder": config.name,
        "weights": config.weights,
    }
    summary_row.update(video_meta)
    summary_row.update(metrics)
    return summary_row


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated videos via temporal-straightening-inspired embedding trajectory metrics."
    )
    parser.add_argument("--input", required=True, type=Path, help="MP4 file or folder of videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_eval/outputs"),
        help="Directory where frames, embeddings, metrics, and plots are written.",
    )
    parser.add_argument(
        "--encoder",
        choices=["raw", "resnet18"],
        default="raw",
        help="Visual embedding backend. raw is dependency-light; resnet18 is semantically stronger.",
    )
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="none",
        help="Weights for resnet18. imagenet may download if not cached.",
    )
    parser.add_argument("--image-size", type=positive_int, default=64)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for torch encoders.",
    )
    parser.add_argument(
        "--frame-stride",
        type=positive_int,
        default=1,
        help="Keep every Nth source frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="Optional maximum number of extracted frames per video.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable per-frame L2 normalization before metric computation.",
    )
    parser.add_argument(
        "--image-ext",
        choices=[".jpg", ".png"],
        default=".jpg",
        help="Extracted frame image format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = iter_videos(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.encoder == "raw" and args.weights != "none":
        raise SystemExit("--weights only applies to --encoder resnet18; use --weights none with raw.")

    config = EncoderConfig(
        name=args.encoder,
        weights=args.weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize_embeddings,
    )
    encoder = build_encoder(config)

    rows: list[dict[str, float | int | str]] = []
    for index, video_path in enumerate(videos, start=1):
        video_output_dir = unique_video_dir(args.output_dir, video_path)
        video_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(videos)}] Processing {video_path}")
        row = process_video(
            video_path=video_path,
            video_output_dir=video_output_dir,
            encoder=encoder,
            config=config,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            image_ext=args.image_ext,
        )
        rows.append(row)

    summary_path = args.output_dir / "metrics_summary.csv"
    write_summary_csv(summary_path, rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
