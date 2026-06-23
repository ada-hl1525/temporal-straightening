#!/usr/bin/env python
"""Run the full generated video evaluation pipeline for one MP4."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import fail, require_file, safe_id


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-video evaluation.")
    parser.add_argument("--video_path", type=Path, required=True, help="Input .mp4 video.")
    parser.add_argument("--video_id", required=True, help="ID used for output folders/files.")
    parser.add_argument("--output_root", type=Path, default=Path("results"), help="Output root.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to extract.")
    parser.add_argument("--size", type=int, default=224, help="Frame resize size.")
    parser.add_argument(
        "--model_name",
        default="facebook/dinov2-base",
        help="Hugging Face DINOv2 model name.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Embedding batch size.")
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print("")
    print(f"=== {name} ===")
    print(" ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {name}. Check input paths, dependencies, CUDA availability, "
            "and whether the DINOv2 model can be downloaded or found in cache."
        )


def main() -> None:
    args = parse_args()
    require_file(args.video_path, "Video")
    if args.num_frames <= 0:
        fail("--num_frames must be positive")

    video_id = safe_id(args.video_id)
    output_root = args.output_root
    frame_dir = output_root / "frames" / video_id
    embedding_path = output_root / "embeddings" / f"{video_id}.npy"
    metrics_path = output_root / "metrics" / f"{video_id}.json"
    figure_dir = output_root / "figures" / video_id

    print(f"Video path: {args.video_path}")
    print(f"Video ID: {video_id}")
    print(f"Output root: {output_root}")

    run_step(
        "01_extract_frames",
        [
            sys.executable,
            str(SCRIPT_DIR / "01_extract_frames.py"),
            "--video_path",
            str(args.video_path),
            "--output_dir",
            str(frame_dir),
            "--num_frames",
            str(args.num_frames),
            "--size",
            str(args.size),
        ],
    )
    run_step(
        "02_extract_embeddings",
        [
            sys.executable,
            str(SCRIPT_DIR / "02_extract_embeddings.py"),
            "--frame_dir",
            str(frame_dir),
            "--output_path",
            str(embedding_path),
            "--model_name",
            args.model_name,
            "--batch_size",
            str(args.batch_size),
        ],
    )
    run_step(
        "03_compute_metrics",
        [
            sys.executable,
            str(SCRIPT_DIR / "03_compute_metrics.py"),
            "--embedding_path",
            str(embedding_path),
            "--output_path",
            str(metrics_path),
        ],
    )
    run_step(
        "04_plot_metrics",
        [
            sys.executable,
            str(SCRIPT_DIR / "04_plot_metrics.py"),
            "--embedding_path",
            str(embedding_path),
            "--metrics_path",
            str(metrics_path),
            "--output_dir",
            str(figure_dir),
        ],
    )

    print("")
    print("Single video evaluation complete.")
    print(f"Frames: {frame_dir}")
    print(f"Embeddings: {embedding_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()

