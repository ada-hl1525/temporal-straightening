#!/usr/bin/env python
"""One-command single-video evaluation runner for cloud smoke tests.

Edit DEFAULT_CONFIG below, then run:

    python generated_eval/run_cloud_single_eval.py

You can also override values from the command line. Example:

    python generated_eval/run_cloud_single_eval.py \
      --video_path generated_videos/toy_basic_physics/Pendulum001.mp4 \
      --video_id Pendulum001_baseline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Edit this block on the cloud server.
DEFAULT_CONFIG = {
    "video_path": "generated_videos/toy_basic_physics/Pendulum001.mp4",
    "video_id": "Pendulum001_baseline",
    "output_root": "results",
    "num_frames": 16,
    "size": 224,
    "batch_size": 16,
    "model_name": "facebook/dinov2-base",
    # Keep this for the DINOv2 baseline. Later, point it to a replacement
    # embedding script with the same CLI contract.
    "embedding_script": "generated_eval/02_extract_embeddings.py",
}


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-video evaluation end to end.")
    parser.add_argument("--video_path", default=DEFAULT_CONFIG["video_path"])
    parser.add_argument("--video_id", default=DEFAULT_CONFIG["video_id"])
    parser.add_argument("--output_root", default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--num_frames", type=int, default=DEFAULT_CONFIG["num_frames"])
    parser.add_argument("--size", type=int, default=DEFAULT_CONFIG["size"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--model_name", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--embedding_script", default=DEFAULT_CONFIG["embedding_script"])
    parser.add_argument(
        "--skip_env_check",
        action="store_true",
        help="Skip generated_eval/00_check_env.py.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print("")
    print(f"=== {name} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"ERROR: Step failed: {name}")


def main() -> None:
    args = parse_args()

    video_path = Path(args.video_path)
    output_root = Path(args.output_root)
    embedding_script = Path(args.embedding_script)

    frame_dir = output_root / "frames" / args.video_id
    embedding_path = output_root / "embeddings" / f"{args.video_id}.npy"
    metrics_path = output_root / "metrics" / f"{args.video_id}.json"
    figure_dir = output_root / "figures" / args.video_id

    print(f"Repository: {REPO_ROOT}")
    print(f"Video: {video_path}")
    print(f"Video ID: {args.video_id}")
    print(f"Output root: {output_root}")
    print(f"Embedding script: {embedding_script}")
    print(f"Model name: {args.model_name}")

    if not args.skip_env_check:
        run_step(
            "00_check_env",
            [sys.executable, "generated_eval/00_check_env.py"],
        )

    run_step(
        "01_extract_frames",
        [
            sys.executable,
            "generated_eval/01_extract_frames.py",
            "--video_path",
            str(video_path),
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
            str(embedding_script),
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
            "generated_eval/03_compute_metrics.py",
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
            "generated_eval/04_plot_metrics.py",
            "--embedding_path",
            str(embedding_path),
            "--metrics_path",
            str(metrics_path),
            "--output_dir",
            str(figure_dir),
        ],
    )

    print("")
    print("Evaluation complete.")
    print(f"Frames: {frame_dir}")
    print(f"Embeddings: {embedding_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
