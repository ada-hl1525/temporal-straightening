#!/usr/bin/env python
"""Run generated video evaluation for all MP4 files under a root directory."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from common import read_json, require_dir, safe_id, summary_from_metrics


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch video evaluation.")
    parser.add_argument("--video_root", type=Path, required=True, help="Root directory of videos.")
    parser.add_argument("--output_root", type=Path, default=Path("results"), help="Output root.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video.")
    parser.add_argument("--size", type=int, default=224, help="Frame resize size.")
    parser.add_argument(
        "--model_name",
        default="facebook/dinov2-base",
        help="Hugging Face DINOv2 model name.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Embedding batch size.")
    return parser.parse_args()


def make_video_id(video_root: Path, video_path: Path) -> str:
    relative = video_path.relative_to(video_root).with_suffix("")
    return safe_id("__".join(relative.parts))


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "scene_type",
        "video_path",
        "status",
        "error",
        "num_frames",
        "embedding_dim",
        "straightness",
        "mean_step_distance",
        "max_step_distance",
        "std_step_distance",
        "mean_curvature",
        "max_curvature",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    require_dir(args.video_root, "Video root")
    videos = sorted(args.video_root.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"ERROR: No .mp4 files found under {args.video_root}")

    rows: list[dict] = []
    print(f"Found {len(videos)} videos under {args.video_root}")

    for index, video_path in enumerate(videos, start=1):
        scene_type = safe_id(video_path.parent.name)
        video_id = make_video_id(args.video_root, video_path)
        metrics_path = args.output_root / "metrics" / f"{video_id}.json"

        print("")
        print(f"[{index}/{len(videos)}] {video_path}")
        print(f"scene_type: {scene_type}")
        print(f"video_id: {video_id}")

        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_single_video_eval.py"),
            "--video_path",
            str(video_path),
            "--video_id",
            video_id,
            "--output_root",
            str(args.output_root),
            "--num_frames",
            str(args.num_frames),
            "--size",
            str(args.size),
            "--model_name",
            args.model_name,
            "--batch_size",
            str(args.batch_size),
        ]
        result = subprocess.run(command)

        row = {
            "video_id": video_id,
            "scene_type": scene_type,
            "video_path": str(video_path),
        }
        if result.returncode == 0:
            try:
                metrics_payload = read_json(metrics_path)
                row.update(summary_from_metrics(metrics_payload))
                row["status"] = "ok"
                row["error"] = ""
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"Could not read metrics JSON: {exc}"
        else:
            row["status"] = "failed"
            row["error"] = (
                "Pipeline failed. Check input video, dependencies, CUDA, and model download/cache."
            )

        rows.append(row)

    summary_path = args.output_root / "metrics_summary.csv"
    write_summary(summary_path, rows)
    print("")
    print(f"Saved batch summary: {summary_path}")
    print("Batch evaluation complete.")


if __name__ == "__main__":
    main()

