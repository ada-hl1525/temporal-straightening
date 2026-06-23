#!/usr/bin/env python
"""Uniformly extract frames from one MP4 video."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from common import fail, require_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uniformly extract PNG frames from a video.")
    parser.add_argument("--video_path", type=Path, required=True, help="Input .mp4 video path.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for PNG frames.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to extract.")
    parser.add_argument("--size", type=int, default=224, help="Output frame size, e.g. 224.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0:
        fail("--num_frames must be positive")
    if args.size <= 0:
        fail("--size must be positive")
    require_file(args.video_path, "Video")
    if args.video_path.suffix.lower() != ".mp4":
        fail(f"Expected an .mp4 file, got: {args.video_path}")

    try:
        import cv2
    except ImportError as exc:
        fail(
            "Could not import cv2. Install opencv-python with: "
            "python -m pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        fail(f"Could not open video. Check the path and codec: {args.video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        fail(f"Could not read frame count from video: {args.video_path}")

    selected_count = min(args.num_frames, total_frames)
    if selected_count < args.num_frames:
        print(
            f"[WARN] Requested {args.num_frames} frames but video only has "
            f"{total_frames}; extracting {selected_count} frames."
        )

    indices = np.linspace(0, total_frames - 1, selected_count, dtype=np.int64)
    indices = np.unique(indices)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video path: {args.video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Extracted frame indices: {indices.tolist()}")
    print(f"Output directory: {args.output_dir}")

    saved = 0
    for output_index, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            fail(f"Failed to read frame index {int(frame_index)} from {args.video_path}")

        frame = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA)
        out_path = args.output_dir / f"frame_{output_index:04d}.png"
        if not cv2.imwrite(str(out_path), frame):
            cap.release()
            fail(f"Failed to write frame: {out_path}")
        saved += 1

    cap.release()
    print(f"Saved {saved} frames as PNG files.")


if __name__ == "__main__":
    main()

