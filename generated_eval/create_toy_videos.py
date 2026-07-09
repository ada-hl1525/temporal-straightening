#!/usr/bin/env python
"""Create a toy video dataset for latent trajectory evaluation.

The dataset contains simple ball/shape videos with either smooth motion or
controlled physical discontinuities. It is intended for testing whether
frame-wise visual embedding trajectory metrics can separate smooth dynamics from
teleportation, shape instability, and unexplained direction changes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


CATEGORIES = {
    "smooth_motion": {
        "prefix": "smooth",
        "failure_type": "none",
        "description": "A coloured ball moves smoothly from left to right.",
    },
    "teleportation": {
        "prefix": "teleport",
        "failure_type": "object_teleportation",
        "description": "A coloured ball moves smoothly, then abruptly jumps to a distant position.",
    },
    "shape_instability": {
        "prefix": "shape",
        "failure_type": "shape_or_identity_instability",
        "description": "A moving object abruptly changes size, shape, and colour for several frames.",
    },
    "direction_change": {
        "prefix": "direction",
        "failure_type": "unexplained_direction_change",
        "description": "A coloured ball moves right, then abruptly changes direction.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a toy MP4 dataset for latent trajectory evaluation."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("toy_videos"))
    parser.add_argument("--num_videos_per_class", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_videos_per_class <= 0:
        raise SystemExit("ERROR: --num_videos_per_class must be positive")
    if args.num_frames < 4:
        raise SystemExit("ERROR: --num_frames must be at least 4")
    if args.size < 64:
        raise SystemExit("ERROR: --size must be at least 64")
    if args.fps <= 0:
        raise SystemExit("ERROR: --fps must be positive")


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: Could not import cv2. Install OpenCV with:\n"
            "  python -m pip install opencv-python-headless\n"
            "or:\n"
            "  python -m pip install opencv-python"
        ) from exc
    return cv2


def sample_style(rng: np.random.Generator, size: int) -> dict:
    radius = int(rng.integers(max(9, size // 18), max(12, size // 12) + 1))
    margin = radius + 12
    y_base = int(rng.integers(margin + 10, size - margin - 10))
    y_amp = float(rng.uniform(0.0, 6.0))
    y_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    color = tuple(int(v) for v in rng.integers(40, 230, size=3))
    background = int(rng.integers(238, 256))
    return {
        "radius": radius,
        "margin": margin,
        "y_base": y_base,
        "y_amp": y_amp,
        "y_phase": y_phase,
        "color": color,
        "background": background,
    }


def smooth_positions(
    num_frames: int,
    size: int,
    radius: int,
    y_base: int,
    y_amp: float,
    y_phase: float,
    speed_scale: float,
) -> np.ndarray:
    margin = radius + 12
    x_start = margin
    x_end = size - margin
    x = np.linspace(x_start, x_end, num_frames) * speed_scale
    x = x - x[0] + x_start
    x = np.clip(x, margin, size - margin)

    t = np.linspace(0.0, 1.0, num_frames)
    y = y_base + y_amp * np.sin(2.0 * np.pi * t + y_phase)
    y = np.clip(y, margin, size - margin)
    return np.stack([x, y], axis=1)


def make_smooth_motion(rng: np.random.Generator, num_frames: int, size: int) -> tuple[list[np.ndarray], dict]:
    style = sample_style(rng, size)
    speed_scale = float(rng.uniform(0.92, 1.08))
    positions = smooth_positions(
        num_frames,
        size,
        style["radius"],
        style["y_base"],
        style["y_amp"],
        style["y_phase"],
        speed_scale,
    )
    return render_sequence(positions, style, "circle", size), style


def make_teleportation(rng: np.random.Generator, num_frames: int, size: int) -> tuple[list[np.ndarray], dict]:
    style = sample_style(rng, size)
    positions = smooth_positions(
        num_frames,
        size,
        style["radius"],
        style["y_base"],
        style["y_amp"],
        style["y_phase"],
        float(rng.uniform(0.95, 1.05)),
    )

    jump_frame = num_frames // 2
    jump = np.array(
        [
            rng.uniform(size * 0.25, size * 0.45),
            rng.choice([-1.0, 1.0]) * rng.uniform(size * 0.25, size * 0.35),
        ]
    )
    shifted = positions[jump_frame:] + jump
    margin = style["margin"]
    shifted[:, 0] = np.clip(shifted[:, 0], margin, size - margin)
    shifted[:, 1] = np.clip(shifted[:, 1], margin, size - margin)
    positions[jump_frame:] = shifted
    return render_sequence(positions, style, "circle", size), style


def make_shape_instability(rng: np.random.Generator, num_frames: int, size: int) -> tuple[list[np.ndarray], dict]:
    style = sample_style(rng, size)
    positions = smooth_positions(
        num_frames,
        size,
        style["radius"],
        style["y_base"],
        style["y_amp"],
        style["y_phase"],
        float(rng.uniform(0.95, 1.05)),
    )

    unstable_start = max(1, num_frames // 2 - 2)
    unstable_end = min(num_frames, unstable_start + 5)
    unstable_color = tuple(int(v) for v in rng.integers(40, 230, size=3))
    frames: list[np.ndarray] = []
    for frame_idx, position in enumerate(positions):
        frame = make_background(size, style["background"])
        if unstable_start <= frame_idx < unstable_end:
            scale = float(rng.choice([0.55, 1.65]))
            radius = max(5, int(style["radius"] * scale))
            if frame_idx % 2 == 0:
                draw_circle(frame, position, radius, unstable_color)
            else:
                draw_square(frame, position, radius, unstable_color)
        else:
            draw_circle(frame, position, style["radius"], style["color"])
        frames.append(frame)
    return frames, style


def make_direction_change(rng: np.random.Generator, num_frames: int, size: int) -> tuple[list[np.ndarray], dict]:
    style = sample_style(rng, size)
    radius = style["radius"]
    margin = style["margin"]
    turn_frame = num_frames // 2

    start = np.array([margin, style["y_base"]], dtype=np.float64)
    mid = np.array([size * rng.uniform(0.55, 0.68), style["y_base"]], dtype=np.float64)
    direction = rng.choice(["left_up", "down"])
    if direction == "left_up":
        end = np.array(
            [
                max(margin, mid[0] - size * rng.uniform(0.25, 0.38)),
                max(margin, mid[1] - size * rng.uniform(0.22, 0.35)),
            ],
            dtype=np.float64,
        )
    else:
        end = np.array(
            [
                min(size - margin, mid[0] + size * rng.uniform(0.04, 0.12)),
                min(size - margin, mid[1] + size * rng.uniform(0.25, 0.38)),
            ],
            dtype=np.float64,
        )

    first = np.linspace(start, mid, turn_frame, endpoint=False)
    second = np.linspace(mid, end, num_frames - turn_frame)
    positions = np.concatenate([first, second], axis=0)
    t = np.linspace(0.0, 1.0, num_frames)
    positions[:, 1] += style["y_amp"] * np.sin(2.0 * np.pi * t + style["y_phase"])
    positions[:, 0] = np.clip(positions[:, 0], margin, size - margin)
    positions[:, 1] = np.clip(positions[:, 1], margin, size - margin)
    return render_sequence(positions, style, "circle", size), style


def make_background(size: int, background_value: int) -> np.ndarray:
    return np.full((size, size, 3), background_value, dtype=np.uint8)


def draw_circle(frame: np.ndarray, position: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    cv2 = require_cv2()
    center = (int(round(position[0])), int(round(position[1])))
    cv2.circle(frame, center, int(radius), color, thickness=-1, lineType=cv2.LINE_AA)


def draw_square(frame: np.ndarray, position: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    cv2 = require_cv2()
    cx, cy = int(round(position[0])), int(round(position[1]))
    top_left = (cx - radius, cy - radius)
    bottom_right = (cx + radius, cy + radius)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness=-1, lineType=cv2.LINE_AA)


def render_sequence(
    positions: np.ndarray,
    style: dict,
    shape: str,
    size: int,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for position in positions:
        frame = make_background(size, style["background"])
        if shape == "circle":
            draw_circle(frame, position, style["radius"], style["color"])
        elif shape == "square":
            draw_square(frame, position, style["radius"], style["color"])
        else:
            raise ValueError(f"Unknown shape: {shape}")
        frames.append(frame)
    return frames


def write_mp4(video_path: Path, frames: list[np.ndarray], fps: int, size: int) -> None:
    cv2 = require_cv2()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (size, size))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {video_path}. "
            "Check OpenCV video codec support."
        )
    for frame in frames:
        writer.write(frame)
    writer.release()


def generate_category_video(
    category: str,
    rng: np.random.Generator,
    num_frames: int,
    size: int,
) -> list[np.ndarray]:
    if category == "smooth_motion":
        frames, _ = make_smooth_motion(rng, num_frames, size)
    elif category == "teleportation":
        frames, _ = make_teleportation(rng, num_frames, size)
    elif category == "shape_instability":
        frames, _ = make_shape_instability(rng, num_frames, size)
    elif category == "direction_change":
        frames, _ = make_direction_change(rng, num_frames, size)
    else:
        raise ValueError(f"Unknown category: {category}")
    return frames


def write_metadata(output_dir: Path, rows: list[dict]) -> Path:
    metadata_path = output_dir / "metadata.csv"
    fieldnames = [
        "video_id",
        "category",
        "video_path",
        "num_frames",
        "fps",
        "width",
        "height",
        "failure_type",
        "description",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_cv2()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []
    for category, category_info in CATEGORIES.items():
        category_dir = args.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        prefix = category_info["prefix"]

        for index in range(1, args.num_videos_per_class + 1):
            video_id = f"{prefix}_{index:03d}"
            video_path = category_dir / f"{video_id}.mp4"
            frames = generate_category_video(category, rng, args.num_frames, args.size)
            write_mp4(video_path, frames, args.fps, args.size)
            print(f"Generated: {video_path}")

            metadata_rows.append(
                {
                    "video_id": video_id,
                    "category": category,
                    "video_path": str(video_path),
                    "num_frames": args.num_frames,
                    "fps": args.fps,
                    "width": args.size,
                    "height": args.size,
                    "failure_type": category_info["failure_type"],
                    "description": category_info["description"],
                }
            )

    metadata_path = write_metadata(args.output_dir, metadata_rows)
    print(f"Generated total videos: {len(metadata_rows)}")
    print(f"Metadata CSV: {metadata_path}")


if __name__ == "__main__":
    main()

