#!/usr/bin/env python
"""Generate controlled collision and occlusion videos for the project-plan extension.

The pendulum benchmark covers gravity-driven dynamics. This script adds two
small controlled 2D scenes from the original project plan:

1. collision-induced motion change;
2. object permanence under occlusion.

The videos are intentionally simple and fully deterministic from metadata.
They are designed for representation/evaluation experiments, not photorealism.

Cloud usage:

    python -m pip install -r generated_simulation/requirements.txt
    python generated_simulation/generate_plan_extension_dataset.py

Then evaluate the output folder with the existing multi-encoder runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated_videos" / "simulated_plan_extensions"


@dataclass(frozen=True)
class VideoSpec:
    scene: str
    physics_label: str
    wrong_type: str
    index: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled collision and occlusion videos.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_variants", type=int, default=6)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene in ["collision", "occlusion"]:
        (output_dir / scene).mkdir(parents=True, exist_ok=True)


def make_specs(num_variants: int) -> list[VideoSpec]:
    wrong_types = {
        "collision": [
            "collision_no_response",
            "collision_wrong_direction",
            "collision_energy_gain",
        ],
        "occlusion": [
            "occlusion_disappear",
            "occlusion_wrong_reappearance",
            "occlusion_identity_change",
        ],
    }
    specs: list[VideoSpec] = []
    seed = 9000
    for scene in ["collision", "occlusion"]:
        for index in range(1, num_variants + 1):
            specs.append(VideoSpec(scene, "correct", "none", index, seed))
            seed += 1
        for index in range(1, num_variants + 1):
            wrong_type = wrong_types[scene][(index - 1) % len(wrong_types[scene])]
            specs.append(VideoSpec(scene, "wrong", wrong_type, index, seed))
            seed += 1
    return specs


def draw_background(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    frame[:] = (248, 248, 245)
    floor_y = int(h * 0.72)
    cv2.rectangle(frame, (0, floor_y), (w, h), (225, 225, 218), thickness=-1)
    for x in range(-60, w + 80, 64):
        cv2.line(frame, (x, floor_y), (x + 75, h), (214, 214, 207), 1, cv2.LINE_AA)
    cv2.line(frame, (0, floor_y), (w, floor_y), (188, 188, 182), 2, cv2.LINE_AA)


def draw_ball(frame: np.ndarray, center: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.circle(frame, (x + 3, y + 4), radius, (176, 176, 170), -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x - radius // 3, y - radius // 3), max(2, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)


def draw_motion_trace(frame: np.ndarray, points: list[np.ndarray], color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        alpha = i / len(points)
        c = tuple(int((1 - alpha) * 230 + alpha * v) for v in color)
        cv2.line(
            frame,
            tuple(np.round(points[i - 1]).astype(int)),
            tuple(np.round(points[i]).astype(int)),
            c,
            2,
            cv2.LINE_AA,
        )


def collision_positions(
    *,
    frame_count: int,
    width: int,
    height: int,
    seed: int,
    physics_label: str,
    wrong_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    rng = np.random.default_rng(seed)
    radius = int(round(width * 0.035))
    floor_y = int(height * 0.72)
    y = float(floor_y - radius)
    x1_start = float(width * rng.uniform(0.13, 0.18))
    x2_start = float(width * rng.uniform(0.82, 0.87))
    impact_frame = frame_count // 2
    contact_midpoint = float(width * rng.uniform(0.48, 0.52))
    x1_contact = contact_midpoint - radius
    x2_contact = contact_midpoint + radius
    post_span = width * rng.uniform(0.25, 0.34)

    p1s: list[np.ndarray] = []
    p2s: list[np.ndarray] = []

    for frame_idx in range(frame_count):
        if frame_idx <= impact_frame:
            alpha = frame_idx / max(1, impact_frame)
            x1 = (1.0 - alpha) * x1_start + alpha * x1_contact
            x2 = (1.0 - alpha) * x2_start + alpha * x2_contact
            y1 = y
            y2 = y
        else:
            beta = (frame_idx - impact_frame) / max(1, frame_count - 1 - impact_frame)
            if physics_label == "correct":
                x1 = x1_contact - post_span * beta
                x2 = x2_contact + post_span * beta
                y1 = y
                y2 = y
            elif wrong_type == "collision_no_response":
                # Wrong but visually clean: the balls stop at contact instead
                # of bouncing away. They remain tangent, never overlapping.
                x1 = x1_contact
                x2 = x2_contact
                y1 = y
                y2 = y
            elif wrong_type == "collision_wrong_direction":
                # Both balls move upward after contact, violating the expected
                # horizontal bounce while staying separated.
                x1 = x1_contact - post_span * 0.55 * beta
                x2 = x2_contact + post_span * 0.55 * beta
                lift = height * 0.18 * beta
                y1 = y - lift
                y2 = y - lift
            elif wrong_type == "collision_energy_gain":
                x1 = x1_contact - post_span * 1.55 * beta
                x2 = x2_contact + post_span * 1.55 * beta
                y1 = y
                y2 = y
            else:
                x1 = x1_contact - post_span * beta
                x2 = x2_contact + post_span * beta
                y1 = y
                y2 = y
        p1s.append(np.array([x1, y1], dtype=np.float64))
        p2s.append(np.array([x2, y2], dtype=np.float64))

    meta = {
        "event_frame": str(impact_frame),
        "event_description": "two balls meet and should exchange horizontal velocities",
        "object_radius": str(radius),
        "minimum_center_distance": str(2 * radius),
    }
    p1_array = np.asarray(p1s)
    p2_array = np.asarray(p2s)
    min_distance = float(np.min(np.linalg.norm(p2_array - p1_array, axis=1)))
    if min_distance < (2 * radius - 1e-6):
        raise RuntimeError(
            f"collision trajectory overlaps: min center distance {min_distance:.3f}, "
            f"required {2 * radius:.3f}"
        )
    meta["observed_min_center_distance"] = f"{min_distance:.3f}"
    return p1_array, p2_array, meta


def render_collision_video(
    *,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration: float,
    seed: int,
    physics_label: str,
    wrong_type: str,
) -> dict[str, str]:
    frame_count = int(round(duration * fps))
    p1s, p2s, meta = collision_positions(
        frame_count=frame_count,
        width=width,
        height=height,
        seed=seed,
        physics_label=physics_label,
        wrong_type=wrong_type,
    )
    radius = int(meta["object_radius"])
    frames = []
    for idx in range(frame_count):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        draw_background(frame)
        draw_motion_trace(frame, [p for p in p1s[max(0, idx - 10) : idx + 1]], (235, 90, 20))
        draw_motion_trace(frame, [p for p in p2s[max(0, idx - 10) : idx + 1]], (45, 68, 235))
        draw_ball(frame, p1s[idx], radius, (235, 90, 20))
        draw_ball(frame, p2s[idx], radius, (45, 68, 235))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264", quality=8)
    return meta


def occlusion_positions(
    *,
    frame_count: int,
    width: int,
    height: int,
    seed: int,
    physics_label: str,
    wrong_type: str,
) -> tuple[np.ndarray, dict[str, str]]:
    rng = np.random.default_rng(seed)
    y = float(height * rng.uniform(0.43, 0.55))
    start_x = float(width * 0.12)
    end_x = float(width * 0.88)
    x = np.linspace(start_x, end_x, frame_count)
    y_curve = y + 6.0 * np.sin(np.linspace(0, np.pi, frame_count) + rng.uniform(-0.2, 0.2))
    positions = np.stack([x, y_curve], axis=1)

    occluder_x0 = int(width * 0.42)
    occluder_x1 = int(width * 0.60)
    occluder_y0 = int(height * 0.24)
    occluder_y1 = int(height * 0.68)
    enter_frame = int(np.argmax(positions[:, 0] > occluder_x0))
    exit_frame = int(np.argmax(positions[:, 0] > occluder_x1))

    if physics_label == "wrong" and wrong_type == "occlusion_wrong_reappearance":
        positions[exit_frame:, 1] += height * 0.17
    elif physics_label == "wrong" and wrong_type == "occlusion_disappear":
        positions[exit_frame:, 0] = -9999

    meta = {
        "event_frame": str((enter_frame + exit_frame) // 2),
        "event_description": "ball passes behind occluder and should reappear with same identity and trajectory",
        "occluder": f"{occluder_x0},{occluder_y0},{occluder_x1},{occluder_y1}",
        "enter_frame": str(enter_frame),
        "exit_frame": str(exit_frame),
    }
    return positions, meta


def render_occlusion_video(
    *,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration: float,
    seed: int,
    physics_label: str,
    wrong_type: str,
) -> dict[str, str]:
    frame_count = int(round(duration * fps))
    positions, meta = occlusion_positions(
        frame_count=frame_count,
        width=width,
        height=height,
        seed=seed,
        physics_label=physics_label,
        wrong_type=wrong_type,
    )
    x0, y0, x1, y1 = [int(v) for v in meta["occluder"].split(",")]
    radius = int(round(width * 0.035))
    frames = []
    for idx in range(frame_count):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        draw_background(frame)
        trace_points = [p for p in positions[max(0, idx - 14) : idx + 1] if p[0] > -1000]
        draw_motion_trace(frame, trace_points, (235, 90, 20))

        color = (235, 90, 20)
        if physics_label == "wrong" and wrong_type == "occlusion_identity_change" and idx >= int(meta["exit_frame"]):
            color = (45, 68, 235)

        pos = positions[idx]
        visible = pos[0] > -1000
        hidden_by_occluder = x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1
        if visible and not hidden_by_occluder:
            draw_ball(frame, pos, radius, color)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (116, 106, 94), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 64, 58), 3)
        cv2.line(frame, (x0 + 10, y0 + 12), (x1 - 10, y0 + 12), (150, 140, 128), 2, cv2.LINE_AA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264", quality=8)
    return meta


def render_video(spec: VideoSpec, args: argparse.Namespace) -> tuple[Path, dict[str, str]]:
    name = f"{spec.scene}_{spec.wrong_type if spec.physics_label == 'wrong' else 'correct'}_{spec.index:03d}.mp4"
    output_path = args.output_dir / spec.scene / name
    if spec.scene == "collision":
        meta = render_collision_video(
            output_path=output_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            seed=spec.seed,
            physics_label=spec.physics_label,
            wrong_type=spec.wrong_type,
        )
    elif spec.scene == "occlusion":
        meta = render_occlusion_video(
            output_path=output_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            seed=spec.seed,
            physics_label=spec.physics_label,
            wrong_type=spec.wrong_type,
        )
    else:
        raise ValueError(f"unknown scene: {spec.scene}")
    return output_path, meta


def write_metadata(rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = [
        "video_id",
        "relative_path",
        "scene",
        "physics_label",
        "wrong_type",
        "index",
        "seed",
        "fps",
        "duration",
        "width",
        "height",
        "event_frame",
        "event_description",
        "object_radius",
        "minimum_center_distance",
        "observed_min_center_distance",
        "occluder",
        "enter_frame",
        "exit_frame",
    ]
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_readme(output_dir: Path, args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    text = f"""# Simulated Project-Plan Extension Dataset

This folder contains controlled 2D videos for two physical behaviours from the
original project plan:

- `collision`: contact-induced motion change;
- `occlusion`: object permanence under occlusion.

The videos are intentionally simple. The goal is to provide controlled correct
and wrong labels for latent trajectory evaluation, not photorealistic rendering.

Generation settings:

- videos: {len(rows)}
- resolution: {args.width} x {args.height}
- fps: {args.fps}
- duration: {args.duration}
- frames per video: {int(round(args.duration * args.fps))}

Suggested evaluation:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \\
  --video_root {output_dir.as_posix()} \\
  --output_root results/simulated_plan_extensions_multi_encoder \\
  --num_frames 16 \\
  --size 224 \\
  --hf_endpoint https://hf-mirror.com
```

For a smaller run:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \\
  --video_root {output_dir.as_posix()} \\
  --output_root results/simulated_plan_extensions_small_encoders \\
  --only dinov2_base clip_vit_base_patch32 siglip_base_patch16_224 \\
  --num_frames 16 \\
  --size 224 \\
  --hf_endpoint https://hf-mirror.com
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_dir)
    rows: list[dict[str, str]] = []
    for spec in make_specs(args.num_variants):
        output_path, event_meta = render_video(spec, args)
        video_id = output_path.stem
        row = {
            "video_id": video_id,
            "relative_path": output_path.relative_to(args.output_dir).as_posix(),
            "scene": spec.scene,
            "physics_label": spec.physics_label,
            "wrong_type": spec.wrong_type,
            "index": str(spec.index),
            "seed": str(spec.seed),
            "fps": str(args.fps),
            "duration": str(args.duration),
            "width": str(args.width),
            "height": str(args.height),
            **event_meta,
        }
        rows.append(row)
        print(f"Wrote {output_path}")
    write_metadata(rows, args.output_dir)
    write_readme(args.output_dir, args, rows)
    print("")
    print("Generated project-plan extension dataset.")
    print(f"Output: {args.output_dir}")
    print(f"Videos: {len(rows)}")
    print(f"Metadata: {args.output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
