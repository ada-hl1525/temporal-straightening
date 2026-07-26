#!/usr/bin/env python
"""Generate controlled pendulum videos with PyBullet offscreen rendering.

This script creates a small simulation dataset for evaluating whether visual
encoders produce different embedding trajectories for physically plausible and
physically implausible pendulum motion.

Cloud usage:

    python -m pip install pybullet opencv-python numpy
    python generated_simulation/generate_pendulum_dataset.py

The output is written to:

    generated_videos/pybullet_pendulum/

It contains MP4 videos plus metadata.csv. The videos can be passed directly to
the existing generated_eval batch scripts.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pybullet as p


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated_videos" / "pybullet_pendulum"


@dataclass(frozen=True)
class VideoSpec:
    scene: str
    physics_label: str
    wrong_type: str
    index: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PyBullet pendulum videos.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_variants", type=int, default=4)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--render_debug", action="store_true", help="Show PyBullet GUI instead of DIRECT.")
    return parser.parse_args()


def ensure_clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["single_pendulum", "double_pendulum"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def quaternion_from_z_axis(direction: np.ndarray) -> tuple[float, float, float, float]:
    """Return quaternion rotating the local z-axis to the given direction."""
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(dot)
    half = angle * 0.5
    quat = np.concatenate([axis * math.sin(half), [math.cos(half)]])
    return tuple(float(v) for v in quat)


def simulate_single_pendulum(
    *,
    duration: float,
    fps: int,
    seed: int,
    physics_label: str,
    wrong_type: str,
) -> dict[str, np.ndarray | float | str]:
    rng = np.random.default_rng(seed)
    frame_count = int(round(duration * fps))
    substeps = 8
    dt = 1.0 / fps / substeps
    length = 1.1
    gravity = 9.81
    damping = 0.035
    theta = float(rng.uniform(0.45, 0.8)) * rng.choice([-1.0, 1.0])
    omega = float(rng.uniform(-0.15, 0.15))
    states = []

    for frame in range(frame_count):
        for _ in range(substeps):
            current_gravity = gravity
            current_damping = damping
            if physics_label == "wrong":
                if wrong_type == "single_energy_gain":
                    current_damping = -0.045
                elif wrong_type == "single_reverse_gravity_after_half" and frame > frame_count // 2:
                    current_gravity = -gravity * 0.65
                elif wrong_type == "single_periodic_kick" and frame % max(1, fps) == 0:
                    omega += 0.9 * np.sign(theta if theta != 0 else 1.0)

            alpha = -(current_gravity / length) * math.sin(theta) - current_damping * omega
            omega += alpha * dt
            theta += omega * dt
        states.append([theta, omega])

    return {
        "states": np.asarray(states, dtype=np.float32),
        "length": length,
        "gravity": gravity,
        "damping": damping,
        "initial_state": f"theta={states[0][0]:.4f};omega={states[0][1]:.4f}",
    }


def simulate_double_pendulum(
    *,
    duration: float,
    fps: int,
    seed: int,
    physics_label: str,
    wrong_type: str,
) -> dict[str, np.ndarray | float | str]:
    rng = np.random.default_rng(seed)
    frame_count = int(round(duration * fps))
    substeps = 10
    dt = 1.0 / fps / substeps
    l1 = 0.82
    l2 = 0.82
    m1 = 1.0
    m2 = 1.0
    gravity = 9.81
    damping = 0.012
    theta1 = float(rng.uniform(0.55, 0.95)) * rng.choice([-1.0, 1.0])
    theta2 = theta1 + float(rng.uniform(-0.55, 0.55))
    omega1 = float(rng.uniform(-0.08, 0.08))
    omega2 = float(rng.uniform(-0.08, 0.08))
    states = []

    for frame in range(frame_count):
        for _ in range(substeps):
            current_gravity = gravity
            current_damping = damping
            if physics_label == "wrong":
                if wrong_type == "double_energy_gain":
                    current_damping = -0.03
                elif wrong_type == "double_reverse_gravity_after_half" and frame > frame_count // 2:
                    current_gravity = -gravity * 0.55
                elif wrong_type == "double_joint_kick" and frame % max(1, fps // 2) == 0:
                    omega2 += 0.7 * np.sign(math.sin(theta2 - theta1) or 1.0)

            delta = theta1 - theta2
            den = 2 * m1 + m2 - m2 * math.cos(2 * delta)
            alpha1 = (
                -current_gravity * (2 * m1 + m2) * math.sin(theta1)
                - m2 * current_gravity * math.sin(theta1 - 2 * theta2)
                - 2
                * math.sin(delta)
                * m2
                * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * math.cos(delta))
            ) / (l1 * den)
            alpha2 = (
                2
                * math.sin(delta)
                * (
                    omega1 * omega1 * l1 * (m1 + m2)
                    + current_gravity * (m1 + m2) * math.cos(theta1)
                    + omega2 * omega2 * l2 * m2 * math.cos(delta)
                )
            ) / (l2 * den)

            omega1 += (alpha1 - current_damping * omega1) * dt
            omega2 += (alpha2 - current_damping * omega2) * dt
            theta1 += omega1 * dt
            theta2 += omega2 * dt
        states.append([theta1, theta2, omega1, omega2])

    return {
        "states": np.asarray(states, dtype=np.float32),
        "length_1": l1,
        "length_2": l2,
        "gravity": gravity,
        "damping": damping,
        "initial_state": (
            f"theta1={states[0][0]:.4f};theta2={states[0][1]:.4f};"
            f"omega1={states[0][2]:.4f};omega2={states[0][3]:.4f}"
        ),
    }


class PendulumRenderer:
    def __init__(self, *, width: int, height: int, gui: bool) -> None:
        self.width = width
        self.height = height
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)
        self.view = p.computeViewMatrix(
            cameraEyePosition=[0.0, -4.4, 1.05],
            cameraTargetPosition=[0.0, 0.0, 0.75],
            cameraUpVector=[0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )
        self.projection = p.computeProjectionMatrixFOV(
            fov=42.0,
            aspect=float(width) / float(height),
            nearVal=0.01,
            farVal=20.0,
            physicsClientId=self.client,
        )
        self._make_world()

    def close(self) -> None:
        p.disconnect(physicsClientId=self.client)

    def _make_world(self) -> None:
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[2.4, 0.02, 0.015],
            rgbaColor=[0.78, 0.78, 0.74, 1.0],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=floor_visual,
            basePosition=[0, 0.18, 0.0],
            physicsClientId=self.client,
        )
        pivot_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.055,
            rgbaColor=[0.08, 0.08, 0.08, 1.0],
            physicsClientId=self.client,
        )
        self.pivot = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pivot_visual,
            basePosition=[0, 0, 1.62],
            physicsClientId=self.client,
        )
        self.rod_visuals = [
            p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.018,
                length=1.1,
                rgbaColor=[0.12, 0.12, 0.12, 1.0],
                physicsClientId=self.client,
            ),
            p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.018,
                length=0.82,
                rgbaColor=[0.12, 0.12, 0.12, 1.0],
                physicsClientId=self.client,
            ),
        ]
        self.bob_visuals = [
            p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.085,
                rgbaColor=[0.05, 0.32, 0.95, 1.0],
                physicsClientId=self.client,
            ),
            p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.075,
                rgbaColor=[0.92, 0.2, 0.12, 1.0],
                physicsClientId=self.client,
            ),
        ]
        self.rods = [
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.rod_visuals[0], physicsClientId=self.client),
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.rod_visuals[1], physicsClientId=self.client),
        ]
        self.bobs = [
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.bob_visuals[0], physicsClientId=self.client),
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.bob_visuals[1], physicsClientId=self.client),
        ]

    def render(self, points: list[np.ndarray]) -> np.ndarray:
        pivot = np.array([0.0, 0.0, 1.62], dtype=np.float64)
        all_points = [pivot] + points

        for i, body in enumerate(self.rods):
            if i < len(points):
                start = all_points[i]
                end = all_points[i + 1]
                midpoint = (start + end) * 0.5
                direction = end - start
                quat = quaternion_from_z_axis(direction)
                p.resetBasePositionAndOrientation(body, midpoint.tolist(), quat, physicsClientId=self.client)
            else:
                p.resetBasePositionAndOrientation(body, [0, 0, -10], [0, 0, 0, 1], physicsClientId=self.client)

        for i, body in enumerate(self.bobs):
            if i < len(points):
                p.resetBasePositionAndOrientation(body, points[i].tolist(), [0, 0, 0, 1], physicsClientId=self.client)
            else:
                p.resetBasePositionAndOrientation(body, [0, 0, -10], [0, 0, 0, 1], physicsClientId=self.client)

        _, _, rgba, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view,
            projectionMatrix=self.projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        return rgb


def single_points(theta: float, length: float) -> list[np.ndarray]:
    pivot_z = 1.62
    return [np.array([length * math.sin(theta), 0.0, pivot_z - length * math.cos(theta)], dtype=np.float64)]


def double_points(theta1: float, theta2: float, l1: float, l2: float) -> list[np.ndarray]:
    pivot_z = 1.62
    p1 = np.array([l1 * math.sin(theta1), 0.0, pivot_z - l1 * math.cos(theta1)], dtype=np.float64)
    p2 = p1 + np.array([l2 * math.sin(theta2), 0.0, -l2 * math.cos(theta2)], dtype=np.float64)
    return [p1, p2]


def write_video(path: Path, frames: Iterable[np.ndarray], *, fps: int, width: int, height: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def video_specs(num_variants: int) -> list[VideoSpec]:
    specs: list[VideoSpec] = []
    single_wrong_types = [
        "single_energy_gain",
        "single_reverse_gravity_after_half",
        "single_periodic_kick",
        "single_energy_gain",
    ]
    double_wrong_types = [
        "double_energy_gain",
        "double_reverse_gravity_after_half",
        "double_joint_kick",
        "double_energy_gain",
    ]
    for scene, wrong_types, seed_base in [
        ("single_pendulum", single_wrong_types, 1100),
        ("double_pendulum", double_wrong_types, 2100),
    ]:
        for i in range(1, num_variants + 1):
            specs.append(VideoSpec(scene, "correct", "none", i, seed_base + i))
        for i in range(1, num_variants + 1):
            wrong_type = wrong_types[(i - 1) % len(wrong_types)]
            specs.append(VideoSpec(scene, "wrong", wrong_type, i, seed_base + 100 + i))
    return specs


def render_spec(
    spec: VideoSpec,
    *,
    output_dir: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
    gui: bool,
) -> dict[str, str | int | float]:
    renderer = PendulumRenderer(width=width, height=height, gui=gui)
    try:
        if spec.scene == "single_pendulum":
            sim = simulate_single_pendulum(
                duration=duration,
                fps=fps,
                seed=spec.seed,
                physics_label=spec.physics_label,
                wrong_type=spec.wrong_type,
            )
            states = sim["states"]
            frames = (
                renderer.render(single_points(float(theta), float(sim["length"])))
                for theta, _omega in states
            )
        elif spec.scene == "double_pendulum":
            sim = simulate_double_pendulum(
                duration=duration,
                fps=fps,
                seed=spec.seed,
                physics_label=spec.physics_label,
                wrong_type=spec.wrong_type,
            )
            states = sim["states"]
            frames = (
                renderer.render(
                    double_points(float(theta1), float(theta2), float(sim["length_1"]), float(sim["length_2"]))
                )
                for theta1, theta2, _omega1, _omega2 in states
            )
        else:
            raise ValueError(f"Unknown scene: {spec.scene}")

        video_id = f"{spec.scene}_{spec.physics_label}_{spec.index:03d}"
        if spec.physics_label == "wrong":
            video_id = f"{spec.scene}_{spec.wrong_type}_{spec.index:03d}"
        filename = f"{video_id}.mp4"
        scene_dir = output_dir / spec.scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        video_path = scene_dir / filename
        write_video(video_path, frames, fps=fps, width=width, height=height)

        rel_path = video_path.relative_to(REPO_ROOT)
        return {
            "video_id": video_id,
            "dataset": "pybullet_pendulum",
            "scene": spec.scene,
            "physics_label": spec.physics_label,
            "wrong_type": spec.wrong_type,
            "filename": filename,
            "video_path": str(rel_path),
            "duration_sec": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "seed": spec.seed,
            "gravity": sim["gravity"],
            "damping": sim["damping"],
            "initial_state": sim["initial_state"],
            "notes": "procedural physics simulation rendered with PyBullet",
        }
    finally:
        renderer.close()


def write_metadata(output_dir: Path, rows: list[dict[str, str | int | float]]) -> None:
    metadata_path = output_dir / "metadata.csv"
    fieldnames = [
        "video_id",
        "dataset",
        "scene",
        "physics_label",
        "wrong_type",
        "filename",
        "video_path",
        "duration_sec",
        "fps",
        "width",
        "height",
        "seed",
        "gravity",
        "damping",
        "initial_state",
        "notes",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_dir: Path) -> None:
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# PyBullet Pendulum Dataset",
                "",
                "Controlled toy physics videos generated from pendulum equations and rendered with PyBullet.",
                "",
                "Scenes:",
                "",
                "- `single_pendulum`: one bob attached to one rod.",
                "- `double_pendulum`: two bobs attached by two rods.",
                "",
                "Labels:",
                "",
                "- `correct`: normal gravity and light damping.",
                "- `wrong`: manually injected physical inconsistency such as energy gain, reversed gravity, or velocity kicks.",
                "",
                "The `metadata.csv` file stores scene labels, wrong-physics type, seed, and video paths.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_clean_output_dir(output_dir)

    rows = []
    specs = video_specs(args.num_variants)
    print(f"Output directory: {output_dir}")
    print(f"Generating {len(specs)} videos")
    for spec in specs:
        print(f"- {spec.scene} / {spec.physics_label} / {spec.wrong_type} / {spec.index:03d}")
        row = render_spec(
            spec,
            output_dir=output_dir,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            gui=args.render_debug,
        )
        rows.append(row)

    write_metadata(output_dir, rows)
    write_readme(output_dir)
    print("")
    print("Done.")
    print(f"Metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
