#!/usr/bin/env python
"""Run qualitative feature/attention visualisations for report examples.

This script calls `visualize_feature_attention.py` for a small set of selected
videos and encoders. It is intentionally small enough to run on the cloud after
the quantitative evaluation is complete.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_SCRIPT = REPO_ROOT / "generated_analysis" / "visualize_feature_attention.py"


DEFAULT_MODELS = [
    ("dinov2_small", "facebook/dinov2-small"),
    ("clip_vit_base_patch32", "openai/clip-vit-base-patch32"),
    ("siglip_base_patch16_224", "google/siglip-base-patch16-224"),
]

DEFAULT_VIDEOS = [
    (
        "single_correct",
        "generated_videos/simulated_pendulum/single_pendulum/single_pendulum_correct_001.mp4",
    ),
    (
        "single_periodic_kick",
        "generated_videos/simulated_pendulum/single_pendulum/single_pendulum_single_periodic_kick_003.mp4",
    ),
    (
        "double_correct",
        "generated_videos/simulated_pendulum/double_pendulum/double_pendulum_correct_001.mp4",
    ),
    (
        "double_reverse_gravity",
        "generated_videos/simulated_pendulum/double_pendulum/double_pendulum_double_reverse_gravity_after_half_002.mp4",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected feature/attention examples.")
    parser.add_argument("--output_root", type=Path, default=Path("results/feature_attention_examples"))
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument(
        "--only_models",
        nargs="*",
        default=None,
        help="Optional model aliases: dinov2_small clip_vit_base_patch32 siglip_base_patch16_224",
    )
    parser.add_argument(
        "--only_videos",
        nargs="*",
        default=None,
        help="Optional video aliases: single_correct single_periodic_kick double_correct double_reverse_gravity",
    )
    parser.add_argument("--package_name", default="feature_attention_examples.tar.gz")
    return parser.parse_args()


def select_pairs(args: argparse.Namespace):
    models = DEFAULT_MODELS
    videos = DEFAULT_VIDEOS
    if args.only_models:
        requested = set(args.only_models)
        models = [item for item in models if item[0] in requested]
        missing = requested - {item[0] for item in models}
        if missing:
            raise SystemExit(f"ERROR: unknown model aliases: {', '.join(sorted(missing))}")
    if args.only_videos:
        requested = set(args.only_videos)
        videos = [item for item in videos if item[0] in requested]
        missing = requested - {item[0] for item in videos}
        if missing:
            raise SystemExit(f"ERROR: unknown video aliases: {', '.join(sorted(missing))}")
    return models, videos


def run_example(model_alias: str, model_name: str, video_alias: str, video_path: str, args: argparse.Namespace) -> None:
    output_dir = args.output_root / model_alias / video_alias
    command = [
        sys.executable,
        str(VIS_SCRIPT),
        "--video_path",
        video_path,
        "--model_name",
        model_name,
        "--output_dir",
        str(output_dir),
        "--num_frames",
        str(args.num_frames),
        "--size",
        str(args.size),
    ]
    if args.device:
        command.extend(["--device", args.device])

    print("")
    print(f"=== {model_alias} / {video_alias} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"ERROR: feature/attention example failed: {model_alias} / {video_alias}")


def write_index(output_root: Path, models, videos) -> None:
    lines = [
        "# Feature And Attention Examples",
        "",
        "This directory contains qualitative visualisations for selected simulated pendulum videos.",
        "",
        "For each model/video pair:",
        "",
        "- `frames/`: sampled input frames.",
        "- `feature_change/`: frame-to-frame patch feature-change overlays.",
        "- `attention/`: CLS-to-patch attention overlays when available.",
        "",
        "Models:",
        "",
    ]
    for alias, model_name in models:
        lines.append(f"- `{alias}`: `{model_name}`")
    lines.extend(["", "Videos:", ""])
    for alias, video_path in videos:
        lines.append(f"- `{alias}`: `{video_path}`")
    lines.append("")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def package_outputs(output_root: Path, package_name: str) -> Path:
    package_path = output_root / package_name
    with tarfile.open(package_path, "w:gz") as tar:
        for path in sorted(output_root.rglob("*")):
            if path == package_path or path.is_dir():
                continue
            if any(part.startswith(".") for part in path.relative_to(output_root).parts):
                continue
            tar.add(path, arcname=path.relative_to(output_root))
    return package_path


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    models, videos = select_pairs(args)
    for model_alias, model_name in models:
        for video_alias, video_path in videos:
            run_example(model_alias, model_name, video_alias, video_path, args)
    write_index(args.output_root, models, videos)
    package_path = package_outputs(args.output_root, args.package_name)
    print("")
    print("Feature/attention examples complete.")
    print(f"Output root: {args.output_root}")
    print(f"Package: {package_path}")


if __name__ == "__main__":
    main()
