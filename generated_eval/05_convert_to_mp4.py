#!/usr/bin/env python
"""Convert video files to MP4 using ffmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import fail, require_file


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one video file, or a folder of videos, to MP4."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Input video file or directory containing videos.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output .mp4 path for single-file conversion.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for batch conversion.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input directory for video files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional output FPS. Leave unset to keep the source timing.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality setting. Lower is higher quality. Default: 18.",
    )
    return parser.parse_args()


def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        fail(
            "ffmpeg was not found. Install it first, for example:\n"
            "  apt-get update && apt-get install -y ffmpeg\n"
            "or with conda:\n"
            "  conda install -c conda-forge ffmpeg"
        )
    return ffmpeg


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    videos = sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        fail(f"No supported video files found in: {input_dir}")
    return videos


def output_for_video(video_path: Path, input_root: Path, output_dir: Path) -> Path:
    relative = video_path.relative_to(input_root).with_suffix(".mp4")
    return output_dir / relative


def convert_video(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
    fps: float | None,
    crf: int,
) -> None:
    require_file(input_path, "Input video")
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        fail(f"Unsupported video format: {input_path}")
    if output_path.suffix.lower() != ".mp4":
        fail(f"Output path must end with .mp4: {output_path}")

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Output exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
    ]
    if fps is not None:
        if fps <= 0:
            fail("--fps must be positive when provided")
        command.extend(["-r", str(fps)])

    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]
    )

    print(f"[CONVERT] {input_path} -> {output_path}")
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
        fail(
            f"ffmpeg failed for {input_path}.\n"
            f"Check the input video codec/path and ffmpeg installation.\n"
            f"ffmpeg error:\n{message}"
        )

    print(f"[OK] Saved MP4: {output_path}")


def main() -> None:
    args = parse_args()
    if args.crf < 0 or args.crf > 51:
        fail("--crf must be between 0 and 51")

    ffmpeg = check_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg}")

    if args.input_path.is_file():
        if args.output_path is None:
            default_output = args.input_path.with_suffix(".mp4")
            if default_output == args.input_path:
                default_output = args.input_path.with_name(f"{args.input_path.stem}_converted.mp4")
            output_path = default_output
        else:
            output_path = args.output_path
        convert_video(ffmpeg, args.input_path, output_path, args.overwrite, args.fps, args.crf)
        return

    if args.input_path.is_dir():
        if args.output_path is not None:
            fail("--output_path is only valid when --input_path is a single file")
        if args.output_dir is None:
            fail("--output_dir is required when --input_path is a directory")

        videos = find_videos(args.input_path, args.recursive)
        print(f"Found {len(videos)} video files.")
        for index, video_path in enumerate(videos, start=1):
            output_path = output_for_video(video_path, args.input_path, args.output_dir)
            print(f"[{index}/{len(videos)}]")
            convert_video(ffmpeg, video_path, output_path, args.overwrite, args.fps, args.crf)
        print("Batch conversion complete.")
        return

    fail(f"Input path does not exist: {args.input_path}")


if __name__ == "__main__":
    main()

