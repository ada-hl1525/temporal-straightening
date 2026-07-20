#!/usr/bin/env python
"""One-command batch evaluation runner for the toy video dataset.

Edit DEFAULT_CONFIG below, then run:

    python generated_eval/run_cloud_batch_eval.py

If the cloud server needs a Hugging Face mirror:

    python generated_eval/run_cloud_batch_eval.py --hf_endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Edit this block on the cloud server.
DEFAULT_CONFIG = {
    "video_root": "generated_videos/toy_basic_physics",
    "output_root": "results/toy_basic_physics_baseline",
    "num_frames": 16,
    "size": 224,
    "batch_size": 16,
    "model_name": "facebook/dinov2-base",
    # Leave empty to use the current environment. Set to https://hf-mirror.com
    # if the server cannot reach huggingface.co directly.
    "hf_endpoint": "",
}


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toy video batch evaluation.")
    parser.add_argument("--video_root", default=DEFAULT_CONFIG["video_root"])
    parser.add_argument("--output_root", default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--num_frames", type=int, default=DEFAULT_CONFIG["num_frames"])
    parser.add_argument("--size", type=int, default=DEFAULT_CONFIG["size"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--model_name", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--hf_endpoint", default=DEFAULT_CONFIG["hf_endpoint"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint

    command = [
        sys.executable,
        "generated_eval/run_batch_eval.py",
        "--video_root",
        args.video_root,
        "--output_root",
        args.output_root,
        "--num_frames",
        str(args.num_frames),
        "--size",
        str(args.size),
        "--model_name",
        args.model_name,
        "--batch_size",
        str(args.batch_size),
    ]

    print(f"Repository: {REPO_ROOT}")
    print(f"Video root: {args.video_root}")
    print(f"Output root: {args.output_root}")
    print(f"Model name: {args.model_name}")
    if args.hf_endpoint:
        print(f"HF_ENDPOINT: {args.hf_endpoint}")
    print("")
    print(" ".join(command))

    result = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit("ERROR: Batch evaluation failed")

    summary_path = Path(args.output_root) / "metrics_summary.csv"
    print("")
    print("Batch evaluation complete.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
