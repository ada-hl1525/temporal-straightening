#!/usr/bin/env python
"""Run the toy video dataset through multiple vision encoders.

Edit ENCODER_RUNS below, then run:

    python generated_eval/run_cloud_multi_encoder_eval.py --hf_endpoint https://hf-mirror.com

Each encoder is evaluated on all videos under video_root. Results are written to
one output subdirectory per encoder.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


# Edit this block on the cloud server.
DEFAULT_CONFIG = {
    "video_root": "generated_videos/toy_basic_physics",
    "output_root": "results/toy_basic_physics_multi_encoder",
    "num_frames": 16,
    "size": 224,
    "batch_size": 16,
    # Leave empty to use the current environment. Set to https://hf-mirror.com
    # if the server cannot reach huggingface.co directly.
    "hf_endpoint": "",
}


# Add or remove encoders here. The model_name can be a Hugging Face model id or
# a local model directory. It must be loadable by transformers AutoImageProcessor
# and AutoModel, because generated_eval/02_extract_embeddings.py is used.
ENCODER_RUNS = [
    {
        "run_id": "dinov2_small",
        "model_name": "facebook/dinov2-small",
        "embedding_script": "generated_eval/02_extract_embeddings.py",
    },
    {
        "run_id": "dinov2_base",
        "model_name": "facebook/dinov2-base",
        "embedding_script": "generated_eval/02_extract_embeddings.py",
    },
]


REPO_ROOT = Path(__file__).resolve().parents[1]


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "encoder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toy video evaluation for multiple encoders.")
    parser.add_argument("--video_root", default=DEFAULT_CONFIG["video_root"])
    parser.add_argument("--output_root", default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--num_frames", type=int, default=DEFAULT_CONFIG["num_frames"])
    parser.add_argument("--size", type=int, default=DEFAULT_CONFIG["size"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--hf_endpoint", default=DEFAULT_CONFIG["hf_endpoint"])
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional run_id list to run, e.g. --only dinov2_small dinov2_base.",
    )
    return parser.parse_args()


def select_runs(run_ids: list[str] | None) -> list[dict[str, str]]:
    if not run_ids:
        return ENCODER_RUNS
    requested = set(run_ids)
    selected = [run for run in ENCODER_RUNS if run["run_id"] in requested]
    missing = sorted(requested - {run["run_id"] for run in selected})
    if missing:
        raise SystemExit(f"ERROR: Unknown encoder run_id(s): {', '.join(missing)}")
    return selected


def run_encoder(args: argparse.Namespace, run: dict[str, str], env: dict[str, str]) -> None:
    run_id = safe_id(run["run_id"])
    output_root = Path(args.output_root) / run_id
    command = [
        sys.executable,
        "generated_eval/run_batch_eval.py",
        "--video_root",
        args.video_root,
        "--output_root",
        str(output_root),
        "--num_frames",
        str(args.num_frames),
        "--size",
        str(args.size),
        "--model_name",
        run["model_name"],
        "--batch_size",
        str(args.batch_size),
    ]

    print("")
    print(f"=== Encoder: {run_id} ===")
    print(f"Model: {run['model_name']}")
    print(f"Embedding script: {run['embedding_script']}")
    print(f"Output root: {output_root}")
    print(" ".join(command))

    # run_batch_eval currently calls run_single_video_eval.py, which uses
    # generated_eval/02_extract_embeddings.py. Keep embedding_script in the config
    # so this runner has an explicit place to swap once a new embedding script is
    # added with the same CLI contract.
    if run["embedding_script"] != "generated_eval/02_extract_embeddings.py":
        raise SystemExit(
            "ERROR: run_batch_eval currently supports only "
            "generated_eval/02_extract_embeddings.py. Add a compatible batch path "
            "before using a different embedding_script."
        )

    result = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"ERROR: Encoder run failed: {run_id}")

    print(f"Summary: {output_root / 'metrics_summary.csv'}")


def main() -> None:
    args = parse_args()
    runs = select_runs(args.only)
    env = os.environ.copy()
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint

    print(f"Repository: {REPO_ROOT}")
    print(f"Video root: {args.video_root}")
    print(f"Output root: {args.output_root}")
    print(f"Encoders: {', '.join(run['run_id'] for run in runs)}")
    if args.hf_endpoint:
        print(f"HF_ENDPOINT: {args.hf_endpoint}")

    for run in runs:
        run_encoder(args, run, env)

    print("")
    print("Multi-encoder evaluation complete.")
    print(f"Results root: {args.output_root}")


if __name__ == "__main__":
    main()
