#!/usr/bin/env python
"""Compute latent trajectory metrics from frame-wise embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import compute_latent_metrics, fail, require_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute embedding trajectory metrics.")
    parser.add_argument("--embedding_path", type=Path, required=True, help="Input .npy path.")
    parser.add_argument("--output_path", type=Path, required=True, help="Output metrics JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_file(args.embedding_path, "Embedding file")

    try:
        embeddings = np.load(args.embedding_path)
    except Exception as exc:
        fail(f"Failed to load embedding .npy file {args.embedding_path}: {exc}")

    metrics = compute_latent_metrics(embeddings)
    payload = {
        "embedding_path": str(args.embedding_path),
        "summary": {
            key: value
            for key, value in metrics.items()
            if key not in {"step_distances", "curvature"}
        },
        "step_distances": metrics["step_distances"],
        "curvature": metrics["curvature"],
    }
    write_json(args.output_path, payload)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"mean_step_distance: {metrics['mean_step_distance']:.6f}")
    print(f"max_step_distance: {metrics['max_step_distance']:.6f}")
    print(f"std_step_distance: {metrics['std_step_distance']:.6f}")
    print(f"mean_curvature: {metrics['mean_curvature']:.6f}")
    print(f"max_curvature: {metrics['max_curvature']:.6f}")
    print(f"straightness: {metrics['straightness']:.6f}")
    print(f"Saved metrics to: {args.output_path}")


if __name__ == "__main__":
    main()

