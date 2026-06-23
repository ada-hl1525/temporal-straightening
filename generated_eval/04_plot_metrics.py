#!/usr/bin/env python
"""Plot trajectory metrics and 2D PCA trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from common import fail, read_json, require_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot embedding trajectory metrics.")
    parser.add_argument("--embedding_path", type=Path, required=True, help="Input .npy path.")
    parser.add_argument("--metrics_path", type=Path, required=True, help="Input metrics JSON path.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output figure directory.")
    return parser.parse_args()


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        fail("Missing sklearn. Install with: python -m pip install scikit-learn")

    if embeddings.shape[0] < 2:
        fail("Need at least 2 embeddings for PCA trajectory plot")
    return PCA(n_components=2).fit_transform(embeddings)


def main() -> None:
    args = parse_args()
    require_file(args.embedding_path, "Embedding file")
    require_file(args.metrics_path, "Metrics JSON")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        fail("Missing matplotlib. Install with: python -m pip install matplotlib")

    try:
        embeddings = np.load(args.embedding_path)
    except Exception as exc:
        fail(f"Failed to load embeddings from {args.embedding_path}: {exc}")
    metrics = read_json(args.metrics_path)

    step_distances = metrics.get("step_distances", [])
    curvature = metrics.get("curvature", [])

    step_path = args.output_dir / "step_distance.png"
    plt.figure()
    plt.plot(step_distances)
    plt.xlabel("t")
    plt.ylabel("||z[t+1] - z[t]||")
    plt.title("Step distance")
    plt.tight_layout()
    plt.savefig(step_path, dpi=160)
    plt.close()
    print(f"Saved: {step_path}")

    curvature_path = args.output_dir / "curvature.png"
    plt.figure()
    plt.plot(curvature)
    plt.xlabel("t")
    plt.ylabel("1 - cosine similarity")
    plt.title("Curvature")
    plt.tight_layout()
    plt.savefig(curvature_path, dpi=160)
    plt.close()
    print(f"Saved: {curvature_path}")

    coords = pca_2d(embeddings)
    pca_path = args.output_dir / "pca_trajectory.png"
    plt.figure()
    plt.plot(coords[:, 0], coords[:, 1], marker="o")
    plt.scatter(coords[0, 0], coords[0, 1], marker="s", label="start")
    plt.scatter(coords[-1, 0], coords[-1, 1], marker="x", label="end")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pca_path, dpi=160)
    plt.close()
    print(f"Saved: {pca_path}")


if __name__ == "__main__":
    main()

