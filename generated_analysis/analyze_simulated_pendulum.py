#!/usr/bin/env python
"""Report-oriented analysis for simulated pendulum evaluation results.

This script uses the existing evaluation outputs:

    results/<run>/<encoder>/metrics_summary.csv
    results/<run>/<encoder>/embeddings/*.npy

It produces tables, figures, a short markdown report, and a final tar.gz
package. It does not re-run encoders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tarfile
from pathlib import Path

import numpy as np


METRICS = [
    "straightness",
    "mean_curvature",
    "max_curvature",
    "std_step_distance",
    "mean_step_distance",
]
MODEL_GROUPS = {
    "DINOv2 scale": ["dinov2_small", "dinov2_base", "dinov2_large"],
    "CLIP scale": ["clip_vit_base_patch32", "clip_vit_large_patch14"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze simulated pendulum evaluation results.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results/pybullet_pendulum_multi_encoder"),
        help="Root containing one subdirectory per encoder.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("generated_videos/simulated_pendulum/metadata.csv"),
        help="Simulated pendulum metadata CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/simulated_pendulum_analysis"),
        help="Directory for analysis tables, figures, report, and package.",
    )
    parser.add_argument("--package_name", default="simulated_pendulum_analysis.tar.gz")
    parser.add_argument(
        "--latex_asset_dir",
        type=Path,
        default=None,
        help=(
            "Optional LaTeX asset directory. When set, write tables.tex, "
            "figures.tex, README.md, and copy report figures into this folder."
        ),
    )
    return parser.parse_args()


def require_matplotlib(output_dir: Path):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("ERROR: missing matplotlib. Install it with: python -m pip install matplotlib") from exc
    return plt


def safe_float(value: str) -> float:
    if value == "" or value is None:
        return math.nan
    return float(value)


def infer_scene(video_id: str) -> str:
    if video_id.startswith("single_pendulum"):
        return "single_pendulum"
    if video_id.startswith("double_pendulum"):
        return "double_pendulum"
    if "__" in video_id:
        return video_id.split("__", 1)[0]
    return "unknown"


def infer_label(video_id: str) -> str:
    return "correct" if "_correct_" in video_id else "wrong"


def infer_wrong_type(video_id: str) -> str:
    if infer_label(video_id) == "correct":
        return "none"
    raw = video_id.split("__", 1)[-1]
    scene = infer_scene(video_id)
    prefix = f"{scene}_"
    if raw.startswith(prefix):
        raw = raw[len(prefix) :]
    parts = raw.rsplit("_", 1)
    return parts[0] if parts else raw


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["video_id"]: row for row in csv.DictReader(handle)}


def canonical_video_id(video_id: str) -> str:
    return video_id.split("__", 1)[-1]


def load_rows(results_root: Path, metadata: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    summaries = sorted(results_root.glob("*/metrics_summary.csv"))
    if not summaries:
        raise SystemExit(f"ERROR: no metrics_summary.csv files found under {results_root}")

    rows: list[dict[str, object]] = []
    for summary_path in summaries:
        encoder = summary_path.parent.name
        with summary_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                video_id = row["video_id"]
                canonical_id = canonical_video_id(video_id)
                meta = metadata.get(canonical_id, {})
                scene = meta.get("scene") or infer_scene(video_id)
                label = meta.get("physics_label") or infer_label(video_id)
                wrong_type = meta.get("wrong_type") or infer_wrong_type(video_id)
                out = {
                    "encoder": encoder,
                    "video_id": video_id,
                    "canonical_video_id": canonical_id,
                    "scene": scene,
                    "label": label,
                    "wrong_type": wrong_type,
                    "status": row.get("status", ""),
                    "video_path": row.get("video_path", ""),
                    "num_frames": int(float(row["num_frames"])) if row.get("num_frames") else 0,
                    "embedding_dim": int(float(row["embedding_dim"])) if row.get("embedding_dim") else 0,
                }
                for metric in METRICS:
                    out[metric] = safe_float(row.get(metric, ""))
                rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def mean(values: list[float]) -> float:
    clean = [v for v in values if np.isfinite(v)]
    return float(np.mean(clean)) if clean else math.nan


def std(values: list[float]) -> float:
    clean = [v for v in values if np.isfinite(v)]
    return float(np.std(clean)) if clean else math.nan


def group_rows(rows: list[dict[str, object]], keys: list[str]) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def summarise_groups(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    output = []
    for key, group in sorted(group_rows(rows, keys).items()):
        out = {name: value for name, value in zip(keys, key)}
        out["n"] = len(group)
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"mean_{metric}"] = mean(values)
            out[f"std_{metric}"] = std(values)
        output.append(out)
    return output


def encoder_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for encoder in sorted({str(row["encoder"]) for row in rows}):
        correct = [row for row in rows if row["encoder"] == encoder and row["label"] == "correct"]
        wrong = [row for row in rows if row["encoder"] == encoder and row["label"] == "wrong"]
        out: dict[str, object] = {"encoder": encoder, "n_correct": len(correct), "n_wrong": len(wrong)}
        for metric in METRICS:
            out[f"correct_{metric}"] = mean([float(row[metric]) for row in correct])
            out[f"wrong_{metric}"] = mean([float(row[metric]) for row in wrong])
            out[f"delta_{metric}"] = float(out[f"wrong_{metric}"]) - float(out[f"correct_{metric}"])
        output.append(out)
    return output


def scene_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for scene in sorted({str(row["scene"]) for row in rows}):
        correct = [row for row in rows if row["scene"] == scene and row["label"] == "correct"]
        wrong = [row for row in rows if row["scene"] == scene and row["label"] == "wrong"]
        out: dict[str, object] = {"scene": scene, "n_correct": len(correct), "n_wrong": len(wrong)}
        for metric in METRICS:
            out[f"delta_{metric}"] = mean([float(row[metric]) for row in wrong]) - mean(
                [float(row[metric]) for row in correct]
            )
        output.append(out)
    return output


def auc_wrong_greater(rows: list[dict[str, object]], metric: str) -> float:
    positives = [float(row[metric]) for row in rows if row["label"] == "wrong"]
    negatives = [float(row[metric]) for row in rows if row["label"] == "correct"]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def auc_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for encoder in sorted({str(row["encoder"]) for row in rows}):
        group = [row for row in rows if row["encoder"] == encoder]
        out: dict[str, object] = {"encoder": encoder}
        for metric in METRICS:
            out[f"auc_wrong_gt_correct_{metric}"] = auc_wrong_greater(group, metric)
        output.append(out)
    return output


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def load_embedding(results_root: Path, encoder: str, video_id: str) -> np.ndarray:
    path = results_root / encoder / "embeddings" / f"{video_id}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def trajectory_distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> dict[str, float]:
    frame_count = min(embedding_a.shape[0], embedding_b.shape[0])
    a = embedding_a[:frame_count].astype(np.float64)
    b = embedding_b[:frame_count].astype(np.float64)
    frame_distances = np.linalg.norm(a - b, axis=1)
    step_a = np.linalg.norm(np.diff(a, axis=0), axis=1)
    step_b = np.linalg.norm(np.diff(b, axis=0), axis=1)
    return {
        "mean_frame_distance": float(np.mean(frame_distances)),
        "max_frame_distance": float(np.max(frame_distances)),
        "endpoint_distance": float(np.linalg.norm(a[-1] - b[-1])),
        "mean_step_profile_distance": float(np.mean(np.abs(step_a - step_b))),
    }


def pairwise_separation_rows(rows: list[dict[str, object]], results_root: Path) -> list[dict[str, object]]:
    output = []
    embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    def cached_embedding(encoder: str, video_id: str) -> np.ndarray:
        key = (encoder, video_id)
        if key not in embedding_cache:
            embedding_cache[key] = load_embedding(results_root, encoder, video_id)
        return embedding_cache[key]

    for encoder in sorted({str(row["encoder"]) for row in rows}):
        for scene in sorted({str(row["scene"]) for row in rows}):
            group = [row for row in rows if row["encoder"] == encoder and row["scene"] == scene]
            for i, row_a in enumerate(group):
                for row_b in group[i + 1 :]:
                    label_a = str(row_a["label"])
                    label_b = str(row_b["label"])
                    if label_a == "correct" and label_b == "correct":
                        pair_type = "correct_correct"
                    elif label_a == "wrong" and label_b == "wrong":
                        pair_type = "wrong_wrong"
                    else:
                        pair_type = "correct_wrong"
                    emb_a = cached_embedding(encoder, str(row_a["video_id"]))
                    emb_b = cached_embedding(encoder, str(row_b["video_id"]))
                    distances = trajectory_distance(emb_a, emb_b)
                    output.append(
                        {
                            "encoder": encoder,
                            "scene": scene,
                            "pair_type": pair_type,
                            "video_a": row_a["video_id"],
                            "video_b": row_b["video_id"],
                            "label_a": label_a,
                            "label_b": label_b,
                            **distances,
                        }
                    )
    return output


def pairwise_summary_rows(pairwise_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for key, group in sorted(group_rows(pairwise_rows, ["encoder", "scene", "pair_type"]).items()):
        encoder, scene, pair_type = key
        out: dict[str, object] = {"encoder": encoder, "scene": scene, "pair_type": pair_type, "n": len(group)}
        for metric in ["mean_frame_distance", "max_frame_distance", "endpoint_distance", "mean_step_profile_distance"]:
            values = [float(row[metric]) for row in group]
            out[f"mean_{metric}"] = mean(values)
            out[f"std_{metric}"] = std(values)
        output.append(out)
    return output


def plot_metric_boxplots(rows: list[dict[str, object]], output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    selected = ["straightness", "std_step_distance", "mean_step_distance"]
    for ax, metric in zip(axes, selected):
        data = []
        labels = []
        for label in ["correct", "wrong"]:
            data.append([float(row[metric]) for row in rows if row["label"] == label])
            labels.append(label)
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Correct vs wrong physics across all encoders")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pairwise_separation(pairwise_rows: list[dict[str, object]], output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    pair_types = ["correct_correct", "correct_wrong", "wrong_wrong"]
    metrics = ["mean_frame_distance", "mean_step_profile_distance"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, metrics):
        data = [[float(row[metric]) for row in pairwise_rows if row["pair_type"] == pair_type] for pair_type in pair_types]
        ax.boxplot(data, tick_labels=["C-C", "C-W", "W-W"], showfliers=True)
        ax.set_title(metric)
        ax.set_ylabel("distance")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Pairwise latent trajectory separation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_wrong_type_step_distance(rows: list[dict[str, object]], output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    wrong_rows = [row for row in rows if row["label"] == "wrong"]
    wrong_types = sorted({str(row["wrong_type"]) for row in wrong_rows})
    if not wrong_types:
        return
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(wrong_types))))
    positions = np.arange(len(wrong_types))
    mean_values = [mean([float(row["mean_step_distance"]) for row in wrong_rows if row["wrong_type"] == wrong_type]) for wrong_type in wrong_types]
    std_values = [mean([float(row["std_step_distance"]) for row in wrong_rows if row["wrong_type"] == wrong_type]) for wrong_type in wrong_types]
    ax.barh(positions - 0.18, std_values, height=0.35, label="std step")
    ax.barh(positions + 0.18, mean_values, height=0.35, label="mean step")
    ax.set_yticks(positions)
    ax.set_yticklabels(wrong_types)
    ax.set_xlabel("metric value averaged across encoders")
    ax.set_title("Wrong-physics type breakdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_encoder_deltas(delta_rows: list[dict[str, object]], output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    encoders = [str(row["encoder"]) for row in delta_rows]
    y = np.arange(len(encoders))
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(encoders))))
    ax.barh(y - 0.18, [float(row["delta_std_step_distance"]) for row in delta_rows], height=0.35, label="std step")
    ax.barh(y + 0.18, [float(row["delta_mean_step_distance"]) for row in delta_rows], height=0.35, label="mean step")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(encoders)
    ax.set_xlabel("wrong - correct")
    ax.set_title("Step-distance changes by encoder")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scale_comparison(delta_rows: list[dict[str, object]], output_dir: Path) -> None:
    plt = require_matplotlib(output_dir)
    by_encoder = {str(row["encoder"]): row for row in delta_rows}
    for title, encoders in MODEL_GROUPS.items():
        available = [encoder for encoder in encoders if encoder in by_encoder]
        if len(available) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(available))
        ax.plot(x, [float(by_encoder[e]["delta_std_step_distance"]) for e in available], marker="o", label="std step")
        ax.plot(x, [float(by_encoder[e]["delta_mean_step_distance"]) for e in available], marker="o", label="mean step")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(available, rotation=20, ha="right")
        ax.set_ylabel("wrong - correct")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        filename = title.lower().replace(" ", "_") + ".png"
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def select_examples(rows: list[dict[str, object]], encoders: list[str]) -> list[tuple[str, str]]:
    examples = []
    for scene in ["single_pendulum", "double_pendulum"]:
        for label in ["correct", "wrong"]:
            candidates = [
                row
                for row in rows
                if row["scene"] == scene and row["label"] == label and row["encoder"] in encoders
            ]
            if candidates:
                examples.append((scene, str(candidates[0]["video_id"])))
    return examples


def plot_pca_trajectories(rows: list[dict[str, object]], results_root: Path, output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    preferred = ["dinov2_small", "clip_vit_base_patch32", "siglip_base_patch16_224"]
    encoders = [encoder for encoder in preferred if any(row["encoder"] == encoder for row in rows)]
    if not encoders:
        encoders = sorted({str(row["encoder"]) for row in rows})[:3]
    examples = select_examples(rows, encoders)
    if not examples:
        return

    fig, axes = plt.subplots(len(encoders), len(examples), figsize=(4.2 * len(examples), 3.6 * len(encoders)))
    axes = np.asarray(axes).reshape(len(encoders), len(examples))
    for row_index, encoder in enumerate(encoders):
        all_embeddings = []
        loaded = {}
        for _scene, video_id in examples:
            emb = load_embedding(results_root, encoder, video_id)
            loaded[video_id] = emb
            all_embeddings.append(emb)
        all_points = np.concatenate(all_embeddings, axis=0)
        all_pca = pca_2d(all_points)
        cursor = 0
        pca_by_video = {}
        for _scene, video_id in examples:
            count = loaded[video_id].shape[0]
            pca_by_video[video_id] = all_pca[cursor : cursor + count]
            cursor += count
        for col_index, (scene, video_id) in enumerate(examples):
            ax = axes[row_index, col_index]
            points = pca_by_video[video_id]
            ax.plot(points[:, 0], points[:, 1], marker="o", markersize=3, linewidth=1.2)
            ax.scatter(points[0, 0], points[0, 1], s=35, label="start")
            ax.scatter(points[-1, 0], points[-1, 1], s=35, label="end")
            label = "correct" if "_correct_" in video_id else "wrong"
            ax.set_title(f"{encoder}\n{scene} {label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("PCA trajectories from frame embeddings")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_step_timeseries(rows: list[dict[str, object]], results_root: Path, output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    encoders = [encoder for encoder in ["dinov2_small", "clip_vit_base_patch32", "siglip_base_patch16_224"] if any(row["encoder"] == encoder for row in rows)]
    if not encoders:
        return
    fig, axes = plt.subplots(len(encoders), 2, figsize=(10, 3.4 * len(encoders)), sharex=True)
    axes = np.asarray(axes).reshape(len(encoders), 2)
    for row_index, encoder in enumerate(encoders):
        for col_index, scene in enumerate(["single_pendulum", "double_pendulum"]):
            ax = axes[row_index, col_index]
            for label, color in [("correct", "#2a7f62"), ("wrong", "#b84545")]:
                candidates = [
                    row
                    for row in rows
                    if row["encoder"] == encoder and row["scene"] == scene and row["label"] == label
                ]
                if not candidates:
                    continue
                series = []
                for candidate in candidates:
                    emb = load_embedding(results_root, encoder, str(candidate["video_id"]))
                    series.append(np.linalg.norm(np.diff(emb, axis=0), axis=1))
                max_len = max(len(item) for item in series)
                padded = np.full((len(series), max_len), np.nan)
                for i, item in enumerate(series):
                    padded[i, : len(item)] = item
                ax.plot(np.nanmean(padded, axis=0), color=color, label=label)
                ax.fill_between(
                    np.arange(max_len),
                    np.nanmean(padded, axis=0) - np.nanstd(padded, axis=0),
                    np.nanmean(padded, axis=0) + np.nanstd(padded, axis=0),
                    color=color,
                    alpha=0.15,
                )
            ax.set_title(f"{encoder} / {scene}", fontsize=9)
            ax.set_ylabel("step distance")
            ax.grid(alpha=0.25)
            if row_index == 0 and col_index == 0:
                ax.legend()
    fig.suptitle("Frame-to-frame embedding movement over time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_delta_heatmap(delta_rows: list[dict[str, object]], output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    encoders = [str(row["encoder"]) for row in delta_rows]
    data = np.array([[float(row[f"delta_{metric}"]) for metric in METRICS] for row in delta_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(encoders))))
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    image = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(encoders)))
    ax.set_yticklabels(encoders)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(METRICS, rotation=25, ha="right")
    ax.set_title("Wrong - correct metric deltas")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    auc_rows: list[dict[str, object]],
    pairwise_count: int,
) -> None:
    best_step = sorted(delta_rows, key=lambda row: float(row["delta_std_step_distance"]), reverse=True)[:3]
    weak = sorted(delta_rows, key=lambda row: float(row["delta_std_step_distance"]))[:2]
    auc_sorted = sorted(auc_rows, key=lambda row: float(row["auc_wrong_gt_correct_mean_step_distance"]), reverse=True)[:3]
    lines = [
        "# Simulated Pendulum Analysis",
        "",
        "## Dataset And Runs",
        "",
        f"- Rows analysed: {len(rows)}",
        f"- Encoders: {len({row['encoder'] for row in rows})}",
        f"- Videos: {len({row['video_id'] for row in rows})}",
        f"- Pairwise comparisons: {pairwise_count}",
        f"- Successful rows: {sum(row['status'] == 'ok' for row in rows)}",
        "",
        "## Main Finding",
        "",
        "The simulated pendulum dataset gives controlled correct and wrong physics examples. "
        "Across most encoders, wrong-physics videos show larger frame-to-frame movement in latent space. "
        "This is clearer for `mean_step_distance` and `std_step_distance` than for `straightness`.",
        "",
        "## Strongest Encoders By Step-Distance Delta",
        "",
    ]
    for row in best_step:
        lines.append(
            f"- `{row['encoder']}`: delta std step = {float(row['delta_std_step_distance']):+.3f}, "
            f"delta mean step = {float(row['delta_mean_step_distance']):+.3f}"
        )
    lines.extend(["", "## Weakest Encoders By Step-Distance Delta", ""])
    for row in weak:
        lines.append(
            f"- `{row['encoder']}`: delta std step = {float(row['delta_std_step_distance']):+.3f}, "
            f"delta mean step = {float(row['delta_mean_step_distance']):+.3f}"
        )
    lines.extend(["", "## Highest AUC For Mean Step Distance", ""])
    for row in auc_sorted:
        lines.append(
            f"- `{row['encoder']}`: AUC = {float(row['auc_wrong_gt_correct_mean_step_distance']):.3f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation For Report",
            "",
            "This supports analysing latent trajectories as temporal signals rather than relying only on static frame embeddings. "
            "However, trajectory metrics alone do not explain where the model is looking. "
            "The next analysis should inspect patch-level feature changes and attention maps to test whether the model response is concentrated on the pendulum bob, rod, and pivot.",
            "",
            "## Generated Figures",
            "",
            "- `figures/metric_boxplots.png`: correct vs wrong metric distributions.",
            "- `figures/pairwise_separation.png`: correct-correct vs correct-wrong trajectory separation.",
            "- `figures/wrong_type_step_distance.png`: step-distance metrics broken down by wrong-physics type.",
            "- `figures/encoder_step_deltas.png`: wrong-correct step-distance changes by encoder.",
            "- `figures/delta_heatmap.png`: wrong-correct deltas across all metrics.",
            "- `figures/pca_trajectories.png`: selected latent trajectories projected to 2D.",
            "- `figures/step_distance_timeseries.png`: temporal step-distance curves.",
            "- `figures/dinov2_scale.png` and `figures/clip_scale.png`: model-scale comparisons when available.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def encoder_label(encoder: str) -> str:
    labels = {
        "clip_vit_base_patch32": "CLIP-B/32",
        "clip_vit_large_patch14": "CLIP-L/14",
        "dinov2_base": "DINOv2-B",
        "dinov2_large": "DINOv2-L",
        "dinov2_small": "DINOv2-S",
        "mae_base": "MAE-B",
        "siglip_base_patch16_224": "SigLIP-B",
        "swin_base": "Swin-B",
        "vit_base_imagenet21k": "ViT-B",
    }
    return labels.get(encoder, encoder)


def fmt_signed(value: object) -> str:
    return f"{float(value):+.3f}"


def fmt_auc(value: object) -> str:
    return f"{float(value):.3f}"


def write_latex_tables(asset_dir: Path, delta_rows: list[dict[str, object]], scene_delta_rows: list[dict[str, object]], auc_rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wrong-minus-correct changes on the controlled simulated pendulum benchmark. Positive step-distance deltas indicate that wrong-physics videos produce larger frame-to-frame movement in latent space.}",
        r"\label{tab:sim-encoder-deltas}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Encoder & $\Delta$ straight. & $\Delta$ curv. & $\Delta$ std step & $\Delta$ mean step \\",
        r"\midrule",
    ]
    for row in delta_rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(encoder_label(str(row["encoder"]))),
                    fmt_signed(row["delta_straightness"]),
                    fmt_signed(row["delta_mean_curvature"]),
                    fmt_signed(row["delta_std_step_distance"]),
                    fmt_signed(row["delta_mean_step_distance"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    auc_by_encoder = {str(row["encoder"]): row for row in auc_rows}
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Encoder-wise AUC when treating larger metric values as more likely to indicate wrong physics. Step-distance metrics are more reliable than straightness or curvature.}",
            r"\label{tab:sim-auc}",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Encoder & straight. & mean curv. & std step & mean step \\",
            r"\midrule",
        ]
    )
    for row in delta_rows:
        encoder = str(row["encoder"])
        auc = auc_by_encoder[encoder]
        lines.append(
            " & ".join(
                [
                    latex_escape(encoder_label(encoder)),
                    fmt_auc(auc["auc_wrong_gt_correct_straightness"]),
                    fmt_auc(auc["auc_wrong_gt_correct_mean_curvature"]),
                    fmt_auc(auc["auc_wrong_gt_correct_std_step_distance"]),
                    fmt_auc(auc["auc_wrong_gt_correct_mean_step_distance"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    scene_labels = {
        "single_pendulum": "Single pendulum",
        "double_pendulum": "Double pendulum",
    }
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Scene-level wrong-minus-correct deltas pooled across encoders. Both single- and double-pendulum wrong videos increase latent step-distance metrics.}",
            r"\label{tab:sim-scene-deltas}",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Scene & $\Delta$ straight. & $\Delta$ curv. & $\Delta$ std step & $\Delta$ mean step \\",
            r"\midrule",
        ]
    )
    for row in scene_delta_rows:
        scene = scene_labels.get(str(row["scene"]), str(row["scene"]))
        lines.append(
            " & ".join(
                [
                    latex_escape(scene),
                    fmt_signed(row["delta_straightness"]),
                    fmt_signed(row["delta_mean_curvature"]),
                    fmt_signed(row["delta_std_step_distance"]),
                    fmt_signed(row["delta_mean_step_distance"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    (asset_dir / "tables.tex").write_text("\n".join(lines), encoding="utf-8")


def write_latex_figures(asset_dir: Path) -> None:
    lines = [
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{generated/simulated_pendulum/figures/delta_heatmap.png}",
        r"\caption{Wrong-minus-correct metric deltas across encoders on the simulated pendulum benchmark. The step-distance metrics show the most consistent positive changes for wrong-physics videos, while straightness and curvature are less stable.}",
        r"\label{fig:sim-delta-heatmap}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{generated/simulated_pendulum/figures/pairwise_separation.png}",
        r"\caption{Pairwise latent trajectory separation. Correct-wrong pairs are compared against correct-correct pairs to test whether physical violations exceed normal variation between simulated correct videos.}",
        r"\label{fig:sim-pairwise-separation}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{generated/simulated_pendulum/figures/wrong_type_step_distance.png}",
        r"\caption{Wrong-physics type breakdown. Abrupt interventions such as velocity kicks and gravity changes are expected to produce stronger latent movement than smooth energy-gain variants.}",
        r"\label{fig:sim-wrong-type-breakdown}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{generated/simulated_pendulum/figures/encoder_step_deltas.png}",
        r"\caption{Encoder-wise changes in mean and standard deviation of frame-to-frame latent step distances. DINOv2 encoders show the largest absolute step-distance changes, while MAE does not separate correct and wrong physics in this setting.}",
        r"\label{fig:sim-encoder-step-deltas}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{generated/simulated_pendulum/figures/metric_boxplots.png}",
        r"\caption{Correct versus wrong metric distributions pooled across encoders. Straightness overlaps strongly between the two classes, whereas wrong-physics videos produce a heavier upper tail for step-distance variability.}",
        r"\label{fig:sim-metric-boxplots}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{generated/simulated_pendulum/figures/step_distance_timeseries.png}",
        r"\caption{Temporal profile of frame-to-frame latent movement for representative encoders. Wrong-physics videos tend to have larger movements and occasional spikes, especially for DINOv2-small and CLIP-B/32.}",
        r"\label{fig:sim-step-timeseries}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{generated/simulated_pendulum/figures/pca_trajectories.png}",
        r"\caption{Qualitative PCA visualisation of selected latent trajectories. Each point is a sampled frame embedding. This figure is intended as an illustration of trajectory behaviour rather than the primary quantitative evidence.}",
        r"\label{fig:sim-pca-trajectories}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=0.48\linewidth]{generated/simulated_pendulum/figures/dinov2_scale.png}",
        r"\hfill",
        r"\includegraphics[width=0.48\linewidth]{generated/simulated_pendulum/figures/clip_scale.png}",
        r"\caption{Model-scale comparisons for DINOv2 and CLIP. DINOv2 models consistently show positive step-distance deltas, but the effect is not monotonic with scale. CLIP-B/32 is more separable than CLIP-L/14 in this small benchmark.}",
        r"\label{fig:sim-scale-comparison}",
        r"\end{figure}",
        "",
    ]
    (asset_dir / "figures.tex").write_text("\n".join(lines), encoding="utf-8")


def sync_latex_assets(asset_dir: Path, figure_dir: Path, delta_rows: list[dict[str, object]], scene_delta_rows: list[dict[str, object]], auc_rows: list[dict[str, object]]) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    latex_figure_dir = asset_dir / "figures"
    latex_figure_dir.mkdir(parents=True, exist_ok=True)
    figure_names = [
        "delta_heatmap.png",
        "pairwise_separation.png",
        "wrong_type_step_distance.png",
        "encoder_step_deltas.png",
        "metric_boxplots.png",
        "step_distance_timeseries.png",
        "pca_trajectories.png",
        "dinov2_scale.png",
        "clip_scale.png",
    ]
    for name in figure_names:
        source = figure_dir / name
        if source.exists():
            shutil.copy2(source, latex_figure_dir / name)
    write_latex_tables(asset_dir, delta_rows, scene_delta_rows, auc_rows)
    write_latex_figures(asset_dir)
    (asset_dir / "README.md").write_text(
        "\n".join(
            [
                "# Simulated Pendulum LaTeX Assets",
                "",
                "Generated by `generated_analysis/analyze_simulated_pendulum.py --latex_asset_dir ...`.",
                "",
                "Keep this directory under `reports/irp-template-main/latex/generated/simulated_pendulum/` when uploading to Overleaf.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def package_outputs(output_dir: Path, package_name: str) -> Path:
    package_path = output_dir / package_name
    with tarfile.open(package_path, "w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if path == package_path or path.is_dir():
                continue
            if any(part.startswith(".") for part in path.relative_to(output_dir).parts):
                continue
            tar.add(path, arcname=path.relative_to(output_dir))
    return package_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(args.metadata)
    rows = load_rows(args.results_root, metadata)
    delta_rows = encoder_deltas(rows)
    scene_delta_rows = scene_deltas(rows)
    auc_rows = auc_table(rows)
    pairwise_rows = pairwise_separation_rows(rows, args.results_root)
    pairwise_summary = pairwise_summary_rows(pairwise_rows)
    encoder_label_rows = summarise_groups(rows, ["encoder", "label"])
    scene_label_rows = summarise_groups(rows, ["scene", "label"])
    wrong_type_rows = summarise_groups([row for row in rows if row["label"] == "wrong"], ["scene", "wrong_type"])

    write_csv(table_dir / "combined_metrics.csv", rows)
    write_csv(table_dir / "encoder_label_summary.csv", encoder_label_rows)
    write_csv(table_dir / "encoder_wrong_correct_deltas.csv", delta_rows)
    write_csv(table_dir / "scene_label_summary.csv", scene_label_rows)
    write_csv(table_dir / "scene_wrong_correct_deltas.csv", scene_delta_rows)
    write_csv(table_dir / "wrong_type_summary.csv", wrong_type_rows)
    write_csv(table_dir / "auc_by_encoder.csv", auc_rows)
    write_csv(table_dir / "pairwise_separation.csv", pairwise_rows)
    write_csv(table_dir / "pairwise_separation_summary.csv", pairwise_summary)

    plot_metric_boxplots(rows, figure_dir / "metric_boxplots.png")
    plot_pairwise_separation(pairwise_rows, figure_dir / "pairwise_separation.png")
    plot_wrong_type_step_distance(rows, figure_dir / "wrong_type_step_distance.png")
    plot_encoder_deltas(delta_rows, figure_dir / "encoder_step_deltas.png")
    plot_scale_comparison(delta_rows, figure_dir)
    plot_delta_heatmap(delta_rows, figure_dir / "delta_heatmap.png")
    plot_pca_trajectories(rows, args.results_root, figure_dir / "pca_trajectories.png")
    plot_step_timeseries(rows, args.results_root, figure_dir / "step_distance_timeseries.png")

    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results_root": str(args.results_root),
                "metadata": str(args.metadata),
                "num_rows": len(rows),
                "num_encoders": len({row["encoder"] for row in rows}),
                "num_videos": len({row["video_id"] for row in rows}),
                "num_pairwise_comparisons": len(pairwise_rows),
            },
            handle,
            indent=2,
        )
    write_report(args.output_dir / "analysis_report.md", rows, delta_rows, auc_rows, len(pairwise_rows))
    if args.latex_asset_dir is not None:
        sync_latex_assets(args.latex_asset_dir, figure_dir, delta_rows, scene_delta_rows, auc_rows)
    package_path = package_outputs(args.output_dir, args.package_name)

    print(f"Rows: {len(rows)}")
    print(f"Encoders: {len({row['encoder'] for row in rows})}")
    print(f"Videos: {len({row['video_id'] for row in rows})}")
    print(f"Pairwise comparisons: {len(pairwise_rows)}")
    print(f"Output: {args.output_dir}")
    if args.latex_asset_dir is not None:
        print(f"LaTeX assets: {args.latex_asset_dir}")
    print(f"Package: {package_path}")


if __name__ == "__main__":
    main()
