#!/usr/bin/env python
"""Analyze and package multi-encoder toy video evaluation results."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {
    "mean_curvature",
    "max_curvature",
    "mean_step_distance",
    "max_step_distance",
    "std_step_distance",
}
HIGHER_IS_BETTER = {"straightness"}
RANK_METRICS = [
    "mean_curvature",
    "max_curvature",
    "std_step_distance",
    "straightness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multi-encoder evaluation results.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results/toy_basic_physics_multi_encoder"),
        help="Root directory containing one subdirectory per encoder.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/toy_basic_physics_multi_encoder_analysis"),
        help="Directory where analysis tables, figures, and package are written.",
    )
    parser.add_argument(
        "--package_name",
        default="toy_multi_encoder_results_with_analysis.tar.gz",
        help="Name of the final tar.gz package written under output_dir.",
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
        raise SystemExit(
            "ERROR: Missing matplotlib. Install generated_eval requirements first."
        ) from exc
    return plt


def find_summary_files(results_root: Path) -> list[Path]:
    if not results_root.exists():
        raise SystemExit(f"ERROR: results_root does not exist: {results_root}")
    files = sorted(results_root.glob("*/metrics_summary.csv"))
    if not files:
        raise SystemExit(f"ERROR: no metrics_summary.csv files found under: {results_root}")
    return files


def load_combined_metrics(results_root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in find_summary_files(results_root):
        encoder = summary_path.parent.name
        df = pd.read_csv(summary_path)
        df.insert(0, "encoder", encoder)
        rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    if "video_id" not in combined.columns:
        raise SystemExit("ERROR: combined metrics are missing required column: video_id")
    return combined


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    skipped = {"num_frames", "embedding_dim"}
    cols = []
    for col in df.columns:
        if col in skipped:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def make_encoder_summary(combined: pd.DataFrame) -> pd.DataFrame:
    metric_cols = numeric_metric_columns(combined)
    grouped = combined.groupby("encoder", dropna=False)[metric_cols]
    mean_df = grouped.mean().add_prefix("mean_")
    std_df = grouped.std(ddof=0).add_prefix("std_")
    count_df = grouped.count().iloc[:, :1].rename(columns={metric_cols[0]: "num_videos"})
    return pd.concat([count_df, mean_df, std_df], axis=1).reset_index()


def minmax_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = series.astype(float)
    spread = values.max() - values.min()
    if not np.isfinite(spread) or spread <= 1e-12:
        return pd.Series(0.5, index=series.index)
    score = (values - values.min()) / spread
    return score if higher_is_better else 1.0 - score


def make_encoder_rankings(combined: pd.DataFrame) -> pd.DataFrame:
    summary = make_encoder_summary(combined)
    score_parts = []
    for metric in RANK_METRICS:
        mean_col = f"mean_{metric}"
        if mean_col not in summary.columns:
            continue
        higher = metric in HIGHER_IS_BETTER
        score = minmax_score(summary[mean_col], higher_is_better=higher)
        summary[f"score_{metric}"] = score
        score_parts.append(f"score_{metric}")
    if score_parts:
        summary["overall_score"] = summary[score_parts].mean(axis=1)
    else:
        summary["overall_score"] = np.nan
    return summary.sort_values("overall_score", ascending=False).reset_index(drop=True)


def pivot_metric(combined: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in combined.columns:
        raise SystemExit(f"ERROR: metric not found in combined results: {metric}")
    return combined.pivot_table(index="encoder", columns="video_id", values=metric, aggfunc="mean")


def plot_heatmap(combined: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    plt = require_matplotlib(output_path.parent)
    pivot = pivot_metric(combined, metric)

    fig_width = max(7.0, 1.2 * len(pivot.columns))
    fig_height = max(4.5, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=metric)

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.3g}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overall_scores(rankings: pd.DataFrame, output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    ordered = rankings.sort_values("overall_score", ascending=True)
    fig_height = max(4.0, 0.45 * len(ordered))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(ordered["encoder"], ordered["overall_score"])
    ax.set_xlabel("Normalized aggregate score")
    ax.set_title("Encoder ranking across toy videos")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_bars(rankings: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    mean_col = f"mean_{metric}"
    if mean_col not in rankings.columns:
        return
    ascending = metric in LOWER_IS_BETTER
    ordered = rankings.sort_values(mean_col, ascending=ascending)
    fig_height = max(4.0, 0.45 * len(ordered))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(ordered["encoder"], ordered[mean_col])
    ax.invert_yaxis()
    ax.set_xlabel(mean_col)
    ax.set_title(f"Encoder average {metric}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    combined: pd.DataFrame,
    rankings: pd.DataFrame,
    output_path: Path,
) -> None:
    best = rankings.iloc[0] if not rankings.empty else None
    lines = [
        "# Multi-Encoder Toy Evaluation Analysis",
        "",
        f"- Encoders: {combined['encoder'].nunique()}",
        f"- Videos: {combined['video_id'].nunique()}",
        f"- Rows: {len(combined)}",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Top Encoder",
                "",
                f"- Encoder: `{best['encoder']}`",
                f"- Overall score: `{best['overall_score']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Ranking Metrics",
            "",
            "- Lower is better: `mean_curvature`, `max_curvature`, `std_step_distance`",
            "- Higher is better: `straightness`",
            "- `overall_score` is a min-max normalized aggregate over those metrics.",
            "",
            "## Key Files",
            "",
            "- `combined_metrics.csv`: one row per encoder/video result.",
            "- `encoder_rankings.csv`: aggregate encoder ranking.",
            "- `encoder_summary.csv`: per-encoder mean/std metrics.",
            "- `figures/`: heatmaps and ranking plots.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def add_path_to_tar(tar: tarfile.TarFile, path: Path, arcname: Path) -> None:
    if path.exists():
        tar.add(path, arcname=str(arcname))


def package_results(results_root: Path, output_dir: Path, package_name: str) -> Path:
    package_path = output_dir / package_name
    if package_path.exists():
        package_path.unlink()
    with tarfile.open(package_path, "w:gz") as tar:
        add_path_to_tar(tar, results_root, Path("evaluation_results") / results_root.name)
        analysis_root = Path("analysis_results") / output_dir.name
        for child in sorted(output_dir.iterdir()):
            if child.resolve() == package_path.resolve():
                continue
            add_path_to_tar(tar, child, analysis_root / child.name)
    return package_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined = load_combined_metrics(args.results_root)
    combined_path = args.output_dir / "combined_metrics.csv"
    combined.to_csv(combined_path, index=False)

    encoder_summary = make_encoder_summary(combined)
    encoder_summary_path = args.output_dir / "encoder_summary.csv"
    encoder_summary.to_csv(encoder_summary_path, index=False)

    rankings = make_encoder_rankings(combined)
    rankings_path = args.output_dir / "encoder_rankings.csv"
    rankings.to_csv(rankings_path, index=False)

    plot_heatmap(
        combined,
        "mean_curvature",
        figures_dir / "mean_curvature_heatmap.png",
        "Mean curvature by encoder and video",
    )
    plot_heatmap(
        combined,
        "straightness",
        figures_dir / "straightness_heatmap.png",
        "Straightness by encoder and video",
    )
    plot_heatmap(
        combined,
        "std_step_distance",
        figures_dir / "std_step_distance_heatmap.png",
        "Step-distance variability by encoder and video",
    )
    plot_overall_scores(rankings, figures_dir / "encoder_overall_score.png")
    for metric in RANK_METRICS:
        plot_metric_bars(rankings, metric, figures_dir / f"encoder_{metric}.png")

    report_path = args.output_dir / "analysis_report.md"
    write_markdown_report(combined, rankings, report_path)

    package_path = package_results(args.results_root, args.output_dir, args.package_name)

    print("Analysis complete.")
    print(f"Combined metrics: {combined_path}")
    print(f"Encoder summary: {encoder_summary_path}")
    print(f"Encoder rankings: {rankings_path}")
    print(f"Figures: {figures_dir}")
    print(f"Report: {report_path}")
    print(f"Package: {package_path}")


if __name__ == "__main__":
    main()
