#!/usr/bin/env python
"""Compare basic-physics and wrong-physics multi-encoder evaluation results."""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


COMPARE_METRICS = [
    "straightness",
    "mean_curvature",
    "max_curvature",
    "std_step_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two toy physics result sets.")
    parser.add_argument(
        "--basic_results_root",
        type=Path,
        default=Path("results/toy_basic_physics_multi_encoder"),
    )
    parser.add_argument(
        "--wrong_results_root",
        type=Path,
        default=Path("results/toy_wrong_physics_multi_encoder"),
    )
    parser.add_argument(
        "--basic_metadata",
        type=Path,
        default=Path("generated_videos/toy_basic_physics/metadata.csv"),
    )
    parser.add_argument(
        "--wrong_metadata",
        type=Path,
        default=Path("generated_videos/toy_wrong_physics/metadata.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/toy_physics_set_comparison"),
    )
    parser.add_argument(
        "--package_name",
        default="toy_physics_set_comparison.tar.gz",
    )
    return parser.parse_args()


def require_matplotlib(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("ERROR: Missing matplotlib. Install generated_eval requirements first.") from exc
    return plt


def load_result_root(results_root: Path, label: str) -> pd.DataFrame:
    if not results_root.exists():
        raise SystemExit(f"ERROR: results root does not exist: {results_root}")
    files = sorted(results_root.glob("*/metrics_summary.csv"))
    if not files:
        raise SystemExit(f"ERROR: no metrics_summary.csv files found under: {results_root}")

    frames = []
    for path in files:
        encoder = path.parent.name
        df = pd.read_csv(path)
        df.insert(0, "encoder", encoder)
        df.insert(1, "set_label", label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_metadata(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: metadata file does not exist: {path}")
    df = pd.read_csv(path)
    if "pair_id" not in df.columns:
        df["pair_id"] = df["video_id"].map(infer_pair_id)
    return df[["video_id", "pair_id"]].assign(set_label=label)


def infer_pair_id(video_id: str) -> str:
    value = video_id.lower()
    if "elasticcollision" in value:
        return "elastic_collision"
    if "freefall" in value:
        return "free_fall"
    if "friction" in value:
        return "friction"
    if "pendulum" in value:
        return "pendulum"
    if "rollingdownaslope" in value:
        return "rolling_slope"
    return value


def attach_pair_ids(results: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(metadata, on=["set_label", "video_id"], how="left")
    missing = merged["pair_id"].isna()
    if missing.any():
        merged.loc[missing, "pair_id"] = merged.loc[missing, "video_id"].map(infer_pair_id)
    return merged


def make_comparison(basic: pd.DataFrame, wrong: pd.DataFrame) -> pd.DataFrame:
    ok_basic = basic[basic["status"] == "ok"].copy()
    ok_wrong = wrong[wrong["status"] == "ok"].copy()
    keep = ["encoder", "pair_id", "video_id"] + COMPARE_METRICS
    left = ok_basic[keep].rename(columns={"video_id": "basic_video_id"})
    right = ok_wrong[keep].rename(columns={"video_id": "wrong_video_id"})
    merged = left.merge(right, on=["encoder", "pair_id"], suffixes=("_basic", "_wrong"))
    for metric in COMPARE_METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_wrong"] - merged[f"{metric}_basic"]
    return merged


def make_encoder_delta_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [f"delta_{metric}" for metric in COMPARE_METRICS]
    summary = comparison.groupby("encoder", dropna=False)[delta_cols].mean().reset_index()
    counts = comparison.groupby("encoder", dropna=False).size().reset_index(name="num_pairs")
    return counts.merge(summary, on="encoder", how="left")


def plot_delta_heatmap(comparison: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt = require_matplotlib(output_path.parent)
    delta_col = f"delta_{metric}"
    pivot = comparison.pivot_table(index="encoder", columns="pair_id", values=delta_col, aggfunc="mean")
    fig_width = max(7.0, 1.2 * len(pivot.columns))
    fig_height = max(4.5, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    limit = np.nanmax(np.abs(pivot.values))
    if not np.isfinite(limit) or limit <= 1e-12:
        limit = 1.0
    image = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Wrong minus basic: {metric}")
    fig.colorbar(image, ax=ax, label=delta_col)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.3g}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    basic: pd.DataFrame,
    wrong: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    table_cols = list(summary.columns)
    table_lines = [
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join("---" for _ in table_cols) + " |",
    ]
    for _, row in summary.iterrows():
        values = []
        for col in table_cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")

    lines = [
        "# Toy Physics Set Comparison",
        "",
        "This report compares `toy_basic_physics` against `toy_wrong_physics`.",
        "Deltas are computed as `wrong - basic` after matching by `encoder` and `pair_id`.",
        "",
        "## Completeness",
        "",
        f"- Basic rows: {len(basic)}; successful: {(basic['status'] == 'ok').sum()}",
        f"- Wrong rows: {len(wrong)}; successful: {(wrong['status'] == 'ok').sum()}",
        f"- Matched encoder/video pairs: {len(comparison)}",
        "",
        "## Interpretation",
        "",
        "- Positive `delta_mean_curvature` means the wrong-physics video has higher curvature.",
        "- Negative `delta_straightness` means the wrong-physics video is less straight.",
        "- These are proxy metrics; visual inspection is still required.",
        "",
        "## Encoder-Level Mean Deltas",
        "",
        "\n".join(table_lines),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def package_outputs(output_dir: Path, package_name: str) -> Path:
    package_path = output_dir / package_name
    if package_path.exists():
        package_path.unlink()
    with tarfile.open(package_path, "w:gz") as tar:
        for child in sorted(output_dir.iterdir()):
            if child.resolve() == package_path.resolve():
                continue
            tar.add(child, arcname=str(Path(output_dir.name) / child.name))
    return package_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    basic = load_result_root(args.basic_results_root, "basic")
    wrong = load_result_root(args.wrong_results_root, "wrong")
    metadata = pd.concat(
        [
            load_metadata(args.basic_metadata, "basic"),
            load_metadata(args.wrong_metadata, "wrong"),
        ],
        ignore_index=True,
    )
    basic = attach_pair_ids(basic, metadata)
    wrong = attach_pair_ids(wrong, metadata)

    combined = pd.concat([basic, wrong], ignore_index=True)
    comparison = make_comparison(basic, wrong)
    summary = make_encoder_delta_summary(comparison)

    combined.to_csv(args.output_dir / "combined_basic_wrong_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "paired_metric_deltas.csv", index=False)
    summary.to_csv(args.output_dir / "encoder_delta_summary.csv", index=False)

    for metric in COMPARE_METRICS:
        plot_delta_heatmap(comparison, metric, figures_dir / f"delta_{metric}_heatmap.png")

    report_path = args.output_dir / "comparison_report.md"
    write_report(report_path, basic, wrong, comparison, summary)
    package_path = package_outputs(args.output_dir, args.package_name)

    print("Comparison complete.")
    print(f"Combined metrics: {args.output_dir / 'combined_basic_wrong_metrics.csv'}")
    print(f"Paired deltas: {args.output_dir / 'paired_metric_deltas.csv'}")
    print(f"Encoder delta summary: {args.output_dir / 'encoder_delta_summary.csv'}")
    print(f"Figures: {figures_dir}")
    print(f"Report: {report_path}")
    print(f"Package: {package_path}")


if __name__ == "__main__":
    main()
