#!/usr/bin/env python
"""Connect simulated-pendulum results back to the original project plan.

This post-processing script consumes two existing packages:

    results/simulated_pendulum_analysis.tar.gz
    results/feature_attention_examples.tar.gz

It does not re-run encoders. It extracts the packaged results, reads the key
CSV/JSON summaries, creates a project-plan-aligned markdown report, writes a
small LaTeX snippet, builds one qualitative montage from the feature/attention
examples, and packages the outputs for download.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise outputs against the original project plan.")
    parser.add_argument(
        "--sim_analysis_package",
        type=Path,
        default=Path("results/simulated_pendulum_analysis.tar.gz"),
        help="Package produced by analyze_simulated_pendulum.py.",
    )
    parser.add_argument(
        "--feature_attention_package",
        type=Path,
        default=Path("results/feature_attention_examples.tar.gz"),
        help="Package produced by run_feature_attention_examples.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/project_plan_alignment"),
        help="Directory for the plan-aligned report, tables, figures, and package.",
    )
    parser.add_argument(
        "--latex_asset_dir",
        type=Path,
        default=None,
        help="Optional LaTeX asset directory for qualitative case-study outputs.",
    )
    parser.add_argument("--package_name", default="project_plan_alignment.tar.gz")
    return parser.parse_args()


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("ERROR: missing Pillow. Install it with: python -m pip install pillow") from exc
    return Image, ImageDraw, ImageFont


def extract_package(package_path: Path, target_dir: Path) -> None:
    if not package_path.exists():
        raise SystemExit(f"ERROR: missing package: {package_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package_path, "r:gz") as tar:
        try:
            tar.extractall(target_dir, filter="data")
        except TypeError:
            tar.extractall(target_dir)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fnum(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def top_rows(rows: list[dict[str, str]], metric: str, n: int = 3) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: fnum(row[metric]), reverse=True)[:n]


def collect_feature_summaries(feature_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(feature_root.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(feature_root)
        if len(rel.parts) < 3:
            continue
        encoder, video_alias = rel.parts[0], rel.parts[1]
        rows.append(
            {
                "encoder": encoder,
                "video_alias": video_alias,
                "video_path": data.get("video_path", ""),
                "num_frames": data.get("num_frames", ""),
                "attention_shape": json.dumps(data.get("attention_shape", [])),
                "num_feature_change_maps": data.get("num_feature_change_maps", 0),
                "num_attention_maps": data.get("num_attention_maps", 0),
                "has_attention_overlay": int(data.get("num_attention_maps", 0) > 0),
            }
        )
    return rows


def image_or_blank(path: Path, size: tuple[int, int], Image):
    if path.exists():
        return Image.open(path).convert("RGB").resize(size)
    return Image.new("RGB", size, "white")


def draw_label(draw, xy: tuple[int, int], text: str, font) -> None:
    x, y = xy
    draw.rectangle([x, y, x + 356, y + 24], fill=(255, 255, 255))
    draw.text((x + 5, y + 4), text, fill=(0, 0, 0), font=font)


def make_case_montage(feature_root: Path, output_path: Path) -> bool:
    Image, ImageDraw, ImageFont = require_pillow()
    cases = [
        ("dinov2_small", "single_correct", "DINOv2-S / single correct"),
        ("dinov2_small", "single_periodic_kick", "DINOv2-S / single kick"),
        ("clip_vit_base_patch32", "double_correct", "CLIP-B / double correct"),
        ("clip_vit_base_patch32", "double_reverse_gravity", "CLIP-B / double reverse gravity"),
    ]
    existing = [(enc, vid, label) for enc, vid, label in cases if (feature_root / enc / vid).exists()]
    if not existing:
        return False

    thumb = (224, 224)
    header_h = 34
    label_w = 190
    row_h = thumb[1]
    cols = 5
    canvas = Image.new("RGB", (label_w + cols * thumb[0], header_h + len(existing) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    headers = ["frame 0", "frame mid", "frame last", "feature change", "attention"]
    for col_idx, header in enumerate(headers):
        x0 = label_w + col_idx * thumb[0]
        draw_label(draw, (x0, 4), header, font)

    for row_idx, (encoder, video_alias, case_label) in enumerate(existing):
        base = feature_root / encoder / video_alias
        images = [
            base / "frames" / "frame_00.png",
            base / "frames" / "frame_03.png",
            base / "frames" / "frame_07.png",
            base / "feature_change" / "feature_change_to_frame_04.png",
            base / "attention" / "attention_frame_03.png",
        ]
        y0 = header_h + row_idx * row_h
        draw.rectangle([0, y0, label_w, y0 + row_h], fill=(248, 248, 248))
        draw.text((8, y0 + row_h // 2 - 8), case_label, fill=(0, 0, 0), font=font)
        for col_idx, path in enumerate(images):
            x0 = label_w + col_idx * thumb[0]
            canvas.paste(image_or_blank(path, thumb, Image), (x0, y0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def build_plan_rows(run_config: dict[str, object], feature_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "project_plan_deliverable": "Controlled prompt/video sample set",
            "current_artifact": "Controlled simulated single- and double-pendulum dataset",
            "status": "Covered by controlled simulation rather than prompt-only generation",
            "evidence": f"{run_config.get('num_videos', 'unknown')} videos recorded in metadata",
        },
        {
            "project_plan_deliverable": "Physical-consistency rubric",
            "current_artifact": "Binary correct/wrong labels plus wrong_type taxonomy",
            "status": "Partially covered",
            "evidence": "Labels are controlled by construction; a human 0/1/2 rubric can be added for generated videos",
        },
        {
            "project_plan_deliverable": "Embedding-based temporal analysis",
            "current_artifact": "Latent trajectory metrics across encoders",
            "status": "Covered",
            "evidence": f"{run_config.get('num_rows', 'unknown')} encoder-video rows",
        },
        {
            "project_plan_deliverable": "Event-based/frame-level evidence",
            "current_artifact": "Patch feature-change and attention examples",
            "status": "Covered for selected case studies",
            "evidence": f"{len(feature_rows)} model/video qualitative examples",
        },
        {
            "project_plan_deliverable": "Failure-mode taxonomy",
            "current_artifact": "Energy gain, kicks, damping changes, gravity changes",
            "status": "Covered for gravity/dynamics; collision and occlusion remain future work",
            "evidence": "Wrong types are explicit in metadata and wrong_type_summary.csv",
        },
    ]


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    run_config: dict[str, object],
    plan_rows: list[dict[str, object]],
    auc_rows: list[dict[str, str]],
    delta_rows: list[dict[str, str]],
    wrong_type_rows: list[dict[str, str]],
    feature_rows: list[dict[str, object]],
    montage_written: bool,
) -> None:
    best_auc = top_rows(auc_rows, "auc_wrong_gt_correct_mean_step_distance", 3)
    best_delta = top_rows(delta_rows, "delta_mean_step_distance", 3)
    strongest_wrong = top_rows(wrong_type_rows, "mean_mean_step_distance", 5)

    lines = [
        "# Project Plan Alignment Report",
        "",
        "## Why The Direction Still Matches The Original Plan",
        "",
        "The original plan asked whether generated videos preserve simple physical regularities over time. "
        "The current work keeps that question, but moves the main benchmark to controlled simulated videos "
        "because the early AI-generated examples were not reliable ground truth. This makes the evaluation "
        "more defensible: correct and wrong dynamics are known by construction, so latent trajectory results "
        "can be interpreted as representation responses to controlled physical changes.",
        "",
        "## Current Evidence Base",
        "",
        f"- Videos analysed: `{run_config.get('num_videos', 'unknown')}`",
        f"- Encoder-video rows: `{run_config.get('num_rows', 'unknown')}`",
        f"- Encoders: `{run_config.get('num_encoders', 'unknown')}`",
        f"- Pairwise trajectory comparisons: `{run_config.get('num_pairwise_comparisons', 'unknown')}`",
        f"- Qualitative feature/attention examples: `{len(feature_rows)}`",
        "",
        "## Deliverable Mapping",
        "",
        markdown_table(
            plan_rows,
            ["project_plan_deliverable", "current_artifact", "status", "evidence"],
        ),
        "",
        "## Strongest Quantitative Signals",
        "",
        "Best encoders by AUC using mean frame-to-frame step distance:",
        "",
    ]
    for row in best_auc:
        lines.append(
            f"- `{row['encoder']}`: AUC = {float(row['auc_wrong_gt_correct_mean_step_distance']):.3f}"
        )
    lines.extend(["", "Largest wrong-minus-correct mean step-distance deltas:", ""])
    for row in best_delta:
        lines.append(f"- `{row['encoder']}`: delta = {float(row['delta_mean_step_distance']):.3f}")
    lines.extend(["", "Wrong-physics types with strongest mean step-distance response:", ""])
    for row in strongest_wrong:
        lines.append(
            f"- `{row['scene']} / {row['wrong_type']}`: "
            f"mean step distance = {float(row['mean_mean_step_distance']):.3f}"
        )

    lines.extend(
        [
            "",
            "## Qualitative Evidence",
            "",
            "The feature/attention package provides selected case studies rather than a full benchmark. "
            "For each selected encoder/video pair, sampled frames show the visual trajectory, feature-change "
            "maps show where patch representations move between consecutive frames, and attention overlays "
            "show class-token focus when the model exposes attention weights.",
            "",
        ]
    )
    if montage_written:
        lines.append("- A compact case-study montage is saved as `figures/qualitative_case_montage.png`.")
    lines.extend(
        [
            "- DINOv2 and CLIP expose attention overlays in the current package.",
            "- SigLIP has patch feature-change maps, but no saved attention overlays in the current run.",
            "",
            "## What Is Still Missing Relative To The Original Plan",
            "",
            "The current benchmark strongly covers gravity-driven dynamics and temporal continuity. "
            "Collision and occlusion/object permanence are still mostly future work. The smallest useful "
            "extension would be to add two simulated tasks: a bouncing/colliding ball pair for contact-induced "
            "motion change, and a pendulum or moving ball passing behind an occluder for object permanence. "
            "These do not need to replace the pendulum work; they would let the final report explicitly echo "
            "all three scenario categories from the project plan.",
            "",
            "## Recommended Report Framing",
            "",
            "State that the project moved from uncontrolled prompt-generated videos to controlled simulations "
            "after discovering that prompt outputs were too noisy for reliable physical labels. Then present "
            "the simulated pendulum benchmark as the controlled core experiment, with AI-generated videos as "
            "pipeline motivation and collision/occlusion as planned extensions.",
            "",
        ]
    )
    (output_dir / "project_plan_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex_snippet(output_dir: Path, montage_written: bool) -> None:
    lines = [
        "% Auto-generated by generated_analysis/summarize_project_plan_results.py",
        "\\subsection{Connection To The Original Project Plan}",
        "The original project plan proposed a small-scale framework for testing whether videos preserve "
        "basic physical regularities over time. The present work keeps this aim but introduces controlled "
        "simulation as the main source of ground-truth dynamics. This change was made because early "
        "AI-generated videos were useful for pipeline validation but did not provide reliable labels for "
        "physical correctness.",
        "",
        "The resulting benchmark directly supports the embedding-based temporal analysis proposed in the "
        "plan. Correct and wrong pendulum dynamics are compared through latent trajectory metrics, pairwise "
        "trajectory separation, and selected patch-level feature and attention visualisations. Collision and "
        "occlusion/object permanence remain natural extensions for the final version of the benchmark.",
    ]
    if montage_written:
        lines.extend(
            [
                "",
                "\\begin{figure}[htbp]",
                "\\centering",
                "\\includegraphics[width=\\textwidth]{generated/simulated_pendulum/qualitative/qualitative_case_montage.png}",
                "\\caption{Selected qualitative case studies. Each row shows sampled frames, a patch feature-change map, and an attention overlay when available.}",
                "\\label{fig:qualitative-case-montage}",
                "\\end{figure}",
            ]
        )
    (output_dir / "project_plan_alignment.tex").write_text("\n".join(lines), encoding="utf-8")


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


def write_quantitative_latex_tables(
    asset_dir: Path,
    delta_rows: list[dict[str, str]],
    scene_delta_rows: list[dict[str, str]],
    auc_rows: list[dict[str, str]],
) -> None:
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


def write_quantitative_latex_figures(asset_dir: Path) -> None:
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
        r"\caption{Wrong-physics type breakdown. Abrupt interventions such as velocity kicks and gravity changes produce stronger latent movement than smoother perturbations.}",
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
        r"\caption{Model-scale comparisons for DINOv2 and CLIP. DINOv2 models consistently show positive step-distance deltas, but the effect is not monotonic with scale. CLIP-B/32 is more separable than CLIP-L/14 in this benchmark.}",
        r"\label{fig:sim-scale-comparison}",
        r"\end{figure}",
        "",
    ]
    (asset_dir / "figures.tex").write_text("\n".join(lines), encoding="utf-8")


def sync_quantitative_latex_assets(
    sim_root: Path,
    latex_asset_dir: Path,
    delta_rows: list[dict[str, str]],
    scene_delta_rows: list[dict[str, str]],
    auc_rows: list[dict[str, str]],
) -> None:
    latex_asset_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = latex_asset_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((sim_root / "figures").glob("*.png")):
        shutil.copy2(source, figure_dir / source.name)
    write_quantitative_latex_tables(latex_asset_dir, delta_rows, scene_delta_rows, auc_rows)
    write_quantitative_latex_figures(latex_asset_dir)
    (latex_asset_dir / "README.md").write_text(
        "Generated report assets synced from simulated_pendulum_analysis.tar.gz.\n",
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


def sync_latex_assets(output_dir: Path, latex_asset_dir: Path) -> None:
    latex_asset_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "project_plan_alignment.tex"
    shutil.copy2(source, latex_asset_dir / "project_plan_alignment.tex")
    montage = output_dir / "figures" / "qualitative_case_montage.png"
    if montage.exists():
        qdir = latex_asset_dir / "qualitative"
        qdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(montage, qdir / montage.name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        sim_root = tmp_root / "sim_analysis"
        feature_root = tmp_root / "feature_attention"
        extract_package(args.sim_analysis_package, sim_root)
        extract_package(args.feature_attention_package, feature_root)

        run_config = json.loads((sim_root / "run_config.json").read_text(encoding="utf-8"))
        auc_rows = read_csv(sim_root / "tables" / "auc_by_encoder.csv")
        delta_rows = read_csv(sim_root / "tables" / "encoder_wrong_correct_deltas.csv")
        scene_delta_rows = read_csv(sim_root / "tables" / "scene_wrong_correct_deltas.csv")
        wrong_type_rows = read_csv(sim_root / "tables" / "wrong_type_summary.csv")
        feature_rows = collect_feature_summaries(feature_root)
        plan_rows = build_plan_rows(run_config, feature_rows)

        write_csv(args.output_dir / "tables" / "plan_deliverable_mapping.csv", plan_rows)
        write_csv(args.output_dir / "tables" / "qualitative_examples.csv", feature_rows)
        montage_written = make_case_montage(feature_root, args.output_dir / "figures" / "qualitative_case_montage.png")

        write_report(
            args.output_dir,
            run_config,
            plan_rows,
            auc_rows,
            delta_rows,
            wrong_type_rows,
            feature_rows,
            montage_written,
        )
        write_latex_snippet(args.output_dir, montage_written)
        if args.latex_asset_dir:
            sync_quantitative_latex_assets(sim_root, args.latex_asset_dir, delta_rows, scene_delta_rows, auc_rows)
            sync_latex_assets(args.output_dir, args.latex_asset_dir)
        package_path = package_outputs(args.output_dir, args.package_name)

    print("Project-plan alignment summary complete.")
    print(f"Output: {args.output_dir}")
    print(f"Package: {package_path}")
    if args.latex_asset_dir:
        print(f"LaTeX assets: {args.latex_asset_dir}")


if __name__ == "__main__":
    main()
