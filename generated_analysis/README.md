# Simulated Pendulum Analysis Pipeline

This folder contains report-oriented analysis scripts for the controlled
single-pendulum and double-pendulum videos.

## 1. Latent Trajectory And Metric Analysis

Run this after `generated_eval/run_cloud_multi_encoder_eval.py` has produced one
directory per encoder.

```bash
python generated_analysis/analyze_simulated_pendulum.py \
  --results_root results/pybullet_pendulum_multi_encoder \
  --metadata generated_videos/simulated_pendulum/metadata.csv \
  --output_dir results/simulated_pendulum_analysis \
  --latex_asset_dir reports/irp-template-main/latex/generated/simulated_pendulum
```

Outputs:

- `tables/combined_metrics.csv`
- `tables/encoder_wrong_correct_deltas.csv`
- `tables/auc_by_encoder.csv`
- `tables/pairwise_separation.csv`
- `tables/pairwise_separation_summary.csv`
- `figures/metric_boxplots.png`
- `figures/pairwise_separation.png`
- `figures/wrong_type_step_distance.png`
- `figures/encoder_step_deltas.png`
- `figures/delta_heatmap.png`
- `figures/pca_trajectories.png`
- `figures/step_distance_timeseries.png`
- `analysis_report.md`
- `simulated_pendulum_analysis.tar.gz`
- optional LaTeX-ready assets under `--latex_asset_dir`

This script does not re-run any encoder. It uses the existing `.npy` embeddings
and `metrics_summary.csv` files.

The pairwise separation analysis compares:

- correct-correct pairs, to estimate normal variation between valid simulations;
- correct-wrong pairs, to test whether wrong physics exceeds that variation;
- wrong-wrong pairs, to inspect how diverse the injected failures are.

## Larger Simulated Dataset

The current 16-video set is useful for smoke testing, but a stronger report
should use more controlled variants. Generate a larger set with:

```bash
python generated_simulation/generate_pendulum_dataset.py \
  --output_dir generated_videos/simulated_pendulum_v2 \
  --num_variants 12 \
  --width 512 \
  --height 512 \
  --fps 32 \
  --duration 4
```

This creates 48 videos:

- 12 correct single-pendulum videos;
- 12 wrong single-pendulum videos;
- 12 correct double-pendulum videos;
- 12 wrong double-pendulum videos.

Then evaluate:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \
  --video_root generated_videos/simulated_pendulum_v2 \
  --output_root results/simulated_pendulum_v2_multi_encoder \
  --num_frames 16 \
  --size 224 \
  --hf_endpoint https://hf-mirror.com
```

## 2. Single-Video Feature Map And Attention Inspection

Run this for selected videos and selected models. It re-runs the encoder and
saves qualitative visualisations.

Example:

```bash
python generated_analysis/visualize_feature_attention.py \
  --video_path generated_videos/simulated_pendulum/single_pendulum/single_pendulum_single_periodic_kick_003.mp4 \
  --model_name facebook/dinov2-small \
  --output_dir results/feature_attention/dinov2_small_single_periodic_kick \
  --num_frames 8 \
  --size 224
```

Recommended qualitative comparisons:

```bash
python generated_analysis/visualize_feature_attention.py \
  --video_path generated_videos/simulated_pendulum/single_pendulum/single_pendulum_correct_001.mp4 \
  --model_name facebook/dinov2-small \
  --output_dir results/feature_attention/dinov2_small_single_correct

python generated_analysis/visualize_feature_attention.py \
  --video_path generated_videos/simulated_pendulum/single_pendulum/single_pendulum_single_periodic_kick_003.mp4 \
  --model_name facebook/dinov2-small \
  --output_dir results/feature_attention/dinov2_small_single_periodic_kick

python generated_analysis/visualize_feature_attention.py \
  --video_path generated_videos/simulated_pendulum/double_pendulum/double_pendulum_correct_001.mp4 \
  --model_name openai/clip-vit-base-patch32 \
  --output_dir results/feature_attention/clip_base_double_correct

python generated_analysis/visualize_feature_attention.py \
  --video_path generated_videos/simulated_pendulum/double_pendulum/double_pendulum_double_reverse_gravity_after_half_002.mp4 \
  --model_name openai/clip-vit-base-patch32 \
  --output_dir results/feature_attention/clip_base_double_reverse_gravity
```

Outputs:

- sampled frames
- patch feature-change overlays
- attention overlays, when the model exposes CLS-to-patch attention

## 3. Batch Qualitative Examples

Run this to generate a small set of report-ready qualitative examples across
selected videos and encoders:

```bash
python generated_analysis/run_feature_attention_examples.py \
  --output_root results/feature_attention_examples \
  --num_frames 8 \
  --size 224
```

To run only the cheapest example first:

```bash
python generated_analysis/run_feature_attention_examples.py \
  --only_models dinov2_small \
  --only_videos single_correct single_periodic_kick \
  --output_root results/feature_attention_examples_smoke \
  --num_frames 8 \
  --size 224
```

The batch script writes:

- one folder per model and video;
- a `README.md` index;
- `feature_attention_examples.tar.gz` for downloading.

## 4. Connect Results Back To The Project Plan

After the quantitative and qualitative packages exist, run:

```bash
python generated_analysis/summarize_project_plan_results.py \
  --sim_analysis_package results/simulated_pendulum_analysis.tar.gz \
  --feature_attention_package results/feature_attention_examples.tar.gz \
  --output_dir results/project_plan_alignment \
  --latex_asset_dir reports/irp-template-main/latex/generated/simulated_pendulum
```

This produces:

- `results/project_plan_alignment/project_plan_alignment_report.md`
- `results/project_plan_alignment/tables/plan_deliverable_mapping.csv`
- `results/project_plan_alignment/tables/qualitative_examples.csv`
- `results/project_plan_alignment/figures/qualitative_case_montage.png`
- `results/project_plan_alignment/project_plan_alignment.tex`
- `results/project_plan_alignment/project_plan_alignment.tar.gz`

It also refreshes the LaTeX assets under
`reports/irp-template-main/latex/generated/simulated_pendulum` from the packaged
results, so the report uses the latest tables and figures rather than stale
local outputs.

## Report Logic

Use the quantitative script for the main results:

- correct vs wrong latent trajectory metrics
- across-model-family comparison
- across-model-scale comparison
- per-scene and per-wrong-type behaviour

Use the feature/attention script for qualitative evidence:

- whether patch-level changes are concentrated around the pendulum bob and rod
- whether attention is physically relevant
- whether model scale or model family changes the spatial focus
