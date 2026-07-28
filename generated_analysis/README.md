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
  --output_dir results/simulated_pendulum_analysis
```

Outputs:

- `tables/combined_metrics.csv`
- `tables/encoder_wrong_correct_deltas.csv`
- `tables/auc_by_encoder.csv`
- `figures/metric_boxplots.png`
- `figures/encoder_step_deltas.png`
- `figures/delta_heatmap.png`
- `figures/pca_trajectories.png`
- `figures/step_distance_timeseries.png`
- `analysis_report.md`
- `simulated_pendulum_analysis.tar.gz`

This script does not re-run any encoder. It uses the existing `.npy` embeddings
and `metrics_summary.csv` files.

## 2. Feature Map And Attention Inspection

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
