# Generated Video Physical Consistency Evaluation MVP

This folder is intentionally independent from the original temporal-straightening
training, planning, MuJoCo, and D4RL code.

The MVP evaluates whether a generated video follows a smooth trajectory in visual
embedding space. Physically inconsistent generations, such as teleportation,
sudden disappearance, severe deformation, abrupt direction changes, or collision
artifacts, often appear as large embedding steps, high second-order variation, or
low straightness.

## Install

From the repository root:

```bash
python -m pip install -r generated_eval/requirements.txt
```

If you only want the dependency-light baseline encoder, `torch` and `torchvision`
are optional:

```bash
python -m pip install numpy opencv-python matplotlib scikit-learn
```

## Run On One Video

```bash
python generated_eval/evaluate_videos.py \
  --input /path/to/video.mp4 \
  --output-dir generated_eval/outputs \
  --encoder raw
```

## Run On A Folder Of Videos

```bash
python generated_eval/evaluate_videos.py \
  --input /path/to/videos \
  --output-dir generated_eval/outputs \
  --encoder raw
```

For stronger semantic embeddings, use ImageNet ResNet18 features:

```bash
python generated_eval/evaluate_videos.py \
  --input /path/to/videos \
  --output-dir generated_eval/outputs \
  --encoder resnet18 \
  --weights imagenet
```

`--weights imagenet` uses torchvision's cached weights if available and may
download them if not. Use `--weights none` to avoid network access.

## Outputs

For each video, the script writes:

- `frames/frame_000000.jpg`, `frames/frame_000001.jpg`, ...
- `embeddings.npy`, shape `[T, D]`
- `metrics.json`
- `plots/step_distance.png`
- `plots/curvature.png`
- `plots/pca_trajectory.png`

For folder input, it also writes:

- `metrics_summary.csv`

## Metric Interpretation

- `straightness`: endpoint distance divided by total path length. Higher is
  straighter and usually smoother.
- `step_distance_mean`, `step_distance_max`, `step_distance_cv`: frame-to-frame
  embedding motion. Large spikes can indicate discontinuities.
- `second_diff_mean`, `second_diff_max`: acceleration-like embedding changes.
  Higher values often indicate jerkiness.
- `turn_angle_mean_deg`, `turn_angle_max_deg`: direction changes in embedding
  trajectory. Large values suggest abrupt motion or identity changes.
- `curvature_mean`, `curvature_max`: turn angle normalized by local step length.
- `discontinuity_ratio_max_to_median`: max step divided by median step. Higher
  values flag sudden jumps.
