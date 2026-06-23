# Generated Video Evaluation

This module is a small, independent MVP for evaluating physical consistency in
generated videos. It does not import or depend on the original repository's
training, planning, MuJoCo, D4RL, or dm_control code.

## Purpose

The goal is to turn a video into a frame-wise visual embedding trajectory and
measure whether that trajectory is smooth and continuous. A physically plausible
video should usually move through visual embedding space in a more coherent way.
Artifacts such as teleportation, sudden disappearance, shape collapse, abrupt
motion reversal, or collision mistakes can produce large jumps, high curvature,
or low straightness.

## Relation To Temporal Straightening

The original temporal-straightening idea studies whether useful visual
representations make temporal trajectories straighter for downstream control and
planning. Here, the same intuition is used as an evaluation proxy for generated
videos: if the generated sequence is physically consistent, its visual embedding
trajectory should be relatively smooth and straight.

This is not a full reproduction of the planning experiments. It is a standalone
evaluation pipeline for generated video files.

## Install

From the repository root:

```bash
python -m pip install -r generated_eval/requirements.txt
```

Main dependencies:

- `torch`
- `torchvision`
- `transformers`
- `pillow`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`

Run an environment check:

```bash
python generated_eval/00_check_env.py
```

## Convert Videos To MP4

The evaluation scripts expect `.mp4` files. If your generated videos are `.mov`,
`.avi`, `.webm`, `.mkv`, `.m4v`, or `.gif`, convert them first with ffmpeg:

```bash
python generated_eval/05_convert_to_mp4.py \
  --input_path raw_videos/bouncing.mov \
  --output_path generated_videos/bouncing/001.mp4
```

For a folder:

```bash
python generated_eval/05_convert_to_mp4.py \
  --input_path raw_videos \
  --output_dir generated_videos \
  --recursive
```

If `ffmpeg` is missing on Linux:

```bash
apt-get update && apt-get install -y ffmpeg
```

## Single Video

Recommended full pipeline:

```bash
python generated_eval/run_single_video_eval.py \
  --video_path generated_videos/test.mp4 \
  --video_id test \
  --output_root results \
  --num_frames 16
```

The same pipeline can be run step by step:

```bash
python generated_eval/01_extract_frames.py \
  --video_path generated_videos/test.mp4 \
  --output_dir results/frames/test \
  --num_frames 16 \
  --size 224

python generated_eval/02_extract_embeddings.py \
  --frame_dir results/frames/test \
  --output_path results/embeddings/test.npy \
  --model_name facebook/dinov2-base

python generated_eval/03_compute_metrics.py \
  --embedding_path results/embeddings/test.npy \
  --output_path results/metrics/test.json

python generated_eval/04_plot_metrics.py \
  --embedding_path results/embeddings/test.npy \
  --metrics_path results/metrics/test.json \
  --output_dir results/figures/test
```

## Batch Evaluation

For a directory like:

```text
generated_videos/
  projectile/
    001.mp4
    002.mp4
  collision/
    001.mp4
    002.mp4
```

run:

```bash
python generated_eval/run_batch_eval.py \
  --video_root generated_videos \
  --output_root results \
  --num_frames 16
```

The batch script recursively finds all `.mp4` files. The `scene_type` is the
parent folder name, such as `projectile` or `collision`. If one video fails, the
error is recorded in the CSV and the script continues with the next video.

## Output Structure

For `video_id=test`, outputs are:

```text
results/
  frames/
    test/
      frame_0000.png
      frame_0001.png
      ...
  embeddings/
    test.npy
  metrics/
    test.json
  figures/
    test/
      step_distance.png
      curvature.png
      pca_trajectory.png
  metrics_summary.csv
```

The embedding file has shape `[T, D]`, where `T` is the number of extracted
frames and `D` is the DINOv2 embedding dimension.

## Metrics

- `step_distances`: `||z_{t+1} - z_t||`. Large spikes can indicate sudden visual
  changes or discontinuities.
- `mean_step_distance`: average frame-to-frame embedding movement.
- `max_step_distance`: largest frame-to-frame embedding movement.
- `std_step_distance`: variability of frame-to-frame movement.
- `curvature`: `1 - cosine_similarity(z_{t+1}-z_t, z_{t+2}-z_{t+1})`. Higher
  values indicate sharper changes in trajectory direction.
- `mean_curvature`: average direction change along the trajectory.
- `max_curvature`: largest direction change along the trajectory.
- `straightness`: `||z_T - z_1|| / sum_t ||z_{t+1} - z_t||`. Higher values mean
  the trajectory is closer to a straight path in embedding space.

These are proxy metrics, not ground-truth physics labels. They are most useful
for comparing videos generated under the same prompt class, model, embedding
model, and frame sampling setting.
