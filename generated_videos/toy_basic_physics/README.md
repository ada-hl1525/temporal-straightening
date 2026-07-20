# Toy Basic Physics Videos

Small AI-generated video set for smoke-testing the generated video evaluation pipeline.

## Contents

- `Pendulum001.mp4`
- `ElasticCollision001.mp4`
- `Friction001.mp4`
- `RollingDownaSlope001.mp4`
- `FreeFall001.mp4`
- `METADATA.md`: human-readable prompts and notes.
- `metadata.csv`: structured metadata for scripts and analysis.

## Evaluation

From the repository root, run all five videos through multiple encoders:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py --hf_endpoint https://hf-mirror.com
```

By default this runs:

- `dinov2_small` (`facebook/dinov2-small`)
- `dinov2_base` (`facebook/dinov2-base`)

Run only selected encoders:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \
  --hf_endpoint https://hf-mirror.com \
  --only dinov2_small
```

Run all five videos with the single baseline encoder only:

```bash
python generated_eval/run_cloud_batch_eval.py --hf_endpoint https://hf-mirror.com
```

Run a single video:

```bash
python generated_eval/run_cloud_single_eval.py
```

Or override the defaults without editing the script:

```bash
python generated_eval/run_cloud_single_eval.py \
  --video_path generated_videos/toy_basic_physics/Pendulum001.mp4 \
  --video_id Pendulum001_baseline
```

The underlying batch command is:

```bash
python generated_eval/run_batch_eval.py \
  --video_root generated_videos/toy_basic_physics \
  --output_root results/toy_basic_physics_baseline \
  --num_frames 16 \
  --size 224 \
  --model_name facebook/dinov2-base \
  --batch_size 16
```
