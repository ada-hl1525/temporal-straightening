# Generated Simulation Scripts

This folder contains lightweight OpenCV-based generators for controlled physics
videos.

## Pendulum Benchmark

```bash
python generated_simulation/generate_pendulum_dataset.py \
  --output_dir generated_videos/simulated_pendulum_v2 \
  --num_variants 12 \
  --width 512 \
  --height 512 \
  --fps 32 \
  --duration 4
```

## Project-Plan Extension Scenes

Generate controlled collision and occlusion videos:

```bash
python generated_simulation/generate_plan_extension_dataset.py \
  --output_dir generated_videos/simulated_plan_extensions \
  --num_variants 6 \
  --width 512 \
  --height 512 \
  --fps 32 \
  --duration 4
```

This creates 24 videos:

- 6 correct collision videos;
- 6 wrong collision videos;
- 6 correct occlusion videos;
- 6 wrong occlusion videos.

The wrong types are controlled:

- collision: no response, wrong direction, energy gain;
- occlusion: disappearance, wrong reappearance position, identity/color change.

Run a small encoder evaluation:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \
  --video_root generated_videos/simulated_plan_extensions \
  --output_root results/simulated_plan_extensions_small_encoders \
  --only dinov2_base clip_vit_base_patch32 siglip_base_patch16_224 \
  --num_frames 16 \
  --size 224 \
  --hf_endpoint https://hf-mirror.com
```

Run all configured encoders:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \
  --video_root generated_videos/simulated_plan_extensions \
  --output_root results/simulated_plan_extensions_multi_encoder \
  --num_frames 16 \
  --size 224 \
  --hf_endpoint https://hf-mirror.com
```

