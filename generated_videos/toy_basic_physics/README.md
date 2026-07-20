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
- `dinov2_large` (`facebook/dinov2-large`)
- `clip_vit_base_patch32` (`openai/clip-vit-base-patch32`)
- `clip_vit_large_patch14` (`openai/clip-vit-large-patch14`)
- `siglip_base_patch16_224` (`google/siglip-base-patch16-224`)
- `vit_base_imagenet21k` (`google/vit-base-patch16-224-in21k`)
- `mae_base` (`facebook/vit-mae-base`)
- `swin_base` (`microsoft/swin-base-patch4-window7-224`)

Run only selected encoders:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \
  --hf_endpoint https://hf-mirror.com \
  --only dinov2_large clip_vit_large_patch14 siglip_base_patch16_224
```

Run all five videos with the single baseline encoder only:

```bash
python generated_eval/run_cloud_batch_eval.py --hf_endpoint https://hf-mirror.com
```

Analyze and package the multi-encoder results:

```bash
python generated_eval/analyze_multi_encoder_results.py \
  --results_root results/toy_basic_physics_multi_encoder \
  --output_dir results/toy_basic_physics_multi_encoder_analysis
```

The final download package is written to:

```text
results/toy_basic_physics_multi_encoder_analysis/toy_multi_encoder_results_with_analysis.tar.gz
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
