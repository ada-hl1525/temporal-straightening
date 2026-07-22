# Toy Wrong Physics Videos

Small AI-generated video set intended for comparison against
`generated_videos/toy_basic_physics`.

These videos are not clean ground-truth physics violations. The video model did
not always follow the intended wrong-physics prompts exactly, so this set should
be treated as a weak-negative or failed-generation set.

## Contents

- `Pendulum002_EnergyGain.mp4`
- `ElasticCollision002_Stop.mp4`
- `Friction002_SpontaneousAcceleration.mp4`
- `RollingDownaSlope002_UphillAcceleration.mp4`
- `FreeFall002_Hovering.mp4`
- `METADATA.md`: human-readable notes.
- `metadata.csv`: structured metadata with `pair_id` for comparison.

## Evaluation

Run all wrong-physics videos through the multi-encoder sweep:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \
  --hf_endpoint https://hf-mirror.com \
  --video_root generated_videos/toy_wrong_physics \
  --output_root results/toy_wrong_physics_multi_encoder
```

Analyze and package the wrong-physics results:

```bash
python generated_eval/analyze_multi_encoder_results.py \
  --results_root results/toy_wrong_physics_multi_encoder \
  --output_dir results/toy_wrong_physics_multi_encoder_analysis \
  --package_name toy_wrong_physics_results_with_analysis.tar.gz
```

Run a focused CLIP/SigLIP retry if needed:

```bash
python generated_eval/run_cloud_multi_encoder_eval.py \
  --hf_endpoint https://hf-mirror.com \
  --video_root generated_videos/toy_wrong_physics \
  --output_root results/toy_wrong_physics_multi_encoder \
  --only clip_vit_base_patch32 clip_vit_large_patch14 siglip_base_patch16_224
```
