#!/usr/bin/env python
"""Visualize patch feature changes and attention maps for a single video.

This is a qualitative inspection tool. It re-runs a Hugging Face vision encoder
on sampled frames and writes:

- sampled RGB frames
- frame-to-frame patch feature change heatmaps
- CLS-to-patch attention overlays when the model exposes attentions
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize feature maps and attention maps.")
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--model_name", default="facebook/dinov2-small")
    parser.add_argument("--output_dir", type=Path, default=Path("results/feature_attention_inspection"))
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for auto.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sample_video_frames(video_path: Path, num_frames: int, size: int) -> list[np.ndarray]:
    if not video_path.exists():
        fail(f"video_path does not exist: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        fail(f"video has no readable frames: {video_path}")
    indices = np.linspace(0, total - 1, num_frames).round().astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame_bgr = capture.read()
        if not ok:
            fail(f"failed to read frame {index} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    capture.release()
    return frames


def save_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = normalize_map(heatmap)
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_u8 = cv2.resize(heatmap_u8, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    colour = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_INFERNO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * frame_rgb + alpha * colour, 0, 255))


def get_vision_model(model):
    return getattr(model, "vision_model", model)


def get_hidden_and_attention(model, inputs):
    vision_model = get_vision_model(model)
    outputs = vision_model(
        pixel_values=inputs["pixel_values"],
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else outputs.last_hidden_state
    attentions = getattr(outputs, "attentions", None)
    attention = attentions[-1] if attentions else None
    return hidden, attention


def patch_grid_size(num_tokens: int) -> tuple[int, int, bool]:
    with_cls = False
    patch_tokens = num_tokens
    side = int(round(math.sqrt(patch_tokens)))
    if side * side == patch_tokens:
        return side, side, with_cls
    patch_tokens = num_tokens - 1
    side = int(round(math.sqrt(patch_tokens)))
    if side * side == patch_tokens:
        return side, side, True
    fail(f"cannot infer square patch grid from token count {num_tokens}")


def run_model(frames: list[np.ndarray], model_name: str, device_arg: str):
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise SystemExit(
            "ERROR: missing torch/transformers/pillow. Install generated_eval requirements first."
        ) from exc

    device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:
        model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    pil_images = [Image.fromarray(frame) for frame in frames]
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        hidden, attention = get_hidden_and_attention(model, inputs)
    return hidden.detach().cpu().numpy(), None if attention is None else attention.detach().cpu().numpy()


def feature_change_maps(hidden: np.ndarray) -> list[np.ndarray]:
    _, token_count, _ = hidden.shape
    grid_h, grid_w, has_cls = patch_grid_size(token_count)
    patch_hidden = hidden[:, 1:, :] if has_cls else hidden
    diffs = np.linalg.norm(np.diff(patch_hidden, axis=0), axis=2)
    return [diff.reshape(grid_h, grid_w) for diff in diffs]


def attention_maps(attention: np.ndarray | None) -> list[np.ndarray]:
    if attention is None:
        return []
    # [B, heads, tokens, tokens], average heads, take CLS query to patch tokens.
    batch, _heads, token_count, _ = attention.shape
    grid_h, grid_w, has_cls = patch_grid_size(token_count)
    if not has_cls:
        return []
    averaged = attention.mean(axis=1)
    maps = []
    for index in range(batch):
        cls_to_patch = averaged[index, 0, 1:]
        maps.append(cls_to_patch.reshape(grid_h, grid_w))
    return maps


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = sample_video_frames(args.video_path, args.num_frames, args.size)
    hidden, attention = run_model(frames, args.model_name, args.device)

    frame_dir = args.output_dir / "frames"
    feature_dir = args.output_dir / "feature_change"
    attention_dir = args.output_dir / "attention"
    for index, frame in enumerate(frames):
        save_rgb(frame_dir / f"frame_{index:02d}.png", frame)

    for index, heatmap in enumerate(feature_change_maps(hidden), start=1):
        overlay = overlay_heatmap(frames[index], heatmap)
        save_rgb(feature_dir / f"feature_change_to_frame_{index:02d}.png", overlay)

    attn_maps = attention_maps(attention)
    for index, heatmap in enumerate(attn_maps):
        overlay = overlay_heatmap(frames[index], heatmap)
        save_rgb(attention_dir / f"attention_frame_{index:02d}.png", overlay)

    print(f"Video: {args.video_path}")
    print(f"Model: {args.model_name}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"Attention shape: {None if attention is None else attention.shape}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
