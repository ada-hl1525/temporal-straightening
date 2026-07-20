#!/usr/bin/env python
"""Extract frame-wise visual embeddings for a directory of image frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import fail, list_frame_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-wise visual embeddings.")
    parser.add_argument("--frame_dir", type=Path, required=True, help="Directory of png/jpg frames.")
    parser.add_argument("--output_path", type=Path, required=True, help="Output .npy path.")
    parser.add_argument(
        "--model_name",
        default="facebook/dinov2-base",
        help="Hugging Face model name. Default: facebook/dinov2-base",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size.")
    return parser.parse_args()


def select_embedding(outputs):
    """Return a [B, D] tensor from common Hugging Face vision model outputs."""
    image_embeds = getattr(outputs, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds

    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        if last_hidden_state.ndim == 2:
            return last_hidden_state
        if last_hidden_state.ndim == 3:
            return last_hidden_state[:, 0, :]
        if last_hidden_state.ndim == 4:
            return last_hidden_state.flatten(2).mean(dim=2)

    vision_output = getattr(outputs, "vision_model_output", None)
    if vision_output is not None:
        return select_embedding(vision_output)

    fail(
        "Model output has no supported embedding field. Expected one of "
        "image_embeds, pooler_output, last_hidden_state, or vision_model_output."
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        fail("--batch_size must be positive")

    frames = list_frame_paths(args.frame_dir)

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        fail(
            "Missing dependency for DINOv2 embeddings. Install with: "
            "python -m pip install torch transformers pillow"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_name}")
    print(f"Input frame count: {len(frames)}")

    try:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
    except Exception as exc:
        fail(
            "Failed to load model or processor. Check internet access, Hugging Face "
            f"cache, and model name '{args.model_name}'. Original error: {exc}"
        )

    model.to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frames), args.batch_size):
            batch_paths = frames[start : start + args.batch_size]
            images = []
            for path in batch_paths:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception as exc:
                    fail(f"Failed to read image frame {path}: {exc}")

            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            embeddings = select_embedding(outputs)

            all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32))

    embedding_array = np.concatenate(all_embeddings, axis=0)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, embedding_array)

    print(f"Output embedding shape: {embedding_array.shape}")
    print(f"Saved embeddings to: {args.output_path}")


if __name__ == "__main__":
    main()
