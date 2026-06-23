#!/usr/bin/env python
"""Extract DINOv2 embeddings for a directory of image frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import fail, list_frame_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-wise DINOv2 embeddings.")
    parser.add_argument("--frame_dir", type=Path, required=True, help="Directory of png/jpg frames.")
    parser.add_argument("--output_path", type=Path, required=True, help="Output .npy path.")
    parser.add_argument(
        "--model_name",
        default="facebook/dinov2-base",
        help="Hugging Face model name. Default: facebook/dinov2-base",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size.")
    return parser.parse_args()


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
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                fail("Model output has neither last_hidden_state nor pooler_output.")

            all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32))

    embedding_array = np.concatenate(all_embeddings, axis=0)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, embedding_array)

    print(f"Output embedding shape: {embedding_array.shape}")
    print(f"Saved embeddings to: {args.output_path}")


if __name__ == "__main__":
    main()

