"""
OCR benchmark script for RIMES-2011 English handwriting line dataset.
Model: TrOCR (microsoft/trocr-base-handwritten) — a Transformer-based
handwriting recognition model from Microsoft, purpose-built for handwritten
line images such as the RIMES-2011 crop used here.

Usage:
    python trocr_test.py --limit 100
    python trocr_test.py --split test --limit 50 --results-dir results

Install (Python 3.11):
    pip install torch torchvision transformers accelerate datasets pillow numpy
    pip install matplotlib   # needed for the per-model graph; skipped automatically if --no-graph
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_eval import run_benchmark


class TrOCRModel:
    def __init__(self, model_id: str = "microsoft/trocr-base-handwritten"):
        try:
            import torch
            from transformers import (
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )
        except ImportError as exc:
            raise SystemExit(
                "Transformers is not installed. Install it with:\n"
                "    pip install torch torchvision transformers accelerate"
            ) from exc
        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Note: CUDA not available — using CPU (will be slow).")
        self.device = device
        print(f"Loading {model_id} on {device}...")
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def predict(self, image: np.ndarray) -> str:
        # RIMES line crops are grayscale; TrOCR expects 3-channel RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        img = Image.fromarray(image.astype(np.uint8), mode="RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_length=60, num_beams=1)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()




TexTellerModel = TrOCRModel  # legacy alias


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TrOCR on RIMES-2011"
    )
    parser.add_argument("--dataset", default="Teklia/RIMES-2011-line")
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--model",
        default="microsoft/trocr-base-handwritten",
        help="HF model id / local repo path for the TrOCR encoder-decoder.",
    )
    parser.add_argument(
        "--image-candidates",
        nargs="+",
        default=["image", "line_image", "line_img", "img", "word"],
    )
    parser.add_argument(
        "--text-candidates",
        nargs="+",
        default=["line", "transcription", "text", "label", "word"],
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip saving the per-model accuracy PNG (requires matplotlib).",
    )
    args = parser.parse_args()
    run_benchmark(
        model=TrOCRModel(args.model),
        canonical_name="trocr",
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        results_dir=args.results_dir,
        image_candidates=args.image_candidates,
        text_candidates=args.text_candidates,
        no_graph=args.no_graph,
    )


if __name__ == "__main__":
    main()