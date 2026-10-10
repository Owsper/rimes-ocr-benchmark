"""
OCR benchmark script for RIMES-2011 English handwriting line dataset.
Model: EasyOCR (chordocr, English)

Usage:
    python easyocr_test.py --limit 100
    python easyocr_test.py --split test --limit 50 --results-dir results

Install (Python 3.11):
    pip install easyocr datasets pillow numpy
"""

import argparse
from pathlib import Path

import numpy as np

from ocr_eval import run_benchmark


class EasyOCRModel:
    def __init__(self, force_gpu: bool | None = None):
        try:
            import easyocr
        except ImportError as exc:
            raise SystemExit(
                "EasyOCR is not installed. Install it with: pip install easyocr"
            ) from exc
        # Auto-detect CUDA; allow explicit override via --gpu / --cpu.
        if force_gpu is None:
            try:
                import torch
                force_gpu = torch.cuda.is_available()
            except Exception:
                force_gpu = False
        self.gpu = force_gpu
        if not self.gpu:
            print("Note: no CUDA GPU detected — EasyOCR running on CPU (slow).")
        self.reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)

    def predict(self, image: np.ndarray) -> str:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        result = self.reader.readtext(image)
        return " ".join(item[1] for item in result if item[1]).strip()


def main():
    parser = argparse.ArgumentParser(description="Benchmark EasyOCR on RIMES-2011")
    parser.add_argument("--dataset", default="Teklia/RIMES-2011-line")
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="want_gpu", action="store_true",
                           help="Force GPU even if no CUDA is detected.")
    gpu_group.add_argument("--cpu", dest="want_gpu", action="store_false",
                           help="Force CPU even if a CUDA GPU is detected.")
    parser.set_defaults(want_gpu=None)
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
        model=EasyOCRModel(force_gpu=args.want_gpu),
        canonical_name="easyocr",
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