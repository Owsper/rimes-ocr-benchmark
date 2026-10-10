"""
OCR benchmark script for RIMES-2011 English handwriting line dataset.
Model: PaddleOCR (PP-OCRv4/v5)

Usage:
    python paddleocr_test.py --limit 100
    python paddleocr_test.py --split test --limit 50 --results-dir results

Install (Python 3.11):
    pip install paddlepaddle paddleocr datasets pillow numpy
"""

import argparse
from pathlib import Path

import numpy as np
from typing import Any

from ocr_eval import run_benchmark


class PaddleOCRModel:
    def __init__(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise SystemExit(
                "PaddleOCR is not installed. Install it with: pip install paddleocr"
            ) from exc
        # PaddleOCR 3.x API; falls back to position-compatible 2.x pattern
        # (PaddleOCR(...) == PaddleOCR3 in both) and uses predict in >=3.0
        # with OCRResult structure. We use the high-level .ocr() when
        # available (3.x exposes it for backward-compat) and fall back to
        # .predict + result parsing when needed.
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            enable_mkldnn=False,
        )
        # PaddleOCR 3.x exposes only .predict(); 2.x exposes only .ocr().
        self._use_legacy_api = hasattr(self.ocr, "ocr") and not hasattr(self.ocr, "predict")

    def _parse_legacy(self, result: Any) -> str:
        if not result:
            return ""
        texts = []
        for page in result:
            if not page:
                continue
            for line in page:
                if line is None:
                    continue
                try:
                    box_info, (text, _score) = line
                    texts.append(text)
                except (TypeError, ValueError):
                    continue
        return " ".join(texts).strip()

    def _parse_new_api(self, result: Any) -> str:
        if not result:
            return ""
        texts = []
        # PaddleOCR 3.x returns list of OCRResult objects with rec_texts
        for item in result:
            rec_texts = getattr(item, "rec_texts", None)
            if rec_texts is None and isinstance(item, dict):
                rec_texts = item.get("rec_texts")
            if rec_texts:
                texts.extend(str(t) for t in rec_texts)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
        return " ".join(texts).strip()

    def predict(self, image: np.ndarray) -> str:
        # PaddleOCR expects 3-channel RGB; convert grayscale to RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if self._use_legacy_api:
            result = self.ocr.ocr(image, cls=True)
            return self._parse_legacy(result)
        result = self.ocr.predict(
            image,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        return self._parse_new_api(result)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PaddleOCR on RIMES-2011")
    parser.add_argument("--dataset", default="Teklia/RIMES-2011-line")
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
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
        model=PaddleOCRModel(),
        canonical_name="paddleocr",
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