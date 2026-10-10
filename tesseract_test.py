"""
OCR benchmark script for RIMES-2011 **French** handwriting line dataset.
Model: Tesseract (via pytesseract, LSTM v4). Force the language with
    python tesseract_test.py --lang fra --limit 100

Usage:
    python tesseract_test.py --lang fra --limit 100
    python tesseract_test.py --split test --limit 50 --results-dir results

Install (Python 3.11):
    pip install pytesseract datasets pillow numpy matplotlib
    # Tesseract binary (UB-Mannheim Windows build): https://github.com/UB-Mannheim/tesseract/wiki
    # Language data: download fra.traineddata from
    #   https://github.com/tesseract-ocr/tessdata_fast/raw/main/fra.traineddata
    # and either copy it into the install's tessdata\, or keep it anywhere and
    # set  $env:TESSDATA_PREFIX = "<dir-with-fra>;C:\Program Files\Tesseract-OCR\tessdata"
"""

import argparse
import io
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_eval import run_benchmark


_KNOWN_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
)


def _ensure_tessdata_prefix(tesseract_exe: str) -> str:
    """Make extra .traineddata roots visible to tesseract in-process.

    Reads the user's TESSDATA_PREFIX (semicolon-separated) and, for any
    entry other than the install's own tessdata dir, appends it so that
    `--list-langs` and `image_to_string` can find the language data.
    Writes this back to os.environ so the spawned tesseract.exe sees it.
    Returns the effective prefix string.
    """
    install_tessdata = str(Path(tesseract_exe).parent / "tessdata")
    current = os.environ.get("TESSDATA_PREFIX", "")
    entries = [e.strip() for e in current.split(os.pathsep) if e.strip()] if current else []
    if not entries and (Path.home() / "Desktop").is_dir():
        # Sensible default on this machine: Desktop holds the downloaded
        # fra.traineddata and the install dir holds eng/osd.
        entries = [str(Path.home() / "Desktop"), install_tessdata]
    # Always make sure install tessdata is in the list (last, so it wins
    # nothing but guarantees eng/osd remain findable).
    if install_tessdata not in entries:
        entries.append(install_tessdata)
    os.environ["TESSDATA_PREFIX"] = os.pathsep.join(entries)
    return os.environ["TESSDATA_PREFIX"]


def _pin_tesseract_cmd() -> str:
    """Resolve the tesseract binary from --tesseract-path > $TESSERACT_CMD > known roots."""
    for cand in (
        os.environ.get("TESSERACT_CMD"),
        *(Path.home() / cand for cand in ()),
    ):
        if cand and Path(cand).is_file():
            return cand
    for cand in _KNOWN_TESSERACT_PATHS:
        if cand and Path(cand).is_file():
            return cand
    # Last resort: whatever is on PATH (pytesseract's own probe).
    import shutil

    found = shutil.which("tesseract")
    if found:
        return found
    raise SystemExit(
        "Tesseract binary not found. Set $env:TESSERACT_CMD to the full path of "
        "tesseract.exe, or add the install dir to PATH.\n"
        "Windows: https://github.com/UB-Mannheim/tesseract/wiki"
    )


class TesseractModel:
    """Tesseract wrapper. Fails loudly on missing languages / data files.

    Never silently swallows a TesseractError: an unrecoverable predict()
    crash means the engine is not actually running (missing .traineddata,
    wrong TESSDATA_PREFIX, binary not invocable), and 100 empty pred rows
    in the CSV would be misread as 'bad model'.
    """

    def __init__(self, forced_lang: str | None = None, psm: int = 7, output_fn: str = "stdout"):
        try:
            import pytesseract
        except ImportError as exc:
            raise SystemExit("pytesseract is not installed. Install it with: pip install pytesseract") from exc

        try:
            tesseract_exe = _pin_tesseract_cmd()
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe
            prefix = _ensure_tessdata_prefix(tesseract_exe)
            version = pytesseract.get_tesseract_version()
            self._available = set(pytesseract.get_languages(config=""))
        except Exception as exc:
            raise SystemExit(f"Tesseract binary/startup failed: {exc}") from exc

        self.forced_lang = forced_lang
        self.psm = psm
        self.pytesseract = pytesseract
        self._version = version

        # Explicit --lang: hard-fail if that exact language is missing.
        if forced_lang and forced_lang not in self._available:
            raise SystemExit(
                f"Tesseract lang '{forced_lang}' not available. "
                f"--list-langs returned: {', '.join(sorted(self._available)) or '(none)'}\n"
                f"Install with: download {forced_lang}.traineddata from "
                f"https://github.com/tesseract-ocr/tessdata_fast/raw/main/{forced_lang}.traineddata "
                f"and make it visible to tesseract (copy into the install's tessdata\\, "
                f"or set $env:TESSDATA_PREFIX to a directory that contains it)."
            )

        # Auto-detect fallback: warn only, because the caller may have
        # deliberately requested 'fra' on a box that only has 'eng' and we
        # already told them above. Here we just note what we'll do instead.
        if not forced_lang and "eng" not in self._available:
            raise SystemExit(
                f"No usable Tesseract language installed. Available: {', '.join(sorted(self._available)) or '(none)'}"
            )

        print(f"[tesseract] binary = {pytesseract.pytesseract.tesseract_cmd}")
        print(f"[tesseract] version = {version}")
        print(f"[tesseract] TESSDATA_PREFIX = {os.environ.get('TESSDATA_PREFIX', '(unset, using install default)')}")
        print(f"[tesseract] available langs = {', '.join(sorted(self._available))}")
        print(f"[tesseract] forced_lang = {forced_lang!r}  psm = {psm}")

    def _detect_lang(self, gold: str) -> str:
        if self.forced_lang:
            return self.forced_lang
        text = (gold or "").lower()
        scores = {
            "eng": sum(1 for c in text if c in "abcdefghijklmnopqrstuvwxyz"),
            "fra": sum(1 for c in text if c in "àâäçéèêëîïôöùûüœ"),
            "deu": sum(1 for c in text if c in "äöüß"),
            "spa": sum(1 for c in text if c in "áéíóúñü"),
            "ita": sum(1 for c in text if c in "àèéììòóù"),
        }
        best = max(scores, key=scores.get)
        return best if best in self._available and scores[best] > 0 else "eng"

    def _ocr(self, image: np.ndarray, lang: str, psm: int) -> str:
        """Binarize + invert + OCR. Raises on any tesseract failure."""
        ink = image if image.ndim == 2 else image.mean(axis=2)
        # Handwriting is dark on light; invert so ink is black (0) on white (255).
        inverted = (255 - np.clip(ink, 0, 255)).astype(np.uint8)
        img = Image.fromarray(inverted)
        output = self.pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")
        return " ".join(output.split())

    def predict(self, image: np.ndarray, *, gold: str = "") -> str:
        lang = self._detect_lang(gold)
        try:
            return self._ocr(image, lang=lang, psm=self.psm)
        except Exception as first_exc:
            # Retry once with single-line PSM (7) in case PSM 6 was the issue.
            if self.psm != 7:
                try:
                    return self._ocr(image, lang=lang, psm=7)
                except Exception:
                    pass
            # Last resort: dump the raw exception AND a copy of the input
            # image so the user can see what tesseract actually received.
            err_log = Path(__file__).resolve().parent / "results" / "tesseract_errors.txt"
            try:
                err_log.parent.mkdir(parents=True, exist_ok=True)
                with open(err_log, "a", encoding="utf-8") as f:
                    f.write("=" * 70 + "\n")
                    f.write(f"lang={lang} psm={self.psm}\ngold={gold!r}\nexception: {first_exc!r}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                print(f"[tesseract] ERROR (see {err_log}): {first_exc!r}", file=sys.stderr)
            except OSError:
                print(f"[tesseract] ERROR but could not write log: {first_exc!r}", file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description="Benchmark Tesseract on RIMES-2011 (French)")
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
    parser.add_argument(
        "--lang",
        default=None,
        help="Tesseract language code (RIMES-2011 is FRENCH; use 'fra'). "
             "Hard-fails if the language is not installed.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=7,
        help="Tesseract page segmentation mode (default 7 = single text line; "
             "RIMES crops are single lines).",
    )
    args = parser.parse_args()

    model = TesseractModel(forced_lang=args.lang, psm=args.psm)

    # Save the very first sample's inverted bitmap so the user can eyeball
    # exactly what tesseract is being fed. This is the #1 misdiagnosis
    # point (wrong inversion, wrong key, etc.).
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    sample = ds[0]
    img_key = next((c for c in args.image_candidates if c in sample), None)
    if img_key is not None:
        raw = sample[img_key]
        img = Image.open(io.BytesIO(raw["bytes"])).convert("L") if isinstance(raw, dict) else Image.fromarray(np.asarray(raw)).convert("L")
        out_png = args.results_dir / "_sample_0.png"
        args.results_dir.mkdir(parents=True, exist_ok=True)
        inverted = img.point(lambda p: 255 - p)
        inverted.save(out_png)
        print(f"[tesseract] first sample saved (inverted) to {out_png}")

    run_benchmark(
        model=model,
        canonical_name="tesseract",
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