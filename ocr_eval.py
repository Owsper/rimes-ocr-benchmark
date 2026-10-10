"""
Shared OCR benchmark utilities for RIMES-2011 handwriting line dataset.

Every per-model test script (paddleocr_test.py, tesseract_test.py,
easyocr_test.py, texthelper_test.py, deepseek_ocr_test.py, got_ocr_test.py,
latex_ocr_test.py) reuses this helper so all models are evaluated
identically: same dataset, same metrics (CER / WER / Exact-match / latency)
and same CSV output layout in ./results.

Python 3.11 compatible.
"""

import csv
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence
import io
from PIL import Image

import numpy as np
from datasets import load_dataset


def slugify(name: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", name.strip().lower()).strip("-")
    return slug[:max_len] or "model"


def as_image_array(item: Any) -> np.ndarray:
    if isinstance(item, dict) and "bytes" in item:
        img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        return np.asarray(img)
    if isinstance(item, dict):
        src = item.get("bytes") or item.get("source")
        img = Image.open(io.BytesIO(src)).convert("RGB")
        return np.asarray(img)
    return np.asarray(item)


def text_key_from(item: Any, candidates: Iterable[str]) -> str:
    if isinstance(item, dict):
        for key in candidates:
            if key in item and item[key] is not None:
                return str(item[key])
    return str(item)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n] / (m or 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref and not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n] / (m or 1)


def exact_match(reference: str, hypothesis: str) -> bool:
    return normalize_text(reference) == normalize_text(hypothesis)


def pick_feature(
    dataset_dict: Any,
    image_candidates: Sequence[str],
    text_candidates: Sequence[str],
) -> tuple[str, str]:
    sample = dataset_dict[0]
    image_key = next((c for c in image_candidates if c in sample), None)
    if image_key is None:
        image_key = next(
            (
                key
                for key, value in sample.items()
                if isinstance(value, dict) and ("bytes" in value or "source" in value)
            ),
            None,
        )
    text_key = next((c for c in text_candidates if c in sample), None)
    if text_key is None:
        text_key = next(
            (
                key
                for key, value in sample.items()
                if isinstance(value, str)
                and key not in {image_key}
                and "id" not in key.lower()
            ),
            None,
        )
    if not image_key or not text_key:
        columns = sorted(sample.keys())
        raise SystemExit(
            f"Could not auto-detect image/text columns in RIMES-2011. Columns found: {columns}"
        )
    return image_key, text_key


def save_graph(
    results: list[dict[str, Any]],
    canonical_name: str,
    split: str,
    results_dir: Path,
) -> Path | None:
    """Save a 2-panel PNG to results_dir/<slug>_<split>.png.

    Panel A: per-sample accuracy (char-accuracy = 1 - CER, word-exact = EM).
    Panel B: cumulative CER / WER / EM (rolling mean).
    Returns the path on success, or None if matplotlib is missing (with a hint).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping PNG. Install with: pip install matplotlib")
        return None

    ids = [r["id"] for r in results]
    cer = [r["cer"] for r in results]
    wer = [r["wer"] for r in results]
    em = [r["em"] for r in results]
    char_acc = [1.0 - c for c in cer]
    word_acc = em

    def _rolling(values):
        out, s = [], 0.0
        for i, v in enumerate(values, start=1):
            s += v
            out.append(s / i)
        return out

    cum_cer = _rolling(cer)
    cum_wer = _rolling(wer)
    cum_em = _rolling(em)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ids, char_acc, marker="o", markersize=3, linestyle="-",
             label="Char accuracy (1-CER)", color="#2a7de1")
    ax1.plot(ids, word_acc, marker="s", markersize=3, linestyle="--",
             label="Word EM accuracy", color="#e07b2a")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Accuracy (0-1)")
    ax1.set_title(f"{canonical_name} — per-sample accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ids, cum_cer, marker="o", markersize=3, label="Cumulative CER", color="#c0392b")
    ax2.plot(ids, cum_wer, marker="s", markersize=3, label="Cumulative WER", color="#8e44ad")
    ax2.plot(ids, cum_em, marker="^", markersize=3, label="Cumulative EM", color="#27ae60")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Cumulative metric")
    ax2.set_title(f"{canonical_name} — cumulative CER / WER / EM")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    mean_cer = sum(cer) / len(cer)
    mean_wer = sum(wer) / len(wer)
    mean_em = sum(em) / len(em)
    fig.suptitle(
        f"{canonical_name}  (n={len(ids)}, split={split})   "
        f"mean CER={mean_cer*100:.2f}%  WER={mean_wer*100:.2f}%  EM={mean_em*100:.2f}%",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    results_dir.mkdir(parents=True, exist_ok=True)
    out_png = results_dir / f"{slugify(canonical_name)}_{split}.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def run_benchmark(
    model,
    canonical_name: str,
    dataset_name: str,
    split: str,
    limit: int,
    results_dir: Path,
    image_candidates: Sequence[str],
    text_candidates: Sequence[str],
    no_graph: bool = False,
) -> None:
    """Run the benchmark loop for any model exposing predict(image) -> str.

    Writes a CSV at results_dir/<slug>_<split>.csv and (unless no_graph) a
    2-panel PNG for per-sample + cumulative accuracy at the same path with
    a .png extension.
    """
    print(f"Loading dataset: {dataset_name} [{split}]")
    loaded = load_dataset(dataset_name, split=split)
    limit = min(limit, len(loaded))
    samples = loaded.select(range(limit)) if limit < len(loaded) else loaded

    image_key, text_key = pick_feature(
        samples,
        image_candidates=image_candidates,
        text_candidates=text_candidates,
    )
    print(f"Using columns: image={image_key!r}, text={text_key!r}")

    results: list[dict[str, Any]] = []

    print(f"Running {canonical_name} on {limit} samples...")
    total_infer_time = 0.0
    # Some models (Tesseract) need the gold text to pick the right language
    # at inference time. Only call the (image, gold=...) signature if the
    # model accepts the kwarg.
    _predict_kwargs_ok = False
    try:
        import inspect
        _predict_kwargs_ok = "gold" in inspect.signature(model.predict).parameters
    except (TypeError, ValueError):
        _predict_kwargs_ok = False

    for idx, sample in enumerate(samples, start=1):
        image = as_image_array(sample[image_key])
        gold = text_key_from(sample[text_key], text_candidates)
        t0 = time.perf_counter()
        if _predict_kwargs_ok:
            pred = model.predict(image, gold=gold)
        else:
            pred = model.predict(image)
        dt = time.perf_counter() - t0
        total_infer_time += dt

        results.append(
            {
                "id": idx,
                "gold": gold,
                "pred": pred,
                "cer": char_error_rate(gold, pred),
                "wer": word_error_rate(gold, pred),
                "em": int(exact_match(gold, pred)),
                "infer_ms": round(dt * 1000, 3),
            }
        )
        if idx % 10 == 0 or idx == limit:
            print(f"  {idx}/{limit} done, avg time {total_infer_time/idx:.2f}s")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{slugify(canonical_name)}_{split}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(results)
    mean_cer = sum(r["cer"] for r in results) / n
    mean_wer = sum(r["wer"] for r in results) / n
    em_acc = sum(r["em"] for r in results) / n
    avg_ms = sum(r["infer_ms"] for r in results) / n
    print("\n===== RESULTS =====")
    print(f"Model:        {canonical_name}")
    print(f"Samples:      {n}")
    print(f"Mean CER:     {mean_cer:.4f} ({mean_cer*100:.2f}%)")
    print(f"Mean WER:     {mean_wer:.4f} ({mean_wer*100:.2f}%)")
    print(f"Exact match:  {em_acc:.4f} ({em_acc*100:.2f}%)")
    print(f"Avg infer:    {avg_ms:.2f} ms/image")
    print(f"Saved to:     {out_file}")

    if not no_graph:
        png = save_graph(results, canonical_name, split, results_dir)
        if png is not None:
            print(f"Graph saved:  {png}")