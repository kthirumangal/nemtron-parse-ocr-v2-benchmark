#!/usr/bin/env python3
"""
test_single_image.py
────────────────────
Run nemotron-ocr-v2 on a single image and print extracted text,
confidence scores, and bounding boxes.

Usage:
    python test_single_image.py --image /path/to/image.png
    python test_single_image.py --image /path/to/image.png --ground-truth /path/to/truth.txt
    python test_single_image.py --image /path/to/image.png --merge-level word
    python test_single_image.py --image /path/to/image.png --output results.json
"""

import argparse
import json
import time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Test nemotron-ocr-v2 on a single image.")
    p.add_argument("--image", type=Path, required=True, help="Path to image file.")
    p.add_argument("--ground-truth", type=Path, default=None, help="Optional .txt file with correct text for accuracy comparison.")
    p.add_argument("--merge-level", choices=["word", "sentence", "paragraph"], default="paragraph")
    p.add_argument("--lang", choices=["multilingual", "en"], default="multilingual")
    p.add_argument("--output", type=Path, default=None, help="Save results to JSON.")
    p.add_argument("--min-confidence", type=float, default=0.0, help="Only show regions above this confidence (0.0-1.0).")
    return p.parse_args()


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate via edit distance."""
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0
    import difflib
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    matches = sum(t.size for t in matcher.get_matching_blocks())
    edits = len(ref) + len(hyp) - 2 * matches
    return round(edits / len(ref) * 100, 2)


def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate via edit distance on word lists."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0
    import difflib
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(t.size for t in matcher.get_matching_blocks())
    edits = len(ref_words) + len(hyp_words) - 2 * matches
    return round(edits / len(ref_words) * 100, 2)


def main():
    args = parse_args()

    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}")
        return

    print("\n" + "═" * 60)
    print("  NEMOTRON OCR v2 — SINGLE IMAGE TEST")
    print("═" * 60)
    print(f"  Image      : {args.image}")
    print(f"  Merge level: {args.merge_level}")
    print(f"  Lang       : {args.lang}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n  Loading model ...")
    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    kwargs = {} if args.lang == "multilingual" else {"lang": "en"}
    model = NemotronOCRV2(**kwargs)

    # ── Run inference ─────────────────────────────────────────────────────────
    print("  Running OCR ...")
    t0 = time.perf_counter()
    results = list(model(str(args.image), merge_level=args.merge_level))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  Done in {elapsed_ms:.1f} ms\n")

    # ── Filter by confidence ──────────────────────────────────────────────────
    # Support both dict and object results
    def get(r, key):
        return r[key] if isinstance(r, dict) else getattr(r, key)

    if args.min_confidence > 0:
        results = [r for r in results if get(r, "confidence") >= args.min_confidence]

    # ── Print results ─────────────────────────────────────────────────────────
    print("─" * 60)
    print(f"  EXTRACTED TEXT  ({len(results)} regions)")
    print("─" * 60)

    all_text = []
    output_regions = []

    for i, r in enumerate(results):
        conf_bar = "█" * int(r.confidence * 10) + "░" * (10 - int(r.confidence * 10))
        print(f"\n  Region {i+1:02d}  [{conf_bar}] {r.confidence:.3f}")
        print(f"  Text: {r.text}")
        print(f"  BBox: {r.bounding_box}")
        all_text.append(r.text)
        output_regions.append({
            "region": i + 1,
            "text": r.text,
            "confidence": round(r.confidence, 4),
            "bounding_box": r.bounding_box,
        })

    full_text = "\n".join(all_text)

    print("\n" + "─" * 60)
    print("  FULL EXTRACTED TEXT")
    print("─" * 60)
    print(full_text)

    # ── Accuracy comparison ───────────────────────────────────────────────────
    accuracy = None
    if args.ground_truth and args.ground_truth.exists():
        gt_text = args.ground_truth.read_text(encoding="utf-8")
        cer = compute_cer(gt_text, full_text)
        wer = compute_wer(gt_text, full_text)
        accuracy = {"cer_percent": cer, "wer_percent": wer}

        print("\n" + "─" * 60)
        print("  ACCURACY vs GROUND TRUTH")
        print("─" * 60)
        print(f"  CER (Character Error Rate): {cer:.2f}%  {'✓ Excellent' if cer < 5 else '△ Good' if cer < 15 else '✗ Poor'}")
        print(f"  WER (Word Error Rate)      : {wer:.2f}%  {'✓ Excellent' if wer < 5 else '△ Good' if wer < 15 else '✗ Poor'}")

    print("\n" + "─" * 60)
    avg_conf = sum(get(r, "confidence") for r in results) / len(results) if results else 0
    print(f"  Regions found  : {len(results)}")
    print(f"  Avg confidence : {avg_conf:.3f}")
    print(f"  Latency        : {elapsed_ms:.1f} ms  ({elapsed_ms/1000:.4f} sec)")
    print("─" * 60 + "\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        out = {
            "image": str(args.image),
            "merge_level": args.merge_level,
            "lang": args.lang,
            "latency_ms": round(elapsed_ms, 2),
            "latency_sec": round(elapsed_ms / 1000, 4),
            "num_regions": len(results),
            "avg_confidence": round(avg_conf, 4),
            "full_text": full_text,
            "regions": output_regions,
            "accuracy": accuracy,
        }
        args.output.write_text(json.dumps(out, indent=2))
        print(f"  Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
