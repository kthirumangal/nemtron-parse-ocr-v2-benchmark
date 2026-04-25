#!/usr/bin/env python3
"""
test_accuracy.py
────────────────
Accuracy evaluation for nemotron-ocr-v2 on a subset of PDF pages.
Extracts ground truth from the original PDF using pdftotext, runs
nemotron-ocr on the converted images, and computes CER and WER per page.

Usage:
    # Full run (first 50 pages)
    python test_accuracy.py --pdf ~/CUDA_C_Programming_Guide.pdf --image-dir ~/pdf_pages --num-pages 50

    # Test only complex pages (tables/images) by page numbers
    python test_accuracy.py --pdf ~/CUDA_C_Programming_Guide.pdf --image-dir ~/pdf_pages --pages 4,10,22,45

    # Save results
    python test_accuracy.py --pdf ~/CUDA_C_Programming_Guide.pdf --image-dir ~/pdf_pages --num-pages 50 --output accuracy_results.json

Requirements:
    sudo apt-get install -y poppler-utils  (for pdftotext)
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0
    import difflib
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    matches = sum(t.size for t in matcher.get_matching_blocks())
    edits = len(ref) + len(hyp) - 2 * matches
    return round(edits / len(ref) * 100, 2)


def compute_wer(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0
    import difflib
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(t.size for t in matcher.get_matching_blocks())
    edits = len(ref_words) + len(hyp_words) - 2 * matches
    return round(edits / len(ref_words) * 100, 2)


def grade(cer: float) -> str:
    if cer < 2:   return "✓ Excellent"
    if cer < 5:   return "✓ Good"
    if cer < 15:  return "△ Fair"
    return "✗ Poor"


# ── Ground truth extraction ───────────────────────────────────────────────────

def extract_ground_truth(pdf_path: Path, page_numbers: list[int]) -> dict[int, str]:
    """Extract text per page from PDF using pdftotext."""
    print("  Extracting ground truth from PDF ...")
    gt = {}
    for page in page_numbers:
        try:
            result = subprocess.check_output(
                ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
                text=True, timeout=30
            )
            gt[page] = result.strip()
        except subprocess.CalledProcessError:
            gt[page] = ""
        except FileNotFoundError:
            print("  ERROR: pdftotext not found. Run: sudo apt-get install -y poppler-utils")
            raise
    print(f"  Extracted ground truth for {len(gt)} pages")
    return gt


# ── Image path resolver ───────────────────────────────────────────────────────

def find_image_for_page(image_dir: Path, page_num: int) -> Path | None:
    """Find the converted image file for a given page number."""
    # pdftoppm names files like page-001.png, page-002.png etc.
    candidates = [
        image_dir / f"page-{page_num:03d}.png",
        image_dir / f"page-{page_num:04d}.png",
        image_dir / f"page{page_num:03d}.png",
        image_dir / f"{page_num:03d}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: sort all PNGs and pick by index
    all_pngs = sorted(image_dir.glob("*.png"))
    if page_num - 1 < len(all_pngs):
        return all_pngs[page_num - 1]
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Accuracy evaluation for nemotron-ocr-v2.")
    p.add_argument("--pdf", type=Path, required=True, help="Path to original PDF.")
    p.add_argument("--image-dir", type=Path, required=True, help="Directory of converted PNG images.")
    p.add_argument("--num-pages", type=int, default=50, help="Number of pages to test (default: 50).")
    p.add_argument("--pages", type=str, default=None, help="Specific page numbers to test, comma-separated. e.g. 4,10,22")
    p.add_argument("--lang", choices=["multilingual", "en"], default="multilingual")
    p.add_argument("--merge-level", choices=["word", "sentence", "paragraph"], default="paragraph")
    p.add_argument("--output", type=Path, default=None, help="Save full results to JSON.")
    p.add_argument("--skip-blank", action="store_true", help="Skip pages with no ground truth text (image-only pages).")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  NEMOTRON OCR v2 — ACCURACY EVALUATION")
    print("═" * 60)

    # ── Determine pages to test ───────────────────────────────────────────────
    if args.pages:
        page_numbers = [int(x.strip()) for x in args.pages.split(",")]
    else:
        page_numbers = list(range(1, args.num_pages + 1))

    print(f"  PDF        : {args.pdf}")
    print(f"  Image dir  : {args.image_dir}")
    print(f"  Pages      : {len(page_numbers)}")
    print(f"  Lang       : {args.lang}")
    print(f"  Merge      : {args.merge_level}")

    # ── Extract ground truth ──────────────────────────────────────────────────
    ground_truth = extract_ground_truth(args.pdf, page_numbers)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n  Loading model ...")
    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    kwargs = {} if args.lang == "multilingual" else {"lang": "en"}
    model = NemotronOCRV2(**kwargs)
    print("  Model ready\n")

    # ── Per-page evaluation ───────────────────────────────────────────────────
    print("─" * 60)
    print(f"  {'Page':<6} {'CER%':>7} {'WER%':>7} {'Regions':>8} {'ms':>7}  Grade")
    print("─" * 60)

    page_results = []
    skipped = 0

    for page_num in page_numbers:
        gt_text = ground_truth.get(page_num, "")

        # Skip blank/image-only pages if requested
        if args.skip_blank and not gt_text.strip():
            skipped += 1
            continue

        image_path = find_image_for_page(args.image_dir, page_num)
        if image_path is None:
            print(f"  {page_num:<6} {'—':>7} {'—':>7} {'—':>8} {'—':>7}  Image not found")
            skipped += 1
            continue

        # Run OCR
        try:
            t0 = time.perf_counter()
            results = list(model(str(image_path), merge_level=args.merge_level))
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"  {page_num:<6} ERROR: {e}")
            skipped += 1
            continue

        ocr_text = "\n".join(r.text for r in results)
        cer = compute_cer(gt_text, ocr_text)
        wer = compute_wer(gt_text, ocr_text)
        avg_conf = sum(r.confidence for r in results) / len(results) if results else 0

        print(f"  {page_num:<6} {cer:>7.1f} {wer:>7.1f} {len(results):>8} {elapsed_ms:>7.0f}  {grade(cer)}")

        page_results.append({
            "page": page_num,
            "image": str(image_path),
            "cer_percent": cer,
            "wer_percent": wer,
            "num_regions": len(results),
            "avg_confidence": round(avg_conf, 4),
            "latency_ms": round(elapsed_ms, 2),
            "ocr_text": ocr_text,
            "ground_truth": gt_text,
            "is_image_only": not gt_text.strip(),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if page_results:
        cer_values = [r["cer_percent"] for r in page_results]
        wer_values = [r["wer_percent"] for r in page_results]
        lat_values = [r["latency_ms"] for r in page_results]
        image_only = sum(1 for r in page_results if r["is_image_only"])

        mean_cer = round(sum(cer_values) / len(cer_values), 2)
        mean_wer = round(sum(wer_values) / len(wer_values), 2)
        mean_lat = round(sum(lat_values) / len(lat_values), 2)

        # Worst pages
        worst = sorted(page_results, key=lambda x: x["cer_percent"], reverse=True)[:5]

        print("\n" + "═" * 60)
        print("  SUMMARY")
        print("═" * 60)
        print(f"  Pages tested   : {len(page_results)}  (skipped: {skipped})")
        print(f"  Image-only pages (no GT): {image_only}")
        print(f"  Mean CER       : {mean_cer:.2f}%  {grade(mean_cer)}")
        print(f"  Mean WER       : {mean_wer:.2f}%")
        print(f"  Mean latency   : {mean_lat:.0f} ms/page")
        print(f"  Throughput     : {1000/mean_lat:.2f} pages/sec")
        print(f"\n  Worst 5 pages by CER:")
        for r in worst:
            print(f"    Page {r['page']:>4}  CER={r['cer_percent']:>6.1f}%  WER={r['wer_percent']:>6.1f}%  {'(image-only)' if r['is_image_only'] else ''}")
        print("═" * 60)

        # Save JSON
        if args.output:
            out = {
                "summary": {
                    "pages_tested": len(page_results),
                    "skipped": skipped,
                    "image_only_pages": image_only,
                    "mean_cer_percent": mean_cer,
                    "mean_wer_percent": mean_wer,
                    "mean_latency_ms": mean_lat,
                    "throughput_pages_per_sec": round(1000 / mean_lat, 2),
                },
                "pages": page_results,
            }
            args.output.write_text(json.dumps(out, indent=2))
            print(f"\n  Results saved → {args.output}\n")
    else:
        print("\n  No pages were successfully evaluated.")


if __name__ == "__main__":
    main()
