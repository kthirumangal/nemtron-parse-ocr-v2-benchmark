#!/usr/bin/env python3
"""
benchmark_nemotron_ocr_v2.py
────────────────────────────
Latency & throughput benchmark for nvidia/nemotron-ocr-v2 on Brev GPU instances.

Usage
─────
  # Basic run (auto-generates synthetic dataset):
  python benchmark_nemotron_ocr_v2.py

  # Point at your own image folder:
  python benchmark_nemotron_ocr_v2.py --dataset-dir /path/to/images

  # Tune batch size and sample count:
  python benchmark_nemotron_ocr_v2.py --batch-size 8 --num-samples 100

  # Use English-only variant:
  python benchmark_nemotron_ocr_v2.py --lang en

  # Save results as JSON:
  python benchmark_nemotron_ocr_v2.py --output results.json

Model variants
──────────────
  --lang multilingual  (default)  nvidia/nemotron-ocr-v2 / v2_multilingual/
  --lang en                       nvidia/nemotron-ocr-v2 / v2_english/

Requirements
────────────
  pip install nemotron-ocr pillow numpy
  CUDA Toolkit 12.x + compatible NVIDIA driver (verify with nvidia-smi)
  Python 3.12 recommended (required by nemotron-ocr sub-packages)
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── Third-party imports (installed via pip) ──────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    sys.exit(
        "Missing dependencies. Run:\n"
        "  pip install nemotron-ocr pillow numpy"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GPU / system info helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """Query nvidia-smi for GPU name, VRAM, driver, and CUDA version."""
    info = {
        "gpu_name": "unknown",
        "vram_total_gb": None,
        "driver_version": "unknown",
        "cuda_version": "unknown",
        "gpu_count": 0,
    }
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()

        info["gpu_count"] = len(query)
        if query:
            name, mem_mb, driver = [x.strip() for x in query[0].split(",")]
            info["gpu_name"] = name
            info["vram_total_gb"] = round(int(mem_mb) / 1024, 1)
            info["driver_version"] = driver

        # CUDA version from nvidia-smi header line
        smi_out = subprocess.check_output(
            ["nvidia-smi"], text=True, timeout=10
        )
        for line in smi_out.splitlines():
            if "CUDA Version" in line:
                info["cuda_version"] = line.split("CUDA Version:")[-1].strip().split()[0]
                break
    except Exception:
        pass
    return info


def get_system_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **get_gpu_info(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generation
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "NVIDIA Nemotron OCR v2 benchmark evaluation.",
    "Page 1 of 10 | Confidential Report Q4-2025",
    "Invoice #: 00482  Date: 2025-04-24  Total: $1,299.00",
    "Section 3.2: Performance Metrics and Latency Analysis",
    "Table of Contents\n1. Introduction ........ 2\n2. Methods ............. 5",
    "Abstract: This paper presents a novel approach to multimodal learning.",
    "CUDA_VISIBLE_DEVICES=0 python train.py --epochs 50 --lr 1e-4",
    "© 2025 NVIDIA Corporation. All rights reserved.",
    "Σελίδα 1 | Page 1 | ページ1 | 第1页 | 페이지 1",  # multilingual
]


def generate_synthetic_dataset(
    output_dir: Path, num_images: int = 50, seed: int = 42
) -> list[Path]:
    """
    Create simple synthetic document images for benchmarking.
    Returns list of image paths.
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Image size variations to stress different resolutions
    sizes = [(800, 600), (1280, 960), (1920, 1080), (640, 480)]

    for i in range(num_images):
        size = sizes[i % len(sizes)]
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]

        # Create white document background
        img = Image.new("RGB", size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try to use a system font; fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
            small_font = font

        # Header block
        draw.rectangle([0, 0, size[0], 60], fill=(30, 30, 120))
        draw.text((20, 15), f"BENCHMARK SAMPLE #{i+1:03d}", fill=(255, 255, 255), font=font)

        # Body text
        y_offset = 80
        for line in text.split("\n"):
            draw.text((20, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += 35

        # Add some noise rows to simulate real document variability
        for j in range(3):
            y = 200 + j * 60
            draw.text((20, y), f"Field {j+1}: Value_{np.random.randint(1000, 9999)}", fill=(50, 50, 50), font=small_font)
            draw.line([20, y + 28, size[0] - 20, y + 28], fill=(200, 200, 200), width=1)

        # Footer
        draw.rectangle([0, size[1] - 40, size[0], size[1]], fill=(240, 240, 240))
        draw.text((20, size[1] - 30), f"Generated for nemotron-ocr-v2 benchmark | img_{i+1:03d}.png", fill=(100, 100, 100), font=small_font)

        path = output_dir / f"img_{i+1:03d}.png"
        img.save(path, "PNG")
        paths.append(path)

    print(f"  Generated {len(paths)} synthetic images → {output_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(lang: str = "multilingual", model_dir: Optional[str] = None):
    """
    Load NemotronOCRV2.  Falls back to v1 NemotronOCR if v2 is unavailable
    (older package version).
    """
    try:
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

        kwargs: dict = {}
        if model_dir:
            kwargs["model_dir"] = model_dir
        elif lang == "en":
            kwargs["lang"] = "en"
        # multilingual is the default; no extra kwarg needed

        print(f"  Loading NemotronOCRV2 (lang={lang!r}) …")
        t0 = time.perf_counter()
        model = NemotronOCRV2(**kwargs)
        load_secs = time.perf_counter() - t0
        print(f"  Model loaded in {load_secs:.2f}s")
        return model, "v2"

    except ImportError:
        print("  NemotronOCRV2 not found — falling back to NemotronOCR (v1)")
        from nemotron_ocr.inference.pipeline import NemotronOCR

        kwargs = {}
        if model_dir:
            kwargs["model_dir"] = model_dir

        t0 = time.perf_counter()
        model = NemotronOCR(**kwargs)
        load_secs = time.perf_counter() - t0
        print(f"  Model loaded in {load_secs:.2f}s")
        return model, "v1"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def warmup(model, image_paths: list[Path], warmup_runs: int = 3) -> None:
    """Run a few inferences to warm up GPU kernels."""
    print(f"  Warming up ({warmup_runs} runs) …")
    for i in range(warmup_runs):
        path = image_paths[i % len(image_paths)]
        try:
            list(model(str(path)))
        except TypeError:
            list(model(str(path), merge_level="paragraph"))


def run_single_image_benchmark(
    model, image_paths: list[Path], num_samples: int, merge_level: str
) -> dict:
    """
    Measure per-image latency in single-image (non-batched) mode.
    Returns statistics dict.
    """
    print(f"\n  Running single-image benchmark ({num_samples} samples) …")
    latencies_ms: list[float] = []
    errors = 0

    for i, path in enumerate(image_paths[:num_samples]):
        try:
            t0 = time.perf_counter()
            preds = list(model(str(path), merge_level=merge_level))
            elapsed = (time.perf_counter() - t0) * 1000  # → ms
            latencies_ms.append(elapsed)
            num_predictions = len(preds)
        except Exception as e:
            errors += 1
            print(f"    Warning: inference error on {path.name}: {e}")

        if (i + 1) % 10 == 0:
            print(f"    … {i+1}/{num_samples} images processed")

    if not latencies_ms:
        return {"error": "All inference calls failed"}

    sorted_ms = sorted(latencies_ms)
    mean_ms = statistics.mean(latencies_ms)

    def ms_to_s(v):
        return round(v / 1000, 4)

    return {
        "mode": "single_image",
        "num_samples": len(latencies_ms),
        "errors": errors,
        "latency_ms": {
            "mean": round(mean_ms, 2),
            "median": round(statistics.median(latencies_ms), 2),
            "p90": round(sorted_ms[int(len(sorted_ms) * 0.90)], 2),
            "p95": round(sorted_ms[int(len(sorted_ms) * 0.95)], 2),
            "p99": round(sorted_ms[int(len(sorted_ms) * 0.99)], 2) if len(sorted_ms) >= 100 else None,
            "min": round(min(latencies_ms), 2),
            "max": round(max(latencies_ms), 2),
            "stdev": round(statistics.stdev(latencies_ms), 2) if len(latencies_ms) > 1 else 0.0,
        },
        "latency_sec": {
            "mean": ms_to_s(mean_ms),
            "median": ms_to_s(statistics.median(latencies_ms)),
            "p90": ms_to_s(sorted_ms[int(len(sorted_ms) * 0.90)]),
            "p95": ms_to_s(sorted_ms[int(len(sorted_ms) * 0.95)]),
            "p99": ms_to_s(sorted_ms[int(len(sorted_ms) * 0.99)]) if len(sorted_ms) >= 100 else None,
            "min": ms_to_s(min(latencies_ms)),
            "max": ms_to_s(max(latencies_ms)),
        },
        "throughput_images_per_sec": round(1000 / mean_ms, 2),
        "throughput_pages_per_sec": round(1000 / mean_ms, 2),  # 1 image = 1 page
    }


def run_batch_benchmark(
    model, image_paths: list[Path], batch_size: int, num_samples: int, merge_level: str
) -> dict:
    """
    Measure batch latency.  nemotron-ocr-v2's batched pipeline is exposed
    via model.run_batch() when available, otherwise we manually batch.
    """
    print(f"\n  Running batch benchmark (batch_size={batch_size}) …")

    batches = [
        image_paths[i : i + batch_size]
        for i in range(0, min(num_samples, len(image_paths)), batch_size)
    ]

    batch_latencies_ms: list[float] = []
    total_images = 0
    errors = 0

    for batch in batches:
        batch_strs = [str(p) for p in batch]
        try:
            t0 = time.perf_counter()

            # Prefer native batch API if available
            if hasattr(model, "run_batch"):
                model.run_batch(batch_strs, merge_level=merge_level)
            else:
                # Manual batching: call sequentially within the timed block
                for img_path in batch_strs:
                    list(model(img_path, merge_level=merge_level))

            elapsed = (time.perf_counter() - t0) * 1000
            batch_latencies_ms.append(elapsed)
            total_images += len(batch)
        except Exception as e:
            errors += 1
            print(f"    Warning: batch error: {e}")

    if not batch_latencies_ms:
        return {"error": "All batch calls failed"}

    total_elapsed_s = sum(batch_latencies_ms) / 1000
    throughput = total_images / total_elapsed_s if total_elapsed_s > 0 else 0

    mean_batch_ms = statistics.mean(batch_latencies_ms)
    latency_per_image_ms = mean_batch_ms / batch_size

    return {
        "mode": "batch",
        "batch_size": batch_size,
        "num_batches": len(batch_latencies_ms),
        "total_images": total_images,
        "errors": errors,
        "batch_latency_ms": {
            "mean": round(mean_batch_ms, 2),
            "median": round(statistics.median(batch_latencies_ms), 2),
            "min": round(min(batch_latencies_ms), 2),
            "max": round(max(batch_latencies_ms), 2),
        },
        "batch_latency_sec": {
            "mean": round(mean_batch_ms / 1000, 4),
            "median": round(statistics.median(batch_latencies_ms) / 1000, 4),
            "min": round(min(batch_latencies_ms) / 1000, 4),
            "max": round(max(batch_latencies_ms) / 1000, 4),
        },
        "latency_per_image_ms": round(latency_per_image_ms, 2),
        "latency_per_image_sec": round(latency_per_image_ms / 1000, 4),
        "throughput_images_per_sec": round(throughput, 2),
        "throughput_pages_per_sec": round(throughput, 2),  # 1 image = 1 page
    }


def get_vram_usage_mb() -> Optional[float]:
    """Return current GPU memory usage in MB (first GPU)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip().splitlines()
        return float(out[0].strip())
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    SEP = "─" * 60

    print(f"\n{'═'*60}")
    print("  NEMOTRON OCR v2 — BENCHMARK RESULTS")
    print(f"{'═'*60}")

    sys_info = results.get("system_info", {})
    print(f"\n  GPU:     {sys_info.get('gpu_name', 'unknown')}")
    print(f"  VRAM:    {sys_info.get('vram_total_gb', '?')} GB")
    print(f"  Driver:  {sys_info.get('driver_version', '?')}")
    print(f"  CUDA:    {sys_info.get('cuda_version', '?')}")
    print(f"  Python:  {sys_info.get('python', '?')}")
    print(f"  Model:   {results.get('model_variant', '?')}  lang={results.get('lang', '?')}")
    print(f"  Dataset: {results.get('dataset_source', '?')}")
    print(f"  Merge:   {results.get('merge_level', 'paragraph')}")

    single = results.get("single_image")
    if single and "error" not in single:
        print(f"\n{SEP}")
        print("  SINGLE-IMAGE LATENCY")
        print(SEP)
        lms = single["latency_ms"]
        lss = single["latency_sec"]
        print(f"  Samples  : {single['num_samples']}")
        print(f"  {'Metric':<10}  {'ms':>10}  {'sec':>10}")
        print(f"  {'-'*34}")
        print(f"  {'Mean':<10}  {lms['mean']:>10.1f}  {lss['mean']:>10.4f}")
        print(f"  {'Median':<10}  {lms['median']:>10.1f}  {lss['median']:>10.4f}")
        print(f"  {'P90':<10}  {lms['p90']:>10.1f}  {lss['p90']:>10.4f}")
        print(f"  {'P95':<10}  {lms['p95']:>10.1f}  {lss['p95']:>10.4f}")
        if lms.get("p99") is not None:
            print(f"  {'P99':<10}  {lms['p99']:>10.1f}  {lss['p99']:>10.4f}")
        print(f"  {'Min':<10}  {lms['min']:>10.1f}  {lss['min']:>10.4f}")
        print(f"  {'Max':<10}  {lms['max']:>10.1f}  {lss['max']:>10.4f}")
        print(f"  {'StdDev':<10}  {lms['stdev']:>10.1f}")
        print(f"  {'-'*34}")
        print(f"  Throughput : {single['throughput_pages_per_sec']:.2f} pages/sec  ({single['throughput_images_per_sec']:.2f} img/s)")
        if single.get("errors"):
            print(f"  Errors   : {single['errors']}")

    batch = results.get("batch")
    if batch and "error" not in batch:
        print(f"\n{SEP}")
        print(f"  BATCH LATENCY  (batch_size={batch['batch_size']})")
        print(SEP)
        blms = batch["batch_latency_ms"]
        blss = batch["batch_latency_sec"]
        print(f"  Batches  : {batch['num_batches']}")
        print(f"  {'Metric':<22}  {'ms':>10}  {'sec':>10}")
        print(f"  {'-'*46}")
        print(f"  {'Mean batch latency':<22}  {blms['mean']:>10.1f}  {blss['mean']:>10.4f}")
        print(f"  {'Median batch latency':<22}  {blms['median']:>10.1f}  {blss['median']:>10.4f}")
        print(f"  {'Min batch latency':<22}  {blms['min']:>10.1f}  {blss['min']:>10.4f}")
        print(f"  {'Max batch latency':<22}  {blms['max']:>10.1f}  {blss['max']:>10.4f}")
        print(f"  {'Latency / page':<22}  {batch['latency_per_image_ms']:>10.1f}  {batch['latency_per_image_sec']:>10.4f}")
        print(f"  {'-'*46}")
        print(f"  Throughput : {batch['throughput_pages_per_sec']:.2f} pages/sec  ({batch['throughput_images_per_sec']:.2f} img/s)")
        if batch.get("errors"):
            print(f"  Errors   : {batch['errors']}")

    vram = results.get("vram_used_mb")
    if vram is not None:
        print(f"\n  VRAM used during inference: {vram:.0f} MB  ({vram/1024:.2f} GB)")

    print(f"\n{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latency benchmark for nvidia/nemotron-ocr-v2 on Brev GPU instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Directory of images to benchmark. If omitted, synthetic images are generated.",
    )
    p.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of images to benchmark (default: 50).",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for the batch-mode benchmark (default: 4).",
    )
    p.add_argument(
        "--lang", choices=["multilingual", "en"], default="multilingual",
        help="Model variant: 'multilingual' (default) or 'en'.",
    )
    p.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to a local model checkpoint directory (skips HuggingFace download).",
    )
    p.add_argument(
        "--merge-level", choices=["word", "sentence", "paragraph"], default="paragraph",
        help="OCR merge granularity (default: paragraph).",
    )
    p.add_argument(
        "--warmup-runs", type=int, default=3,
        help="Number of warm-up inferences before timing (default: 3).",
    )
    p.add_argument(
        "--skip-batch", action="store_true",
        help="Skip the batch-mode benchmark.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Save full results as JSON to this path.",
    )
    p.add_argument(
        "--synthetic-seed", type=int, default=42,
        help="Random seed for synthetic image generation (default: 42).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "═" * 60)
    print("  NEMOTRON OCR v2 — GPU LATENCY BENCHMARK")
    print("  github: nvidia/nemotron-ocr-v2  |  Brev-ready")
    print("═" * 60)

    # ── 1. System info ───────────────────────────────────────────────────────
    print("\n[1/5] Collecting system info …")
    sys_info = get_system_info()
    print(f"  GPU  : {sys_info['gpu_name']}  ({sys_info['vram_total_gb']} GB VRAM)")
    print(f"  CUDA : {sys_info['cuda_version']}   Driver: {sys_info['driver_version']}")

    # ── 2. Dataset ───────────────────────────────────────────────────────────
    print("\n[2/5] Preparing dataset …")
    if args.dataset_dir and args.dataset_dir.exists():
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        image_paths = sorted(
            p for p in args.dataset_dir.iterdir()
            if p.suffix.lower() in exts
        )
        if not image_paths:
            sys.exit(f"  ERROR: No image files found in {args.dataset_dir}")
        dataset_source = f"user-supplied ({args.dataset_dir})"
        print(f"  Found {len(image_paths)} images in {args.dataset_dir}")
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="nemocr_bench_"))
        image_paths = generate_synthetic_dataset(
            tmp_dir, num_images=max(args.num_samples, 20), seed=args.synthetic_seed
        )
        dataset_source = f"synthetic ({len(image_paths)} generated images)"

    # Clamp num_samples to available images
    num_samples = min(args.num_samples, len(image_paths))
    if num_samples < args.num_samples:
        print(f"  Note: --num-samples reduced to {num_samples} (dataset size)")

    # ── 3. Load model ────────────────────────────────────────────────────────
    print(f"\n[3/5] Loading model (lang={args.lang!r}) …")
    model, model_version = load_model(lang=args.lang, model_dir=args.model_dir)

    # ── 4. Warm up ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Warming up …")
    warmup(model, image_paths, warmup_runs=args.warmup_runs)

    # ── 5. Benchmark ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Running benchmarks …")

    single_results = run_single_image_benchmark(
        model, image_paths, num_samples=num_samples, merge_level=args.merge_level
    )

    batch_results = None
    if not args.skip_batch and args.batch_size > 1:
        batch_results = run_batch_benchmark(
            model, image_paths, batch_size=args.batch_size,
            num_samples=num_samples, merge_level=args.merge_level,
        )

    vram_used = get_vram_usage_mb()

    # ── Collate ──────────────────────────────────────────────────────────────
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system_info": sys_info,
        "model_variant": f"nemotron-ocr-{model_version}",
        "lang": args.lang,
        "merge_level": args.merge_level,
        "dataset_source": dataset_source,
        "warmup_runs": args.warmup_runs,
        "single_image": single_results,
        "batch": batch_results,
        "vram_used_mb": vram_used,
    }

    print_report(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
