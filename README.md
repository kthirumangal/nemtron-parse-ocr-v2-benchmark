# nemotron-ocr-v2 Benchmark

Latency, throughput, and accuracy benchmark for [`nvidia/nemotron-ocr-v2`](https://huggingface.co/nvidia/nemotron-ocr-v2) on Brev GPU instances. Measures pages/sec, per-image latency (ms and sec), CER/WER accuracy, and VRAM usage across different GPU SKUs.

---

## Files

| File | Purpose |
|---|---|
| `setup_brev.sh` | One-time environment setup |
| `run_benchmark.sh` | Latency & throughput benchmark |
| `run_accuracy.sh` | Accuracy evaluation & single image test |
| `benchmark_nemotron_ocr_v2.py` | Core benchmark script |
| `test_single_image.py` | Test OCR on a specific image |
| `test_accuracy.py` | CER/WER accuracy evaluation against PDF ground truth |
| `sample_images/gpu_computing_table.png` | Bundled complex table image for testing |

---

## Quickstart

### Step 1 — Clone the repo on your Brev instance

```bash
git clone https://github.com/kthirumangal/nemtron-parse-ocr-v2-benchmark.git
cd nemtron-parse-ocr-v2-benchmark
chmod +x setup_brev.sh run_benchmark.sh run_accuracy.sh
```

### Step 2 — Run setup (once per instance, ~15 mins)

```bash
./setup_brev.sh
```

This installs Python 3.12, PyTorch, compiles the CUDA C++ extension, and creates `.venv` inside the repo folder.

### Step 3 — Run latency benchmark

```bash
# Synthetic dataset — no PDF needed
./run_benchmark.sh

# Your own PDF pages
./run_benchmark.sh --dataset-dir ~/pdf_pages --num-samples 100 --batch-size 8
```

### Step 4 — Run accuracy test

```bash
# Bundled sample image — no upload needed
./run_accuracy.sh

# Your own image
./run_accuracy.sh --image ~/your_image.png

# Your own PDF — auto-converts to images then runs accuracy
./run_accuracy.sh --pdf ~/your_document.pdf --num-pages 50

# Specific pages only (e.g. complex tables/diagrams)
./run_accuracy.sh --pdf ~/your_document.pdf --pages 4,10,22,45

# If you already converted the PDF to images
./run_accuracy.sh --pdf ~/your_document.pdf --image-dir ~/pdf_pages --num-pages 50
```

---

## Using Your Own PDF

Upload the PDF from your local machine (**only the PDF needs uploading — scripts come from the repo**):

```bash
# From your LOCAL machine
scp your_document.pdf ubuntu@<instance-ip>:~/
```

Then run — PDF to image conversion happens automatically:

```bash
./run_accuracy.sh --pdf ~/your_document.pdf --num-pages 50
```

To manually convert first (e.g. for benchmarking):

```bash
sudo apt-get install -y poppler-utils
mkdir -p ~/pdf_pages
pdftoppm -r 150 -png ~/your_document.pdf ~/pdf_pages/page

# Check page count
ls ~/pdf_pages/ | wc -l

# Run benchmark on real pages
./run_benchmark.sh --dataset-dir ~/pdf_pages --num-samples 100 --batch-size 8
```

---

## Accuracy Results Explained

```
  Page   CER%    WER%  Regions      ms  Grade
  ────────────────────────────────────────────
  1       1.2     2.1       12      98  ✓ Excellent
  2       4.8     6.3        8     103  ✓ Good
  4      18.3    24.1        5     112  ✗ Poor       ← complex table
  10      2.1     3.4       15      95  ✓ Excellent
```

| Metric | Description |
|---|---|
| **CER** | Character Error Rate — % of characters wrong. Best for precise accuracy |
| **WER** | Word Error Rate — % of words wrong. Good for readability assessment |
| **Regions** | Number of text regions detected on the page |
| **Grade** | Excellent < 2% CER · Good < 5% · Fair < 15% · Poor ≥ 15% |

**Why some pages score poorly:**
- Complex tables with merged cells and colored backgrounds
- Pages with embedded images overlapping text
- Code blocks with special characters
- Image-only pages (diagrams with no selectable text in the PDF)

---

## Benchmark Options

| Flag | Default | Description |
|---|---|---|
| `--dataset-dir` | synthetic | Path to folder of images or PDF pages |
| `--num-samples` | 50 | Number of images to benchmark |
| `--batch-size` | 4 | Batch size for batch-mode benchmark |
| `--lang` | `multilingual` | `multilingual` or `en` (English-only, faster) |
| `--merge-level` | `paragraph` | `word`, `sentence`, or `paragraph` |
| `--warmup-runs` | 3 | GPU warmup inferences before timing starts |
| `--skip-batch` | off | Skip batch benchmark |

---

## Benchmark Results (NVIDIA A10G)

| Metric | Value |
|---|---|
| Mean latency | 101.6 ms / page |
| Throughput | 9.85 pages/sec |
| P90 latency | 110.7 ms |
| VRAM used | 2.66 GB |

> At 9.85 pages/sec each page takes ~100ms. To hit 50ms/page you need ~20 pages/sec.

---

## Comparing Across GPUs

Results auto-save as `results_<GPU_NAME>.json` in the repo folder. Pull them back locally:

```bash
scp ubuntu@<instance-ip>:~/nemtron-parse-ocr-v2-benchmark/results_*.json ./
```

---

## Model Variants

| Variant | Flag | Languages | Size |
|---|---|---|---|
| Multilingual (default) | `--lang multilingual` | EN, ZH, JA, KO, AR, HI | 83.9M params |
| English-only | `--lang en` | EN only | 53.8M params |

---

## Setup Notes

- Python 3.12 is required (installed via deadsnakes PPA)
- PyTorch cu124 is pinned — nemotron-ocr overrides it during install, setup reinstalls it at the end
- Dependency warnings about `torch>=2.8.0` are harmless — ignore them
- CUDA minor version mismatch warning (12.9 vs 12.4) is harmless — ignore it
- nemotron-ocr-v2 cannot run on vLLM — it's a vision OCR model with custom CUDA kernels
- OCR runs on images only — `run_accuracy.sh` handles PDF→image conversion automatically
