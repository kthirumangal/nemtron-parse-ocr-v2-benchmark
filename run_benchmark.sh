#!/usr/bin/env bash
# run_benchmark.sh — Run the nemotron-ocr-v2 latency benchmark
#
# Usage:
#   ./run_benchmark.sh                          # synthetic dataset, defaults
#   ./run_benchmark.sh --dataset-dir /path/to/images
#   ./run_benchmark.sh --num-samples 200 --batch-size 8
#   ./run_benchmark.sh --lang en
#
# All extra args are passed directly to benchmark_nemotron_ocr_v2.py

set -euo pipefail

# Always work relative to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
  echo "ERROR: .venv not found in $SCRIPT_DIR. Run setup_brev.sh first."
  exit 1
fi
source "$SCRIPT_DIR/.venv/bin/activate"

# ── Set CUDA paths ────────────────────────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ── Check benchmark script exists ─────────────────────────────────────────────
if [ ! -f "benchmark_nemotron_ocr_v2.py" ]; then
  echo "ERROR: benchmark_nemotron_ocr_v2.py not found in current directory."
  echo "Upload it with: scp benchmark_nemotron_ocr_v2.py ubuntu@<instance-ip>:~/"
  exit 1
fi


# ── Download sample PDF if no dataset provided and no --dataset-dir given ─────
if [[ "$*" != *"--dataset-dir"* ]]; then
  SAMPLE_PDF="$SCRIPT_DIR/sample_pdfs/attention_is_all_you_need.pdf"
  SAMPLE_PAGES="$SCRIPT_DIR/sample_pdfs/pages"

  if [ ! -d "$SAMPLE_PAGES" ] || [ -z "$(ls -A $SAMPLE_PAGES 2>/dev/null)" ]; then
    echo "No --dataset-dir provided. Using sample PDF for benchmarking ..."
    mkdir -p "$SCRIPT_DIR/sample_pdfs"

    if [ ! -f "$SAMPLE_PDF" ]; then
      echo "  Downloading sample PDF (Attention Is All You Need, arxiv) ..."
      curl -sL "https://arxiv.org/pdf/1706.03762" -o "$SAMPLE_PDF"
      echo "  Downloaded: $SAMPLE_PDF"
    fi

    echo "  Converting PDF to images ..."
    sudo apt-get install -y poppler-utils -qq
    mkdir -p "$SAMPLE_PAGES"
    pdftoppm -r 150 -png "$SAMPLE_PDF" "$SAMPLE_PAGES/page"
    echo "  Pages ready: $(ls $SAMPLE_PAGES | wc -l) pages"
  fi

  set -- --dataset-dir "$SAMPLE_PAGES" "$@"
fi

# ── Auto-name output JSON by GPU ──────────────────────────────────────────────
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr -d '/')
OUTPUT="results_${GPU}.json"

echo "GPU: $GPU"
echo "Output: $OUTPUT"
echo ""

# ── Run benchmark, passing all CLI args through ───────────────────────────────
python benchmark_nemotron_ocr_v2.py --output "$OUTPUT" "$@"

echo ""
echo "Results saved to: $OUTPUT"
