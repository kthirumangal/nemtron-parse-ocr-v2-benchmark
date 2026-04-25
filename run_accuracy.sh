#!/usr/bin/env bash
# run_accuracy.sh — Run accuracy evaluation or single image test
#
# Usage:
#   ./run_accuracy.sh                                                         # bundled sample image
#   ./run_accuracy.sh --image ~/your_image.png                                # your own image
#   ./run_accuracy.sh --image ~/your_image.png --merge-level word
#   ./run_accuracy.sh --image ~/your_image.png --output result.json
#   ./run_accuracy.sh --pdf ~/doc.pdf --num-pages 50                          # auto-converts PDF
#   ./run_accuracy.sh --pdf ~/doc.pdf --num-pages 50 --pages 4,10,22
#   ./run_accuracy.sh --pdf ~/doc.pdf --image-dir ~/pdf_pages --num-pages 50  # use existing pages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
  echo "ERROR: .venv not found. Run ./setup_brev.sh first."
  exit 1
fi
source "$SCRIPT_DIR/.venv/bin/activate"

# ── Set CUDA paths ────────────────────────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ── Route based on args ───────────────────────────────────────────────────────
if [[ "$*" == *"--pdf"* ]]; then

  # If --image-dir not provided, auto-convert the PDF first
  if [[ "$*" != *"--image-dir"* ]]; then
    PDF_PATH=""
    for arg in "$@"; do
      if [[ "$PREV" == "--pdf" ]]; then PDF_PATH="$arg"; fi
      PREV="$arg"
    done

    PAGES_DIR="$SCRIPT_DIR/sample_pdfs/pages"

    echo "Converting PDF to images (this is a one-time step) ..."
    sudo apt-get install -y poppler-utils -qq
    mkdir -p "$PAGES_DIR"
    pdftoppm -r 150 -png "$PDF_PATH" "$PAGES_DIR/page"
    echo "  Done — $(ls $PAGES_DIR | wc -l) pages converted"
    echo ""

    python test_accuracy.py "$@" --image-dir "$PAGES_DIR"
  else
    python test_accuracy.py "$@"
  fi

elif [[ "$*" == *"--image"* ]]; then
  echo "Running single image test ..."
  python test_single_image.py "$@"

else
  # No args — use bundled sample image
  SAMPLE="$SCRIPT_DIR/sample_images/gpu_computing_table.png"
  if [ ! -f "$SAMPLE" ]; then
    echo "ERROR: Sample image not found at $SAMPLE"
    exit 1
  fi
  echo "No arguments provided. Running OCR on bundled sample image:"
  echo "  $SAMPLE"
  echo ""
  python test_single_image.py --image "$SAMPLE" --merge-level word
fi
