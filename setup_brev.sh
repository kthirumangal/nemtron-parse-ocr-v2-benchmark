#!/usr/bin/env bash
# setup_brev.sh — Tested, working setup for nemotron-ocr-v2 on Brev
# Ubuntu 22.04, CUDA 12.9, Python 3.12
#
# Usage:
#   chmod +x setup_brev.sh && ./setup_brev.sh

set -euo pipefail

# Always work relative to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "  Working directory: $SCRIPT_DIR"

echo "══════════════════════════════════════════════"
echo " Nemotron OCR v2 — Environment Setup"
echo "══════════════════════════════════════════════"

# ── Check GPU ─────────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &> /dev/null; then
  echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Step 1: System packages + Python 3.12 ────────────────────────────────────
echo ""
echo "[1/8] Installing system packages and Python 3.12 ..."
sudo apt-get update -qq 2>&1 | grep -v "^W:" || true
sudo apt-get install -y --no-install-recommends \
  software-properties-common curl git \
  libgl1 libglib2.0-0 fonts-dejavu-core

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq 2>&1 | grep -v "^W:" || true
sudo apt-get install -y --no-install-recommends python3.12 python3.12-venv
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 - --break-system-packages || true
echo "  Python: $(python3.12 --version)"

# ── Step 2: Virtual environment ───────────────────────────────────────────────
echo ""
echo "[2/8] Creating virtual environment ..."
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

# ── Step 3: PyTorch pinned to CUDA 12.x ──────────────────────────────────────
echo ""
echo "[3/8] Installing PyTorch (cu124) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
python -c "import torch; print('  torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# ── Step 4: Build dependencies ────────────────────────────────────────────────
echo ""
echo "[4/8] Installing build dependencies ..."
pip install hatchling editables numpy pillow --quiet

# ── Step 5: Set CUDA paths ────────────────────────────────────────────────────
echo ""
echo "[5/8] Setting CUDA paths ..."
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "  CUDA_HOME=$CUDA_HOME"

# ── Step 6: Clone model repo (no weights) ────────────────────────────────────
echo ""
echo "[6/8] Cloning nemotron-ocr-v2 (skipping weights) ..."
if [ ! -d "nemotron-ocr-v2" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/nvidia/nemotron-ocr-v2 --depth 1
else
  echo "  nemotron-ocr-v2 already cloned, skipping."
fi

# ── Step 7: Build and install nemotron-ocr ────────────────────────────────────
echo ""
echo "[7/8] Building nemotron-ocr (compiling CUDA C++ kernels, ~5-10 mins) ..."
BUILD_CPP_EXTENSION=1 BUILD_CPP_FORCE=1 pip install -e nemotron-ocr-v2/nemotron-ocr --no-build-isolation

# ── Step 8: Reinstall correct PyTorch (nemotron overrides it) ────────────────
echo ""
echo "[8/8] Reinstalling correct PyTorch (nemotron-ocr overrides it during install) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --quiet
python -c "import torch; print('  torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "Verifying installation ..."
python -c "from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2; print('  nemotron-ocr-v2: OK')"

echo ""
echo "══════════════════════════════════════════════"
echo " Setup complete! Run the benchmark:"
echo ""
echo "   source .venv/bin/activate"
echo "   export CUDA_HOME=/usr/local/cuda-12.9"
echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "   ./run_benchmark.sh"
echo "══════════════════════════════════════════════"
