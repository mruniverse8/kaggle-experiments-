#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-dz2-qwen}"
MODE="${2:-gpu}" # gpu|cpu

cd "${ROOT_DIR}"

echo "[1/4] Creating conda environment: ${ENV_NAME}"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda env create -n "${ENV_NAME}" -f environment.yml

echo "[2/4] Activating environment"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[3/4] Upgrading pip"
python -m pip install --upgrade pip

if [[ "${MODE}" == "gpu" ]]; then
  echo "[4/4] Installing GPU PyTorch (CUDA 12.1 wheels)"
  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "[4/4] Installing CPU-only PyTorch"
  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "Done. Environment '${ENV_NAME}' is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
