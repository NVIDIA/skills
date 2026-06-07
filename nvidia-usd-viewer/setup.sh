#!/usr/bin/env bash
set -euo pipefail

echo "===  USD Viewer — Setup  ==="
echo ""

# ---- Python virtual environment ----------------------------------------
echo "[1/4] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install ovrtx ovstream warp-lang numpy

# ---- Verify GPU / ovrtx ------------------------------------------------
echo "[3/4] Verifying ovrtx renderer (GPU required)..."
python3 - <<'PYCHECK'
import os
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
import ovrtx
print(f"  ovrtx version: {getattr(ovrtx, '__version__', 'unknown')}")
r = ovrtx.Renderer()
print("  ovrtx.Renderer() — OK")
del r
PYCHECK

# ---- Frontend ----------------------------------------------------------
echo "[4/4] Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "=== Setup complete ==="
echo "Run: ./run.sh"
echo "Browser UI: http://localhost:5173"
