#!/usr/bin/env bash
# Setup script for the USD Viewer RTX streaming tutorial.
# Run once on any machine with an NVIDIA GPU before ./run.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $*"; }
die() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

echo ""
echo "==========================================================="
echo "  USD Viewer — RTX Streaming — Local Setup"
echo "==========================================================="
echo ""

# ---- 1. GPU / driver check -----------------------------------------
echo "[1/6] Checking NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
  die "nvidia-smi not found. Install the NVIDIA driver first."
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
ok "GPU detected"

# ---- 2. Python version check ----------------------------------------
echo ""
echo "[2/6] Checking Python (3.10–3.13 required)..."
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYVER" in
  3.10|3.11|3.12|3.13) ok "Python $PYVER" ;;
  *) die "Python $PYVER is not supported. Use 3.10–3.13." ;;
esac

# ---- 3. Python virtual environment ---------------------------------
echo ""
echo "[3/6] Creating Python virtual environment (.venv)..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ok "Virtual environment created"
else
  ok "Virtual environment already exists"
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel --quiet

# ---- 4. Python dependencies ----------------------------------------
echo ""
echo "[4/6] Installing Python dependencies..."

# ovrtx lives on the NVIDIA Python package index (not PyPI).
echo "  Installing ovrtx from NVIDIA package index..."
pip install --upgrade ovrtx \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple \
  --quiet

# ovstream, warp-lang, numpy are on standard PyPI.
echo "  Installing ovstream, warp-lang, numpy..."
pip install --upgrade ovstream warp-lang numpy --quiet

# ---- 5. Validate ovrtx + GPU ----------------------------------------
echo ""
echo "[5/6] Validating ovrtx renderer on GPU..."
export OVRTX_SKIP_USD_CHECK=1

# Derive OVRTX_BIN_PATH from the installed package.
export OVRTX_BIN_PATH="$(python3 -c 'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))' 2>/dev/null || true)"
if [ -n "$OVRTX_BIN_PATH" ] && [ -d "$OVRTX_BIN_PATH" ]; then
  export LD_LIBRARY_PATH="${OVRTX_BIN_PATH}/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ok "OVRTX_BIN_PATH=$OVRTX_BIN_PATH"
else
  warn "OVRTX_BIN_PATH not found — renderer plugins may fail to load"
fi

python3 - <<'PYCHECK'
import os
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
print("  Importing ovrtx...")
import ovrtx
print(f"  ovrtx version: {getattr(ovrtx, '__version__', 'unknown')}")
print("  Constructing Renderer (first run compiles shaders, may take minutes)...")
r = ovrtx.Renderer()
print("  ovrtx.Renderer() — OK")
del r
print("  Validating ovstream...")
import ovstream
ovstream.initialize()
ovstream.shutdown()
print("  ovstream — OK")
PYCHECK

ok "GPU + ovrtx + ovstream validated"

# ---- 6. Frontend dependencies --------------------------------------
echo ""
echo "[6/6] Installing frontend dependencies (Node.js required)..."

if ! command -v node &>/dev/null; then
  die "node not found. Install Node.js 20+ from https://nodejs.org"
fi

NODE_MAJOR=$(node -e 'process.stdout.write(process.version.slice(1).split(".")[0])')
if [ "$NODE_MAJOR" -lt 18 ]; then
  die "Node.js $NODE_MAJOR is too old. Use Node.js 18+."
fi
ok "Node.js v$(node --version | tr -d 'v')"

cd frontend
npm install --prefer-offline 2>&1 | tail -5
cd ..
ok "Frontend dependencies installed"

echo ""
echo "==========================================================="
echo "  Setup complete!"
echo ""
echo "  Start server:  ./run.sh"
echo "  Open browser:  http://localhost:5173"
echo "===========================================================" 
echo ""
