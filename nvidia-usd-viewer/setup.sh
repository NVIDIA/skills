#!/usr/bin/env bash
# Setup script for the USD Viewer RTX streaming tutorial.
# Works on Linux x86_64 and Windows (Git Bash / WSL) with Python 3.10-3.13.
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

# ---- 2. Detect Python (python3 on Linux/WSL; python on Windows/conda) ----
echo ""
echo "[2/6] Checking Python (3.10-3.13 required)..."

PY=""
for candidate in python3 python python3.13 python3.12 python3.11 python3.10; do
  if command -v "$candidate" &>/dev/null; then
    VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
    case "$VER" in
      3.10|3.11|3.12|3.13)
        PY="$candidate"
        ok "Found $candidate ($VER)"
        break
        ;;
    esac
  fi
done

if [ -z "$PY" ]; then
  die "Python 3.10-3.13 not found.\n  Windows: install from https://www.python.org/downloads/\n  Conda:   conda install python=3.11"
fi

# ---- 3. Virtual environment ----------------------------------------
echo ""
echo "[3/6] Creating virtual environment (.venv)..."
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  ok "Virtual environment created"
else
  ok "Virtual environment already exists"
fi

# Activate: Linux/macOS path vs Windows Git Bash path
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
else
  die "Could not find .venv activation script"
fi

python -m pip install --upgrade pip setuptools wheel --quiet

# ---- 4. Python dependencies ----------------------------------------
echo ""
echo "[4/6] Installing Python dependencies..."

echo "  ovrtx  (NVIDIA package index — pinned to 0.2.0 for stability)..."
pip install "ovrtx==0.2.0.280040" \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.org/simple \
  --quiet

echo "  ovstream, warp-lang, numpy  (PyPI)..."
pip install --upgrade ovstream warp-lang numpy --quiet

# ---- 5. Validate ovrtx + GPU ----------------------------------------
echo ""
echo "[5/6] Validating ovrtx renderer (first run compiles shaders, may take minutes)..."
export OVRTX_SKIP_USD_CHECK=1

# Derive OVRTX_BIN_PATH from the installed package.
OVRTX_BIN_PATH=$(python -c \
  'import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), "bin"))' \
  2>/dev/null || true)
if [ -n "$OVRTX_BIN_PATH" ] && [ -d "$OVRTX_BIN_PATH" ]; then
  # Linux/WSL: add to LD_LIBRARY_PATH; Windows: DLLs are found via PATH
  export LD_LIBRARY_PATH="${OVRTX_BIN_PATH}/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="${OVRTX_BIN_PATH}:$PATH"
  ok "OVRTX_BIN_PATH=$OVRTX_BIN_PATH"
else
  warn "OVRTX_BIN_PATH not found — renderer plugins may fail to load"
fi

python - <<'PYCHECK'
import os
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
print("  Importing ovrtx...")
import ovrtx
print(f"  ovrtx version: {getattr(ovrtx, '__version__', 'unknown')}")
print("  Constructing Renderer...")
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
echo "[6/6] Installing frontend dependencies..."

if ! command -v node &>/dev/null; then
  die "node not found. Install Node.js 20+ from https://nodejs.org"
fi
NODE_MAJOR=$(node -e 'process.stdout.write(process.version.slice(1).split(".")[0])')
if [ "$NODE_MAJOR" -lt 18 ]; then
  die "Node.js $NODE_MAJOR is too old. Use Node.js 18+."
fi
ok "Node.js $(node --version)"

cd frontend
npm install 2>&1 | tail -5
cd ..
ok "Frontend dependencies installed"

echo ""
echo "==========================================================="
echo "  Setup complete!"
echo ""
echo "  Start server : bash run.sh"
echo "  Open browser : http://localhost:5173"
echo "==========================================================="
echo ""
