#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)

PY=${PYTHON:-python3}
CXX=${CXX:-g++}

PTOAS_BIN=${PTOAS_BIN:-$ROOT/PTOAS/build/tools/ptoas/ptoas}
if [[ ! -x "$PTOAS_BIN" ]]; then
  PTOAS_BIN=$(command -v ptoas || true)
fi
if [[ -z "${PTOAS_BIN}" || ! -x "${PTOAS_BIN}" ]]; then
  echo "[ERROR] ptoas not found. Set PTOAS_BIN or build PTOAS first." >&2
  exit 1
fi

cd "$HERE"

if ! "$PY" -c "import ptodsl" >/dev/null 2>&1; then
  "$PY" -m pip install -e "$ROOT/PTODSL/ptodsl" >/dev/null
fi

if ! "$PY" -c "import mlir, mlir.ir; from mlir.dialects import pto" >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[ERROR] Python MLIR bindings / PTO dialect not found.
Set PYTHONPATH like:
  export PYTHONPATH="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$ROOT/PTOAS/install:$PYTHONPATH"
MSG
  exit 2
fi

echo "[1/4] Generate .pto (small CPU-friendly sizes)"
"$PY" gen_pto.py > matmul.pto

echo "[2/4] PTOAS → C++"
"$PTOAS_BIN" --enable-insert-sync matmul.pto -o matmul.cpp

echo "[3/4] Compile (CPU sim, cube)"
"$CXX" -std=c++20 -O2 -D__CPU_SIM -D__DAV_CUBE__ \
  -I"$ROOT/include" \
  runner.cpp matmul.cpp -o run_cpu

echo "[4/4] Run"
./run_cpu
