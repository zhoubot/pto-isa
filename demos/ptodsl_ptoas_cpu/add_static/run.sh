#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)

PY=${PYTHON:-python3}
CXX=${CXX:-g++}

# Prefer a locally-built ptoas from the submodule; fall back to PATH.
PTOAS_BIN=${PTOAS_BIN:-$ROOT/PTOAS/build/tools/ptoas/ptoas}
if [[ ! -x "$PTOAS_BIN" ]]; then
  PTOAS_BIN=$(command -v ptoas || true)
fi
if [[ -z "${PTOAS_BIN}" || ! -x "${PTOAS_BIN}" ]]; then
  echo "[ERROR] ptoas not found. Set PTOAS_BIN or build PTOAS first." >&2
  exit 1
fi

cd "$HERE"

echo "[0/4] Ensure ptodsl is importable"
if ! "$PY" -c "import ptodsl" >/dev/null 2>&1; then
  echo "[INFO] Installing PTODSL (editable) into current Python: $PY" >&2
  "$PY" -m pip install -e "$ROOT/PTODSL/ptodsl" >/dev/null
fi

echo "[1/4] Generate PTO IR (.pto) from PTODSL"
# PTODSL requires MLIR python bindings + PTO dialect python package on PYTHONPATH.
# We don't guess those paths here. Instead, we fail with a clear message.
if ! "$PY" -c "import mlir, mlir.ir; from mlir.dialects import pto" >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[ERROR] Python MLIR bindings / PTO dialect not found.

You need to build LLVM/MLIR with python bindings and build PTOAS with python packages.
Then set (example):
  export PYTHONPATH="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$ROOT/PTOAS/install:$PYTHONPATH"

After that, re-run this script.
MSG
  exit 2
fi

"$PY" "$ROOT/PTODSL/examples/aot/add_static_multicore/add_builder.py" > add.pto

echo "[2/4] Emit C++ with PTOAS"
"$PTOAS_BIN" --enable-insert-sync add.pto -o add.cpp

echo "[3/4] Compile generated C++ against PTO-ISA CPU backend"
"$CXX" -std=c++20 -O2 -D__CPU_SIM -D__DAV_VEC__ \
  -I"$ROOT/include" \
  runner.cpp add.cpp -o run_cpu

echo "[4/4] Run"
./run_cpu
