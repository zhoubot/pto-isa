#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/docs/bytecode/samples
mkdir -p "$OUT"

PY=${PYTHON:-python3}

# Requires:
# - PTODSL submodule present
# - PYTHONPATH can import mlir.dialects.pto

"$PY" "$ROOT/PTODSL/examples/aot/add_static_multicore/add_builder.py" > "$OUT/add_static_multicore.pto"
"$PY" "$ROOT/PTODSL/examples/aot/add_dynamic_multicore/add_builder.py" > "$OUT/add_dynamic_multicore.pto"
"$PY" "$ROOT/PTODSL/examples/aot/relu_dynamic_multicore/relu_builder.py" > "$OUT/relu_dynamic_multicore.pto"
"$PY" "$ROOT/PTODSL/examples/aot/matmul_static_singlecore/matmul_builder.py" > "$OUT/matmul_static_singlecore.pto"
"$PY" "$ROOT/PTODSL/examples/aot/matmul_dynbatch_multicore/matmul_builder.py" > "$OUT/matmul_dynbatch_multicore.pto"

echo "Wrote samples to: $OUT"
ls -lh "$OUT"
