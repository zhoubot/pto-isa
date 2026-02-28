#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# Ensure submodules are present.
git -C "$ROOT" submodule update --init --recursive >/dev/null

EXAMPLES=(
  add_static
  add_dynamic_multicore
  relu_dynamic_multicore
  matmul_static_singlecore
  matmul_dynbatch_multicore
)

for ex in "${EXAMPLES[@]}"; do
  echo "============================================================"
  echo "[RUN] $ex"
  echo "============================================================"
  bash "$ROOT/demos/ptodsl_ptoas_cpu/$ex/run.sh"
  echo
  echo

done

echo "Done."
