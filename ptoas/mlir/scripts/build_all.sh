#!/usr/bin/env bash
set -euo pipefail

LLVM_PROJECT_DIR="${LLVM_PROJECT_DIR:-$HOME/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$HOME/llvm-project/build-mlir}"
PTO_REPO_ROOT="${PTO_REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PTOAS_BUILD_DIR="${PTOAS_BUILD_DIR:-$PTO_REPO_ROOT/ptoas/mlir/build}"

if [[ ! -d "$LLVM_PROJECT_DIR/llvm" ]]; then
  echo "error: LLVM_PROJECT_DIR does not look like llvm-project: $LLVM_PROJECT_DIR" >&2
  exit 1
fi

cmake -G Ninja -S "$LLVM_PROJECT_DIR/llvm" -B "$LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=AArch64 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release

ninja -C "$LLVM_BUILD_DIR" mlir-opt

cmake -G Ninja -S "$PTO_REPO_ROOT/ptoas/mlir" -B "$PTOAS_BUILD_DIR" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"

ninja -C "$PTOAS_BUILD_DIR" ptoas

echo "Built:"
echo "  mlir-opt: $LLVM_BUILD_DIR/bin/mlir-opt"
echo "  ptoas:    $PTOAS_BUILD_DIR/bin/ptoas"

