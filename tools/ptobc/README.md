# ptobc — PTO Bytecode encoder/decoder (C++ / MLIR)

This tool implements **PTO-BC v0** encoding/decoding as specified in:

- `docs/bytecode/pto-bc.md`

Design goals:
- Reuse MLIR infrastructure for parsing `.pto` and (optionally) printing.
- Provide a single standalone binary.

## Build

This repo already builds LLVM/MLIR for PTOAS. Point `ptobc` at the same build:

```bash
LLVM_BUILD_DIR=/home/zhoubot/llvm-workspace/llvm-project/build-shared

cmake -S tools/ptobc -B build/ptobc -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build/ptobc
```

## Usage

Encode:

```bash
build/ptobc/ptobc encode input.pto -o out.ptobc
```

Decode:

```bash
build/ptobc/ptobc decode out.ptobc -o out.pto
```
