# ptobc — PTO Bytecode encoder/decoder (C++ / MLIR)

This tool implements **PTO-BC v0** encoding/decoding as specified in:

- `docs/bytecode/pto-bc.md`

Design goals:
- Reuse MLIR infrastructure for parsing `.pto` and (optionally) printing.
- Provide a single standalone binary.

## Build

`ptobc encode` needs the **PTO dialect** to parse PTODSL-produced `.pto` (custom op/type syntax).
We get this by linking against **PTOAS::PTOIR** from a **built+installed** PTOAS.

By default, CMake looks for an in-tree install at:

- `PTOAS/install`

If your PTOAS install is elsewhere, pass `-DPTOAS_ROOT=/path/to/PTOAS/install`.

```bash
LLVM_BUILD_DIR=/home/zhoubot/llvm-workspace/llvm-project/build-shared

cmake -S tools/ptobc -B build/ptobc -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DPTOAS_ROOT="$PWD/PTOAS/install" \
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
