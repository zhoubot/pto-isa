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

## DebugInfo

`DEBUGINFO` is optional in PTO-BC v0.

- Emit DebugInfo during **encode** (ValueNames + OpLocations when source IR has `FileLineColLoc`):

```bash
PTOBC_EMIT_DEBUGINFO=1 build/ptobc/ptobc encode input.pto -o out.ptobc
```

- Print `loc(...)` during **decode** (parseable form):

```bash
PTOBC_PRINT_LOC=1 build/ptobc/ptobc decode out.ptobc -o out.pto
```

Notes:
- The canonical printer strips `loc(unknown)` to avoid noise.
- Float constants are printed in hex bitpattern form (`0x... : f32/f16/f64`) to guarantee lossless round-trip.
