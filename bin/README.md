This folder contains prebuilt helper binaries for PTO-ISA.

- `bin/ptoas` is a small wrapper that dispatches to an OS/arch-specific binary:
  - Linux aarch64: `bin/linux-aarch64/ptoas` (**included**)
  - macOS aarch64: `bin/macos-aarch64/ptoas` (**not included**; build from source)

If you need a missing binary, build `ptoas` from source:

```bash
cmake -G Ninja -S ptoas/mlir -B ptoas/mlir/build \
  -DMLIR_DIR=$HOME/llvm-project/build-mlir/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build-mlir/lib/cmake/llvm

ninja -C ptoas/mlir/build ptoas
```

Then copy the resulting `ptoas/mlir/build/bin/ptoas` to the appropriate subfolder.

