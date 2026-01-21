# `ptoas/mlir`: MLIR-based `ptoas` tool (prototype)

This folder is a starter implementation for a distributable `ptoas` binary:

- Input: PTO-AS `*.pto` (DPS syntax, `!pto.tensor<...>` / `!pto.tile<...>`).
- Optional: run an MLIR pass to insert `tsync` + event records (prototype heuristic).
- Output: Ascend CCE source (`*.cce`) and (optionally) compiled binary (`*.bin`) via `bisheng`.

## 1) Build MLIR (`mlir-opt`) from `~/llvm-project`

From anywhere:

```bash
cmake -G Ninja -S ~/llvm-project/llvm -B ~/llvm-project/build-mlir \\
  -DLLVM_ENABLE_PROJECTS=mlir \\
  -DLLVM_TARGETS_TO_BUILD=AArch64 \\
  -DLLVM_ENABLE_ASSERTIONS=ON \\
  -DCMAKE_BUILD_TYPE=Release

ninja -C ~/llvm-project/build-mlir mlir-opt
```

After this, `MLIRConfig.cmake` is usually at:

```text
~/llvm-project/build-mlir/lib/cmake/mlir/MLIRConfig.cmake
```

## 2) Build `ptoas` (this repo) against that MLIR build

From repo root:

```bash
cmake -G Ninja -S ptoas/mlir -B ptoas/mlir/build \\
  -DMLIR_DIR=$HOME/llvm-project/build-mlir/lib/cmake/mlir \\
  -DLLVM_DIR=$HOME/llvm-project/build-mlir/lib/cmake/llvm

ninja -C ptoas/mlir/build ptoas
```

The resulting binary:

```text
ptoas/mlir/build/bin/ptoas
```

## 3) Run

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest

./ptoas/mlir/build/bin/ptoas ptoas/examples/add16.pto -o /tmp/add16.cce
./ptoas/mlir/build/bin/ptoas ptoas/examples/add16.pto --emit-bin=/tmp/add16.bin
```

## Notes

- This is a prototype frontend: it parses PTO-AS directly (not MLIR text).
- The event insertion pass is intentionally conservative: it only inserts events for common pipeline changes like `tload -> tadd` and `tadd -> tstore`.

