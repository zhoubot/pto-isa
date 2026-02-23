# PTODSL → PTOAS → PTO-ISA CPU backend (end-to-end)

This example demonstrates an **end-to-end** workflow:

1. **PTODSL** (Python) generates PTO MLIR text (`.pto`)
2. **PTOAS** emits C++ (`add.cpp`)
3. The generated C++ is compiled and executed using **PTO-ISA CPU simulator** (`__CPU_SIM`)

> This is a **CPU-only** flow. No Ascend drivers / CANN / bisheng required.

## Prerequisites

- A working C++ compiler (GCC/Clang)
- `ptoas` built (recommended from the `PTOAS` submodule)
- Python environment that can import:
  - `mlir` python bindings
  - `mlir.dialects.pto` (from PTOAS python packages)

## Run

From repo root:

```bash
# If you built LLVM/MLIR+python and PTOAS, make sure PYTHONPATH includes them.
# Example (adjust paths):
# export PYTHONPATH="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$PWD/PTOAS/install:$PYTHONPATH"

bash examples/ptodsl_ptoas_cpu/add_static/run.sh
```

Expected output:

```
PASS: CPU-sim vec_add_kernel_2d_dynamic
```

## What this validates

- PTODSL can build a PTO MLIR program for a simple kernel
- PTOAS can lower PTO MLIR → C++
- The generated C++ can be compiled and executed on CPU via PTO-ISA headers
- CPU launch emulation uses `pto::cpu_sim::set_launch_context(block, subblock, subblock_dim)`
