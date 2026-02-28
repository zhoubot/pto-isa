# add_dynamic_multicore (CPU simulator)

End-to-end flow:

- PTODSL generates `add.pto` (dynamic length)
- PTOAS emits C++ (`add.cpp`)
- Compile + run on CPU with PTO-ISA headers (`__CPU_SIM`)

## Run

```bash
# Ensure PYTHONPATH has MLIR python bindings + PTO dialect, and ptoas is built.
# Example:
# export PYTHONPATH="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$PWD/PTOAS/install:$PYTHONPATH"

PTO_CPU_MAX_THREADS=16 bash examples/ptodsl_ptoas_cpu/add_dynamic_multicore/run.sh
```

Expected:

```text
PASS: CPU-sim vec_add_1d_dynamic (multicore)
```

## Note on tail padding

This kernel uses 1024-element tile loads/stores even for the last tile.
For CPU simulation, the runner allocates **padded** buffers to avoid OOB.
