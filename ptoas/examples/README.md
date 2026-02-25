# `ptoas/examples`

- These examples are valid MLIR modules for the LLVM `ptoas` tool (vendored at `bin/ptoas`).
- `add16.pto`: minimal vec add kernel (16×16).
- `add16_min.pto`: same kernel, kept as a small “does it compile” input.
- `add16_e2e.pto`: add16 with an embedded host-spec header (used by `ptoas/tools/run_e2e_cpu.py` and `ptoas/tools/run_e2e_npu.py`).
- `gemm16_e2e.pto`: 16×16 GEMM (cube core) f16xf16->f32 with an embedded host-spec header (used by `ptoas/tools/run_e2e_cpu.py` and `ptoas/tools/run_e2e_npu.py`).
- `pypto_flash_attention.py`: Python object-DSL example (AST-parsed) that emits a larger PTO-AS program.

Related:

- These `*.pto` files can be regenerated from Python kernels via `binding/python/ptoas/python/binding.py`.
