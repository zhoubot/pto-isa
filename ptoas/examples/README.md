# `ptoas/examples`

- These examples use the new-format PTO-AS surface syntax (`pto.make_tensor_view`, `pto.alloc_tile`, and `%dst = pto.op ...` sugar).
- `add16.pto`: minimal PTO-AS program that loads one 16x16 tile from two GM tensors, adds them, and stores back.
- `add16_min.pto`: minimal MLIR `ptoas` input used by `ptoas/mlir/README.md`.
- `add16_e2e.pto`: “complete-ish” add example with prologue/epilogue + block id queries (used by `ptoas/tools/run_e2e_npu.py`).
- `gemm16_e2e.pto`: 16x16 GEMM (cube core) f16xf16->f32 (used by `ptoas/tools/run_e2e_npu.py`).
- `pypto_flash_attention.py`: Python object-DSL example (AST-parsed) that emits a larger PTO-AS program.

Related:

- `binding/python/ptoas/python/pto_asm.py` can generate equivalent `*.pto` programs from Python.
