# `kernels/python`

Python-first examples that generate PTO-AS (`*.pto`), then drive `ptoas` to emit:

- device kernel source (`foo.cpp`, compiled via `bisheng -xcce` on a real Ascend environment)
- CPU-simulator C++ (`*.cpu.cpp`, runnable on macOS/Linux without NPU)
- host launcher (`host.cpp`, launches the fatobj `.so` on NPU via ACL)

Examples:

- `kernels/python/fa`: a small “FA” toy kernel (vector add3) for end-to-end validation on CPU.
- `kernels/python/gemm`: a 16x16 GEMM kernel (cube core on NPU / matching CPU simulator layout on CPU).

Flat (single-file) kernels live directly in this folder (no per-example subdirs), e.g.
`add16.py`, `gemm16.py`, `softmax16.py`, and ports of upstream examples like `pto_isa_sinh.py`.

## Regression (NPU)

Requires a working Ascend toolkit install (set `ASCEND_HOME_PATH` or pass `--ascend-home`)
and a built `ptoas` binary.

Run everything:

`python3 kernels/python/run_regression.py --run-mode npu`

Run a single case:

`python3 kernels/python/run_regression.py --run-mode npu --cases pto_fused_softmax`

## SPMD / multi-block cases

Some kernels (e.g. `spmd_tiled_add256`, `spmd_tiled_transpose256`, `spmd_tiled_rowsum256`) are intended to
run with `block_dim > 1`. The regression runner supports this via per-case overrides in
`kernels/python/run_regression.py` (cases can carry a `block_dim` field).
