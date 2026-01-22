# `kernels/python`

Python-first examples that generate PTO-AS (`*.pto`), then drive `ptoas` to emit:

- device kernel source (`foo.cpp`, compiled via `bisheng -xcce` on a real Ascend environment)
- CPU-simulator C++ (`*.cpu.cpp`, runnable on macOS/Linux without NPU)
- host launcher (`host.cpp`, launches the fatobj `.so` on NPU via ACL)

Examples:

- `kernels/python/fa`: a small “FA” toy kernel (vector add3) for end-to-end validation on CPU.
- `kernels/python/gemm`: a 16x16 GEMM kernel (cube core on NPU / matching CPU simulator layout on CPU).
