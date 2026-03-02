# Kernels

This directory contains kernel/operator implementations that complement PTO Tile Lib.

Most kernel subdirectories are **self-contained mini-projects** (kernel + host + tests/scripts) with their own `README.md`, `CMakeLists.txt`, and `run.sh`.

## Where to start

- Manual (hand-tuned) NPU kernels: `kernels/manual/README.md`
- Custom operator scaffolding: `kernels/custom/README.md`
- End-to-end examples (including CPU): `examples/`

## Directory layout

- `manual/`: Hand-tuned kernels with explicit buffering/synchronization (NPU-focused)
  - `manual/a2a3/`: Kernels for A2/A3 platforms
    - `manual/a2a3/gemm_performance/`: High-performance GEMM example
    - `manual/a2a3/flash_atten/`: Manual Flash-Attention example
- `custom/`: Examples/scaffolding for custom kernel/operator extensions

## Notes

- Public interfaces live in `include/`; tests live in `tests/`.
- If you add a new kernel project here, prefer adding a small `README.md` and a `run.sh` so it can be discovered and executed consistently.
