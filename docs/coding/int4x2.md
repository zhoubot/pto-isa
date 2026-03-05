# int4x2_t (packed int4) on A2/A3

PTO uses `int4x2_t` as a **packed storage type** that holds **two 4-bit signed integers** in **one byte**.

- On A2/A3 **AICORE** compilation, `int4x2_t` comes from the CCE toolchain builtin headers (`__clang_cce_types.h`).
- On the host side (ST / CPU compilation units), PTO provides a lightweight definition to enable compilation.

## What is supported today

In the A2/A3 NPU backend, `int4x2_t` is currently supported as a **byte-sized element type** for:

- `TLOAD` / `TSTORE` data movement (ND ↔ ND, ND ↔ UB), by treating `int4x2_t` as a 1-byte payload.

A minimal ST testcase is provided:

- `tests/npu/a2a3/src/st/testcase/tint4x2` (`TINT4X2Test.case_copy_64x64`, `TINT4X2Test.case_copy_32x128`, `TINT4X2Test.case_copy_32x96_v32x95`)

This testcase validates **bitwise correctness** of a `TLOAD` + `TSTORE` copy roundtrip on NPU.

## Notes

- `int4x2_t` here is a *storage format*. Arithmetic (e.g. INT4 matmul) requires additional ISA/operator support and is **not** covered by this testcase.
