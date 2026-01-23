from __future__ import annotations

from typing import Literal

from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec_from_source

Target = Literal["cpu", "npu"]


def make_gemm16_kernel(*, target: Target) -> KernelSpec:
    if target not in ("cpu", "npu"):
        raise ValueError("target must be cpu|npu")

    # Single cross-platform PTO kernel for both CPU and NPU.
    # (CPU TMATMUL accepts broader layouts; see include/pto/cpu/TMatmul.hpp.)
    src = r"""
def gemm16():
    prologue()

    a = tensor(dtype="f16", shape=(16, 16))
    b = tensor(dtype="f16", shape=(16, 16))
    c = tensor(dtype="f32", shape=(16, 16))

    a_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    a_left = tile(loc="Left", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="RowMajor")
    b_right = tile(loc="Right", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    tload(a_mat, a, 0, 0)
    tload(b_mat, b, 0, 0)
    tmov(a_left, a_mat)
    tmov(b_right, b_mat)
    tmatmul(c_acc, a_left, b_right)
    tstore(c, 0, 0, c_acc)

    epilogue()
"""
    return compile_kernel_spec_from_source(src, func_name="gemm16")


def make_gemm16_pto(*, target: Target) -> str:
    return make_gemm16_kernel(target=target).pto
