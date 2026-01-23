from __future__ import annotations

from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec_from_source


def make_fa16_kernel(*, target: str) -> KernelSpec:
    # This is a minimal "FA" placeholder kernel for validating the end-to-end Python->PTO-AS->CPU flow.
    # Signature (tensor args):
    #   arg0: q f16[16,16]
    #   arg1: k f16[16,16]
    #   arg2: v f16[16,16]
    #   arg3: out f16[16,16]
    #
    # Compute:
    #   out = q + k + v
    src = r"""
def fa16():
    prologue()

    q = tensor(dtype="f16", shape=(16, 16))
    k = tensor(dtype="f16", shape=(16, 16))
    v = tensor(dtype="f16", shape=(16, 16))
    out = tensor(dtype="f16", shape=(16, 16))

    tq = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    tk = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    tv = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    to = tile(loc="Vec", dtype="f16", rows=16, cols=16)

    tload(tq, q, 0, 0)
    tload(tk, k, 0, 0)
    tadd(to, tq, tk)
    tload(tv, v, 0, 0)
    tadd(to, to, tv)
    tstore(out, 0, 0, to)

    epilogue()
"""
    if target not in ("cpu", "npu"):
        raise ValueError("target must be cpu|npu")
    return compile_kernel_spec_from_source(src, func_name="fa16")


def make_fa16_pto(*, target: str) -> str:
    return make_fa16_kernel(target=target).pto
