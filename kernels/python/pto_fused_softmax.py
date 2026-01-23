from __future__ import annotations

from pto_as import PTO


def pto_fused_softmax():
    # Upstream ref: `~/github/pto-isa/examples/pto_fused_softmax.py`
    pto = PTO("pto_fused_softmax")
    pto.prologue()

    x = pto.tensor("x", (16, 16), dtype="f32", role="in")
    y = pto.tensor("y", (16, 16), dtype="f32", role="out")

    tx = pto.vec_tile("tx", dtype="f32", shape=(16, 16))
    row_max = pto.vec_tile("row_max", dtype="f32", shape=(16, 1), blayout="ColMajor")
    tmp = pto.vec_tile("tmp", dtype="f32", shape=(16, 16))
    centered = pto.vec_tile("centered", dtype="f32", shape=(16, 16))
    exp_x = pto.vec_tile("exp_x", dtype="f32", shape=(16, 16))
    row_sum = pto.vec_tile("row_sum", dtype="f32", shape=(16, 1), blayout="ColMajor")
    out = pto.vec_tile("out", dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    row_max = pto.trowmax(tx, tmp)
    centered = pto.trowexpandsub(tx, row_max)
    exp_x = pto.texp(centered)
    row_sum = pto.trowsum(exp_x, tmp)
    out = pto.trowexpanddiv(exp_x, row_sum)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
