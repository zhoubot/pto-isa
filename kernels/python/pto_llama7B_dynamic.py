from __future__ import annotations

from pto_as import PTO


def pto_llama7B_dynamic():
    # Upstream ref: `~/github/pto-isa/examples/pto_llama7B_dynamic.py`
    pto = PTO("pto_llama7B_dynamic")
    pto.prologue()

    x = pto.tensor("x", (32, 16), dtype="f32", role="in")
    y = pto.tensor("y", (32, 16), dtype="f32", role="out")

    tx = pto.vec_tile("tx", dtype="f32", shape=(16, 16))
    row_max = pto.vec_tile("row_max", dtype="f32", shape=(16, 1), blayout="ColMajor")
    tmp = pto.vec_tile("tmp", dtype="f32", shape=(16, 16))
    centered = pto.vec_tile("centered", dtype="f32", shape=(16, 16))
    exp_x = pto.vec_tile("exp_x", dtype="f32", shape=(16, 16))
    row_sum = pto.vec_tile("row_sum", dtype="f32", shape=(16, 1), blayout="ColMajor")
    out = pto.vec_tile("out", dtype="f32", shape=(16, 16))

    for it in range(2):
        r0 = it * 16
        tx = pto.load(x, r0, 0)
        row_max = pto.trowmax(tx, tmp)
        centered = pto.trowexpandsub(tx, row_max)
        exp_x = pto.texp(centered)
        row_sum = pto.trowsum(exp_x, tmp)
        out = pto.trowexpanddiv(exp_x, row_sum)
        pto.store(y, r0, 0, out)

    pto.epilogue()
    return pto.program()
