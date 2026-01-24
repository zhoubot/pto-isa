from __future__ import annotations

from pto_as import PTO


def pto_torch_flexattention():
    # Upstream ref: `~/github/pto-isa/examples/pto_torch_flexattention.py`
    pto = PTO("pto_torch_flexattention")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    row_max = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    centered = pto.vec(dtype="f32", shape=(16, 16))
    exp_x = pto.vec(dtype="f32", shape=(16, 16))
    row_sum = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    out = pto.vec(dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    row_max = pto.trowmax(tx, tmp)
    centered = pto.trowexpandsub(tx, row_max)
    exp_x = pto.texp(centered)
    row_sum = pto.trowsum(exp_x, tmp)
    out = pto.trowexpanddiv(exp_x, row_sum)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
