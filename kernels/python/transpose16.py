from __future__ import annotations

from pto_as import PTO


def transpose16():
    pto = PTO("transpose16")
    pto.prologue()

    x = pto.tensor("x", (16, 16), dtype="f32", role="in")
    y = pto.tensor("y", (16, 16), dtype="f32", role="out")

    tx = pto.vec_tile("tx", dtype="f32", shape=(16, 16))
    ty = pto.vec_tile("ty", dtype="f32", shape=(16, 16))
    tmp = pto.vec_tile("tmp", dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.ttrans(tx, tmp)
    pto.store(y, ty)

    pto.epilogue()
    return pto.program()
