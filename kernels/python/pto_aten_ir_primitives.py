from __future__ import annotations

from pto_as import PTO


def pto_aten_ir_primitives():
    # Upstream ref: `~/github/pto-isa/examples/pto_aten_ir_primitives.py`
    pto = PTO("pto_aten_ir_primitives")
    pto.prologue()

    x = pto.tensor("x", (16, 16), dtype="f32", role="in")
    y = pto.tensor("y", (16, 16), dtype="f32", role="in")
    z = pto.tensor("z", (16, 16), dtype="f32", role="out")

    tx = pto.vec_tile("tx", dtype="f32", shape=(16, 16))
    ty = pto.vec_tile("ty", dtype="f32", shape=(16, 16))
    tz = pto.vec_tile("tz", dtype="f32", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.load(y)
    tz = pto.tmul(tx, ty)
    pto.store(z, tz)

    pto.epilogue()
    return pto.program()
