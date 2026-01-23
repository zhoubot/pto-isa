from __future__ import annotations

from pto_as import PTO


def pto_torch_tensor():
    # Upstream ref: `~/github/pto-isa/examples/pto_torch_tensor.py`
    pto = PTO("pto_torch_tensor")
    pto.prologue()

    x = pto.tensor("x", (16, 16), dtype="f16", role="in")
    y = pto.tensor("y", (16, 16), dtype="f16", role="in")
    z = pto.tensor("z", (16, 16), dtype="f16", role="out")

    tx = pto.vec_tile("tx", dtype="f16", shape=(16, 16))
    ty = pto.vec_tile("ty", dtype="f16", shape=(16, 16))
    tz = pto.vec_tile("tz", dtype="f16", shape=(16, 16))

    tx = pto.load(x)
    ty = pto.load(y)
    tz = pto.tadd(tx, ty)
    pto.store(z, tz)

    pto.epilogue()
    return pto.program()
