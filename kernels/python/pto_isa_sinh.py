from __future__ import annotations

from pto_as import PTO, scalar


def pto_isa_sinh():
    # Upstream ref: `~/github/pto-isa/examples/pto_isa_sinh.py`
    # Identity: sinh(x) = (exp(x) - exp(-x)) / 2
    pto = PTO("pto_isa_sinh")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    neg_x = pto.vec(dtype="f32", shape=(16, 16))
    exp_x = pto.vec(dtype="f32", shape=(16, 16))
    exp_neg_x = pto.vec(dtype="f32", shape=(16, 16))
    diff = pto.vec(dtype="f32", shape=(16, 16))
    out = pto.vec(dtype="f32", shape=(16, 16))

    half = pto.const("half", 0.5, scalar("f32"))

    tx = pto.load(x)
    neg_x = pto.tneg(tx)
    exp_x = pto.texp(tx)
    exp_neg_x = pto.texp(neg_x)
    diff = pto.tsub(exp_x, exp_neg_x)
    out = pto.tmuls(diff, half)
    pto.store(y, out)

    pto.epilogue()
    return pto.program()
