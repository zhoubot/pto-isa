from __future__ import annotations

from pto_as import PTO


def gemm16():
    pto = PTO("gemm16")
    pto.prologue()

    a = pto.tensor("a", (16, 16), dtype="f16", role="in")
    b = pto.tensor("b", (16, 16), dtype="f16", role="in")
    c = pto.tensor("c", (16, 16), dtype="f32", role="out")

    a_mat = pto.mat_tile("a_mat", dtype="f16", shape=(16, 16))
    b_mat = pto.mat_tile("b_mat", dtype="f16", shape=(16, 16))

    a_left = pto.left_tile("a_left", dtype="f16", shape=(16, 16))
    b_right = pto.right_tile("b_right", dtype="f16", shape=(16, 16))
    c_acc = pto.acc_tile("c_acc", dtype="f32", shape=(16, 16))

    a_mat = pto.load(a)
    b_mat = pto.load(b)
    a_left = pto.mov(a_mat)
    b_right = pto.mov(b_mat)
    c_acc = pto.tmatmul(a_left, b_right)
    pto.store(c, c_acc)

    pto.epilogue()
    return pto.program()
