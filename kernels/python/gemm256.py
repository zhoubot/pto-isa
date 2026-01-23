from __future__ import annotations

from pto_as import PTO


def gemm256():
    # C[256,256] = A[256,256] @ B[256,256], using (16,16,16) tiles.
    pto = PTO("gemm256")
    pto.prologue()

    a = pto.tensor("a", (256, 256), dtype="f16", role="in")
    b = pto.tensor("b", (256, 256), dtype="f16", role="in")
    c = pto.tensor("c", (256, 256), dtype="f32", role="out")

    a_mat = pto.mat_tile("a_mat", dtype="f16", shape=(16, 16))
    b_mat = pto.mat_tile("b_mat", dtype="f16", shape=(16, 16))

    a_left_0 = pto.left_tile("a_left_0", dtype="f16", shape=(16, 16))
    a_left_1 = pto.left_tile("a_left_1", dtype="f16", shape=(16, 16))
    b_right_0 = pto.right_tile("b_right_0", dtype="f16", shape=(16, 16))
    b_right_1 = pto.right_tile("b_right_1", dtype="f16", shape=(16, 16))
    c_acc = pto.acc_tile("c_acc", dtype="f32", shape=(16, 16))

    for mi in range(0, 256, 16):
        for nj in range(0, 256, 16):
            for kk in range(0, 256, 16):
                a_mat = pto.load(a, mi, kk)
                b_mat = pto.load(b, kk, nj)

                it0 = kk // 16
                lane = it0 % 2
                if lane == 0:
                    a_left_0 = pto.mov(a_mat)
                    b_right_0 = pto.mov(b_mat)
                    if kk == 0:
                        c_acc = pto.tmatmul(a_left_0, b_right_0)
                    else:
                        c_acc = pto.tmatmul_acc(c_acc, a_left_0, b_right_0)
                else:
                    a_left_1 = pto.mov(a_mat)
                    b_right_1 = pto.mov(b_mat)
                    c_acc = pto.tmatmul_acc(c_acc, a_left_1, b_right_1)

            pto.store(c, mi, nj, c_acc)

    pto.epilogue()
    return pto.program()
