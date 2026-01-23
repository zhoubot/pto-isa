from __future__ import annotations

from math import sqrt

from pto_as import PTO, scalar


def build():
    # Fixed demo shape (matches demos/cpu/flash_attention_demo).
    s = 64
    d = 32

    pto = PTO("pypto_flash_attention")
    pto.prologue()

    q = pto.tensor("q", (s, d), dtype="f32")
    k = pto.tensor("k", (s, d), dtype="f32")
    v = pto.tensor("v", (s, d), dtype="f32")
    o = pto.tensor("o", (s, d), dtype="f32")

    q_tile = pto.vec_tile("q_tile", dtype="f32", shape=(s, d))
    k_tile = pto.vec_tile("k_tile", dtype="f32", shape=(s, d))
    v_tile = pto.vec_tile("v_tile", dtype="f32", shape=(s, d))
    q_left = pto.left_tile("q_left", dtype="f32", shape=(s, d))
    kt_tile = pto.vec_tile("kt_tile", dtype="f32", shape=(d, s))
    k_right = pto.right_tile("k_right", dtype="f32", shape=(d, s))
    scores_acc = pto.acc_tile("scores_acc", dtype="f32", shape=(s, s))
    scores = pto.vec_tile("scores", dtype="f32", shape=(s, s))
    row_max = pto.vec_tile("row_max", dtype="f32", shape=(s, s), b="col")
    centered = pto.vec_tile("scores_centered", dtype="f32", shape=(s, s))
    exp_scores = pto.vec_tile("exp_scores", dtype="f32", shape=(s, s))
    row_sum = pto.vec_tile("row_sum", dtype="f32", shape=(s, s), b="col")
    probs = pto.vec_tile("probs", dtype="f32", shape=(s, s))
    p_left = pto.left_tile("p_left", dtype="f32", shape=(s, s))
    v_right = pto.right_tile("v_right", dtype="f32", shape=(s, d))
    out_acc = pto.acc_tile("out_acc", dtype="f32", shape=(s, d))

    scale = pto.const("scale", 1.0 / sqrt(d), scalar("f32"))

    for it in range(1):
        if it == 0:
            q_tile = pto.load(q)
            k_tile = pto.load(k)
            v_tile = pto.load(v)

            q_left = pto.mov(q_tile)
            kt_tile = pto.ttrans(k_tile, k_tile)
            k_right = pto.mov(kt_tile)

            scores_acc = pto.tmatmul(q_left, k_right)
            scores = pto.mov(scores_acc)
            scores = pto.tmuls(scores, scale)

            row_max = pto.trowmax(scores, scores)
            centered = pto.trowexpandsub(scores, row_max)
            exp_scores = pto.texp(centered)
            row_sum = pto.trowsum(exp_scores, exp_scores)
            probs = pto.trowexpanddiv(exp_scores, row_sum)

            p_left = pto.mov(probs)
            v_right = pto.mov(v_tile)
            out_acc = pto.tmatmul(p_left, v_right)
            pto.store(o, out_acc)
        else:
            pto.comment("unreachable")

    pto.comment("O = softmax(Q K^T / sqrt(d)) V (single-head demo)")
    pto.epilogue()
    return pto.program()
