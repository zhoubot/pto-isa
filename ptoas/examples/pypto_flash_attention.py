from __future__ import annotations

from math import sqrt

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BINDING_PY = _REPO_ROOT / "binding" / "python"
if str(_BINDING_PY) not in sys.path:
    sys.path.insert(0, str(_BINDING_PY))

from pto_as import PTO, scalar


def build():
    # Fixed demo shape (matches demos/cpu/flash_attention_demo).
    s = 64
    d = 32

    pto = PTO("pypto_flash_attention")
    pto.prologue()

    q = pto.tensor(dtype="f32", shape=(s, d), role="in")
    k = pto.tensor(dtype="f32", shape=(s, d), role="in")
    v = pto.tensor(dtype="f32", shape=(s, d), role="in")
    o = pto.tensor(dtype="f32", shape=(s, d), role="out")

    q_tile = pto.vec(dtype="f32", shape=(s, d))
    k_tile = pto.vec(dtype="f32", shape=(s, d))
    v_tile = pto.vec(dtype="f32", shape=(s, d))
    q_left = pto.left(dtype="f32", shape=(s, d))
    kt_tile = pto.vec(dtype="f32", shape=(d, s))
    k_right = pto.right(dtype="f32", shape=(d, s))
    scores_acc = pto.acc(dtype="f32", shape=(s, s))
    scores = pto.vec(dtype="f32", shape=(s, s))
    row_max = pto.vec(dtype="f32", shape=(s, s), b="col")
    centered = pto.vec(dtype="f32", shape=(s, s))
    exp_scores = pto.vec(dtype="f32", shape=(s, s))
    row_sum = pto.vec(dtype="f32", shape=(s, s), b="col")
    probs = pto.vec(dtype="f32", shape=(s, s))
    p_left = pto.left(dtype="f32", shape=(s, s))
    v_right = pto.right(dtype="f32", shape=(s, d))
    out_acc = pto.acc(dtype="f32", shape=(s, d))

    scale = pto.const("scale", 1.0 / sqrt(d), scalar("f32"))

    for it in range(1):
        if it == 0:
            q_tile = pto.load(q)
            k_tile = pto.load(k)
            v_tile = pto.load(v)

            q_left = pto.mov(q_tile)
            kt_tile = pto.trans(k_tile, k_tile)
            k_right = pto.mov(kt_tile)

            scores_acc = pto.matmul(q_left, k_right)
            scores = pto.mov(scores_acc)
            scores = pto.muls(scores, scale)

            row_max = pto.rowmax(scores, scores)
            centered = pto.rowexpandsub(scores, row_max)
            exp_scores = pto.exp(centered)
            row_sum = pto.rowsum(exp_scores, exp_scores)
            probs = pto.rowexpanddiv(exp_scores, row_sum)

            p_left = pto.mov(probs)
            v_right = pto.mov(v_tile)
            out_acc = pto.matmul(p_left, v_right)
            pto.store(o, out_acc)
        else:
            pto.comment("unreachable")

    pto.comment("O = softmax(Q K^T / sqrt(d)) V (single-head demo)")
    pto.epilogue()
    return pto.program()
