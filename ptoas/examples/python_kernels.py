from __future__ import annotations

# Example Python kernel definitions for the AST frontend.
# These functions are parsed (not executed) by ptoas/python/ast_frontend.py.

from ptoas.python.dsl import epilogue, prologue, tensor, tile, tadd, tload, tmatmul, tmov, tstore


def add16():
    prologue()

    x = tensor(dtype="f16", shape=(16, 16))
    y = tensor(dtype="f16", shape=(16, 16))
    z = tensor(dtype="f16", shape=(16, 16))

    tx = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    ty = tile(loc="Vec", dtype="f16", rows=16, cols=16)
    tz = tile(loc="Vec", dtype="f16", rows=16, cols=16)

    tload(tx, x, 0, 0)
    tload(ty, y, 0, 0)
    tadd(tz, tx, ty)
    tstore(z, 0, 0, tz)

    epilogue()


def gemm16_cpu():
    prologue()

    a = tensor(dtype="f16", shape=(16, 16))
    b = tensor(dtype="f16", shape=(16, 16))
    c = tensor(dtype="f32", shape=(16, 16))

    a_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    a_left = tile(loc="Left", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_right = tile(loc="Right", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    tload(a_mat, a, 0, 0)
    tload(b_mat, b, 0, 0)
    tmov(a_left, a_mat)
    tmov(b_right, b_mat)
    tmatmul(c_acc, a_left, b_right)
    tstore(c, 0, 0, c_acc)

    epilogue()


def gemm16():
    prologue()

    a = tensor(dtype="f16", shape=(16, 16))
    b = tensor(dtype="f16", shape=(16, 16))
    c = tensor(dtype="f32", shape=(16, 16))

    a_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")
    b_mat = tile(loc="Mat", dtype="f16", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    a_left = tile(loc="Left", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="RowMajor")
    b_right = tile(loc="Right", dtype="f16", rows=16, cols=16, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows=16, cols=16, blayout="ColMajor", slayout="RowMajor")

    tload(a_mat, a, 0, 0)
    tload(b_mat, b, 0, 0)
    tmov(a_left, a_mat)
    tmov(b_right, b_mat)
    tmatmul(c_acc, a_left, b_right)
    tstore(c, 0, 0, c_acc)

    epilogue()
