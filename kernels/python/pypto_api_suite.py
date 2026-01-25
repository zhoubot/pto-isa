from __future__ import annotations

"""
pyPTO API coverage suite.

Goals:
- Exercise a broad set of PTO instructions exposed by `pto_as.PTO` (and its aliases).
- Cover shapes: (16x16, 32x16, 16x32, 128x128, 256x256).
- Keep the *max tile size* <= 32KB (we cap tiles to at most 32x32; f32 => 16KB).

These kernels are AST-parsed (not executed). Avoid helper function calls; keep control
flow to `for range(...)` and simple `if`.
"""

from pto_as import PTO, scalar


def api_memory_ops():
    # Covers: tload/load, tmov/mov, tstore/store.
    #
    # NOTE: `tprefetch` is intentionally not used here because the CPU reference
    # backend does not provide `TPREFETCH_IMPL` (regression uses CPU as reference).
    pto = PTO("api_memory_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small
    # (AIV stack limit is 32KB).
    t = pto.vec(dtype="f32", shape=(16, 16))

    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            t = pto.load(x16, r, c)
            t = pto.mov(t)
            pto.store(y16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            t = pto.load(x32x16, r, c)
            t = pto.mov(t)
            pto.store(y32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            t = pto.load(x16x32, r, c)
            t = pto.mov(t)
            pto.store(y16x32, r, c, t)

    for r in range(0, 128, 16):
        for c in range(0, 128, 16):
            t = pto.load(x128, r, c)
            t = pto.mov(t)
            pto.store(y128, r, c, t)

    for r in range(0, 256, 16):
        for c in range(0, 256, 16):
            t = pto.load(x256, r, c)
            t = pto.mov(t)
            pto.store(y256, r, c, t)

    pto.epilogue()
    return pto.program()


def api_vec_binary_ops():
    # Covers: tadd, tsub, tmul, tmin, tmax.
    pto = PTO("api_vec_binary_ops")
    pto.prologue()

    # Inputs/outputs for 5 shapes.
    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    z16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    z32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    z16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    z128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    z256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small.
    a = pto.vec(dtype="f32", shape=(16, 16))
    b = pto.vec(dtype="f32", shape=(16, 16))
    t = pto.vec(dtype="f32", shape=(16, 16))

    # NOTE: helper function calls are not supported by the AST frontend. Inline below.
    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            a = pto.load(x16, r, c)
            b = pto.load(y16, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            a = pto.load(x32x16, r, c)
            b = pto.load(y32x16, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            a = pto.load(x16x32, r, c)
            b = pto.load(y16x32, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z16x32, r, c, t)

    for r in range(0, 128, 16):
        for c in range(0, 128, 16):
            a = pto.load(x128, r, c)
            b = pto.load(y128, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z128, r, c, t)

    for r in range(0, 256, 16):
        for c in range(0, 256, 16):
            a = pto.load(x256, r, c)
            b = pto.load(y256, r, c)
            t = pto.add(a, b)
            t = pto.sub(t, a)
            t = pto.mul(t, b)
            t = pto.min(t, a)
            t = pto.max(t, b)
            pto.store(z256, r, c, t)

    pto.epilogue()
    return pto.program()


def api_vec_unary_ops():
    # Covers: tabs, tneg, texp, tlog, tsqrt, trsqrt, trelu.
    pto = PTO("api_vec_unary_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small.
    t = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))

    # Pattern: keep values in a safe range:
    #   t = relu(log(exp(-abs(x))) + (rsqrt(exp(-abs(x)))^2))
    # which should be close to relu(-abs(x) + 1).

    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            t = pto.load(x16, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            t = pto.load(x32x16, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            t = pto.load(x16x32, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y16x32, r, c, t)

    for r in range(0, 128, 16):
        for c in range(0, 128, 16):
            t = pto.load(x128, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y128, r, c, t)

    for r in range(0, 256, 16):
        for c in range(0, 256, 16):
            t = pto.load(x256, r, c)
            tmp = pto.abs(t)
            tmp = pto.neg(tmp)
            tmp = pto.exp(tmp)
            t = pto.log(tmp)
            tmp = pto.sqrt(tmp)
            tmp = pto.rsqrt(tmp)
            tmp = pto.mul(tmp, tmp)
            t = pto.add(t, tmp)
            t = pto.relu(t)
            pto.store(y256, r, c, t)

    pto.epilogue()
    return pto.program()


def api_vec_scalar_ops():
    # Covers: tadds, tsubs, tmuls, tdivs, tmins, tmaxs, tlrelu.
    pto = PTO("api_vec_scalar_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Use a single 16x16 tile shape across all tensors to keep local storage small.
    t = pto.vec(dtype="f32", shape=(16, 16))

    s_add = pto.const("s_add", 0.125, scalar("f32"))
    s_sub = pto.const("s_sub", 0.25, scalar("f32"))
    s_mul = pto.const("s_mul", 1.5, scalar("f32"))
    s_div = pto.const("s_div", 0.75, scalar("f32"))
    s_min = pto.const("s_min", 0.5, scalar("f32"))
    s_max = pto.const("s_max", -0.5, scalar("f32"))
    s_lrelu = pto.const("s_lrelu", 0.01, scalar("f32"))

    for r in range(0, 16, 16):
        for c in range(0, 16, 16):
            t = pto.load(x16, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y16, r, c, t)

    for r in range(0, 32, 16):
        for c in range(0, 16, 16):
            t = pto.load(x32x16, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y32x16, r, c, t)

    for r in range(0, 16, 16):
        for c in range(0, 32, 16):
            t = pto.load(x16x32, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y16x32, r, c, t)

    for r in range(0, 128, 16):
        for c in range(0, 128, 16):
            t = pto.load(x128, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y128, r, c, t)

    for r in range(0, 256, 16):
        for c in range(0, 256, 16):
            t = pto.load(x256, r, c)
            t = pto.adds(t, s_add)
            t = pto.subs(t, s_sub)
            t = pto.muls(t, s_mul)
            t = pto.divs(t, s_div)
            t = pto.mins(t, s_min)
            t = pto.maxs(t, s_max)
            t = pto.lrelu(t, s_lrelu)
            pto.store(y256, r, c, t)

    pto.epilogue()
    return pto.program()


def api_row_reduce_ops():
    # Covers: rowmax (alias for trowmax), trowsum, tcolsum.
    pto = PTO("api_row_reduce_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16_rowmax = pto.tensor(dtype="f32", shape=(16, 1), role="out")
    y16_rowsum = pto.tensor(dtype="f32", shape=(16, 1), role="out")
    y16_colsum = pto.tensor(dtype="f32", shape=(1, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16_rowmax = pto.tensor(dtype="f32", shape=(32, 1), role="out")
    y32x16_rowsum = pto.tensor(dtype="f32", shape=(32, 1), role="out")
    y32x16_colsum = pto.tensor(dtype="f32", shape=(1, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32_rowmax = pto.tensor(dtype="f32", shape=(16, 1), role="out")
    y16x32_rowsum = pto.tensor(dtype="f32", shape=(16, 1), role="out")
    y16x32_colsum = pto.tensor(dtype="f32", shape=(1, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128_rowmax = pto.tensor(dtype="f32", shape=(128, 1), role="out")
    y128_rowsum = pto.tensor(dtype="f32", shape=(128, 1), role="out")
    y128_colsum = pto.tensor(dtype="f32", shape=(1, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256_rowmax = pto.tensor(dtype="f32", shape=(256, 1), role="out")
    y256_rowsum = pto.tensor(dtype="f32", shape=(256, 1), role="out")
    y256_colsum = pto.tensor(dtype="f32", shape=(1, 256), role="out")

    # Use a single 16x16 tile across all shapes to keep local storage small.
    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    rm = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    rs = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    rm_acc = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    rs_acc = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    cs = pto.vec(dtype="f32", shape=(1, 16))
    cs_acc = pto.vec(dtype="f32", shape=(1, 16))

    # (16,16)
    for r in range(0, 16, 16):
        tx = pto.load(x16, r, 0)
        rm_acc = pto.rowmax(tx, tmp)
        rs_acc = pto.rowsum(tx, tmp)
        pto.store(y16_rowmax, r, 0, rm_acc)
        pto.store(y16_rowsum, r, 0, rs_acc)
    for c in range(0, 16, 16):
        tx = pto.load(x16, 0, c)
        cs_acc = pto.colsum(tx)
        pto.store(y16_colsum, 0, c, cs_acc)

    # (32,16)
    for r in range(0, 32, 16):
        tx = pto.load(x32x16, r, 0)
        rm_acc = pto.rowmax(tx, tmp)
        rs_acc = pto.rowsum(tx, tmp)
        pto.store(y32x16_rowmax, r, 0, rm_acc)
        pto.store(y32x16_rowsum, r, 0, rs_acc)
    for c in range(0, 16, 16):
        tx = pto.load(x32x16, 0, c)
        cs_acc = pto.colsum(tx)
        for r in range(16, 32, 16):
            tx = pto.load(x32x16, r, c)
            cs = pto.colsum(tx)
            cs_acc = pto.add(cs_acc, cs)
        pto.store(y32x16_colsum, 0, c, cs_acc)

    # (16,32)
    for r in range(0, 16, 16):
        tx = pto.load(x16x32, r, 0)
        rm_acc = pto.rowmax(tx, tmp)
        rs_acc = pto.rowsum(tx, tmp)
        for c in range(16, 32, 16):
            tx = pto.load(x16x32, r, c)
            rm = pto.rowmax(tx, tmp)
            rs = pto.rowsum(tx, tmp)
            rm_acc = pto.max(rm_acc, rm)
            rs_acc = pto.add(rs_acc, rs)
        pto.store(y16x32_rowmax, r, 0, rm_acc)
        pto.store(y16x32_rowsum, r, 0, rs_acc)
    for c in range(0, 32, 16):
        tx = pto.load(x16x32, 0, c)
        cs_acc = pto.colsum(tx)
        pto.store(y16x32_colsum, 0, c, cs_acc)

    # (128,128)
    for r in range(0, 128, 16):
        tx = pto.load(x128, r, 0)
        rm_acc = pto.rowmax(tx, tmp)
        rs_acc = pto.rowsum(tx, tmp)
        for c in range(16, 128, 16):
            tx = pto.load(x128, r, c)
            rm = pto.rowmax(tx, tmp)
            rs = pto.rowsum(tx, tmp)
            rm_acc = pto.max(rm_acc, rm)
            rs_acc = pto.add(rs_acc, rs)
        pto.store(y128_rowmax, r, 0, rm_acc)
        pto.store(y128_rowsum, r, 0, rs_acc)
    for c in range(0, 128, 16):
        tx = pto.load(x128, 0, c)
        cs_acc = pto.colsum(tx)
        for r in range(16, 128, 16):
            tx = pto.load(x128, r, c)
            cs = pto.colsum(tx)
            cs_acc = pto.add(cs_acc, cs)
        pto.store(y128_colsum, 0, c, cs_acc)

    # (256,256)
    for r in range(0, 256, 16):
        tx = pto.load(x256, r, 0)
        rm_acc = pto.rowmax(tx, tmp)
        rs_acc = pto.rowsum(tx, tmp)
        for c in range(16, 256, 16):
            tx = pto.load(x256, r, c)
            rm = pto.rowmax(tx, tmp)
            rs = pto.rowsum(tx, tmp)
            rm_acc = pto.max(rm_acc, rm)
            rs_acc = pto.add(rs_acc, rs)
        pto.store(y256_rowmax, r, 0, rm_acc)
        pto.store(y256_rowsum, r, 0, rs_acc)
    for c in range(0, 256, 16):
        tx = pto.load(x256, 0, c)
        cs_acc = pto.colsum(tx)
        for r in range(16, 256, 16):
            tx = pto.load(x256, r, c)
            cs = pto.colsum(tx)
            cs_acc = pto.add(cs_acc, cs)
        pto.store(y256_colsum, 0, c, cs_acc)

    pto.epilogue()
    return pto.program()


def api_row_expand_ops():
    # Covers: trowexpand, trowexpandadd/sub/mul/div.
    pto = PTO("api_row_expand_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    v16 = pto.tensor(dtype="f32", shape=(16, 1), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    v32x16 = pto.tensor(dtype="f32", shape=(32, 1), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    v16x32 = pto.tensor(dtype="f32", shape=(16, 1), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    v128 = pto.tensor(dtype="f32", shape=(128, 1), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    v256 = pto.tensor(dtype="f32", shape=(256, 1), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Use a single 16x16 tile across all shapes to keep local storage small.
    t = pto.vec(dtype="f32", shape=(16, 16))
    rv = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    out = pto.vec(dtype="f32", shape=(16, 16))

    one = pto.const("one", 1.0, scalar("f32"))

    # (16,16)
    for r0 in range(0, 16, 16):
        rv = pto.load(v16, r0, 0)
        rv = pto.abs(rv)
        rv = pto.adds(rv, one)
        for c0 in range(0, 16, 16):
            t = pto.load(x16, r0, c0)
            out = pto.rowexpand(rv)
            t = pto.rowexpandadd(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandsub(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandmul(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpanddiv(t, rv)
            out = pto.add(out, t)
            pto.store(y16, r0, c0, out)

    # (32,16)
    for r0 in range(0, 32, 16):
        rv = pto.load(v32x16, r0, 0)
        rv = pto.abs(rv)
        rv = pto.adds(rv, one)
        for c0 in range(0, 16, 16):
            t = pto.load(x32x16, r0, c0)
            out = pto.rowexpand(rv)
            t = pto.rowexpandadd(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandsub(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandmul(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpanddiv(t, rv)
            out = pto.add(out, t)
            pto.store(y32x16, r0, c0, out)

    # (16,32)
    for r0 in range(0, 16, 16):
        rv = pto.load(v16x32, r0, 0)
        rv = pto.abs(rv)
        rv = pto.adds(rv, one)
        for c0 in range(0, 32, 16):
            t = pto.load(x16x32, r0, c0)
            out = pto.rowexpand(rv)
            t = pto.rowexpandadd(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandsub(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandmul(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpanddiv(t, rv)
            out = pto.add(out, t)
            pto.store(y16x32, r0, c0, out)

    # (128,128)
    for r0 in range(0, 128, 16):
        rv = pto.load(v128, r0, 0)
        rv = pto.abs(rv)
        rv = pto.adds(rv, one)
        for c0 in range(0, 128, 16):
            t = pto.load(x128, r0, c0)
            out = pto.rowexpand(rv)
            t = pto.rowexpandadd(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandsub(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandmul(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpanddiv(t, rv)
            out = pto.add(out, t)
            pto.store(y128, r0, c0, out)

    # (256,256)
    for r0 in range(0, 256, 16):
        rv = pto.load(v256, r0, 0)
        rv = pto.abs(rv)
        rv = pto.adds(rv, one)
        for c0 in range(0, 256, 16):
            t = pto.load(x256, r0, c0)
            out = pto.rowexpand(rv)
            t = pto.rowexpandadd(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandsub(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpandmul(t, rv)
            out = pto.add(out, t)
            t = pto.rowexpanddiv(t, rv)
            out = pto.add(out, t)
            pto.store(y256, r0, c0, out)

    pto.epilogue()
    return pto.program()


def api_transpose_ops():
    # Covers: ttrans (including non-square 32x16 and 16x32).
    pto = PTO("api_transpose_ops")
    pto.prologue()

    x16 = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    x32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="in")
    y32x16 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    x16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="in")
    y16x32 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    x128 = pto.tensor(dtype="f32", shape=(128, 128), role="in")
    y128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    x256 = pto.tensor(dtype="f32", shape=(256, 256), role="in")
    y256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    tx16 = pto.vec(dtype="f32", shape=(16, 16))
    tmp16 = pto.vec(dtype="f32", shape=(16, 16))
    ty16 = pto.vec(dtype="f32", shape=(16, 16))

    tx32x16 = pto.vec(dtype="f32", shape=(32, 16))
    tmp32x16 = pto.vec(dtype="f32", shape=(32, 16))
    ty16x32 = pto.vec(dtype="f32", shape=(16, 32))

    tx16x32 = pto.vec(dtype="f32", shape=(16, 32))
    tmp16x32 = pto.vec(dtype="f32", shape=(16, 32))
    ty32x16 = pto.vec(dtype="f32", shape=(32, 16))

    tx32 = pto.vec(dtype="f32", shape=(32, 32))
    tmp32 = pto.vec(dtype="f32", shape=(32, 32))
    ty32 = pto.vec(dtype="f32", shape=(32, 32))

    # (16,16)
    tx16 = pto.load(x16)
    ty16 = pto.trans(tx16, tmp16)
    pto.store(y16, ty16)

    # (32,16) -> (16,32)
    tx32x16 = pto.load(x32x16)
    ty16x32 = pto.trans(tx32x16, tmp32x16)
    pto.store(y32x16, ty16x32)

    # (16,32) -> (32,16)
    tx16x32 = pto.load(x16x32)
    ty32x16 = pto.trans(tx16x32, tmp16x32)
    pto.store(y16x32, ty32x16)

    # (128,128) tiled (32,32)
    for r in range(0, 128, 32):
        for c in range(0, 128, 32):
            tx32 = pto.load(x128, r, c)
            ty32 = pto.trans(tx32, tmp32)
            pto.store(y128, c, r, ty32)

    # (256,256) tiled (32,32)
    for r in range(0, 256, 32):
        for c in range(0, 256, 32):
            tx32 = pto.load(x256, r, c)
            ty32 = pto.trans(tx32, tmp32)
            pto.store(y256, c, r, ty32)

    pto.epilogue()
    return pto.program()


def api_matmul_ops():
    # Covers: matmul (alias for tmatmul), tmatmul_acc.
    #
    # Shapes covered:
    # - (16,16): A(16x16) * B(16x16) -> C(16x16)
    # - (32,16): A(32x16) * B(16x16) -> C(32x16)
    # - (16,32): A(16x16) * B(16x32) -> C(16x32)
    # - (128,128): A(128x128) * B(128x128) -> C(128x128) (tiled 32)
    # - (256,256): A(256x256) * B(256x256) -> C(256x256) (tiled 32)
    pto = PTO("api_matmul_ops")
    pto.prologue()

    a16 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b16 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    c16 = pto.tensor(dtype="f32", shape=(16, 16), role="out")

    a32x16 = pto.tensor(dtype="f16", shape=(32, 16), role="in")
    b16_for_32x16 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    c32x16 = pto.tensor(dtype="f32", shape=(32, 16), role="out")

    a16_for_16x32 = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    b16x32 = pto.tensor(dtype="f16", shape=(16, 32), role="in")
    c16x32 = pto.tensor(dtype="f32", shape=(16, 32), role="out")

    a128 = pto.tensor(dtype="f16", shape=(128, 128), role="in")
    b128 = pto.tensor(dtype="f16", shape=(128, 128), role="in")
    c128 = pto.tensor(dtype="f32", shape=(128, 128), role="out")

    a256 = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    b256 = pto.tensor(dtype="f16", shape=(256, 256), role="in")
    c256 = pto.tensor(dtype="f32", shape=(256, 256), role="out")

    # Tiles.
    a_m16 = pto.mat(dtype="f16", shape=(16, 16))
    b_m16 = pto.mat(dtype="f16", shape=(16, 16))
    a_l16 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_r16 = pto.right(dtype="f16", shape=(16, 16))
    c_acc16 = pto.acc(dtype="f32", shape=(16, 16))

    a_m32x16 = pto.mat(dtype="f16", shape=(32, 16))
    b_m16_for_32x16 = pto.mat(dtype="f16", shape=(16, 16))
    a_l32x16 = pto.left(dtype="f16", shape=(32, 16), blayout="ColMajor", slayout="RowMajor")
    b_r16_for_32x16 = pto.right(dtype="f16", shape=(16, 16))
    c_acc32x16 = pto.acc(dtype="f32", shape=(32, 16))

    a_m16_for_16x32 = pto.mat(dtype="f16", shape=(16, 16))
    b_m16x32 = pto.mat(dtype="f16", shape=(16, 32))
    a_l16_for_16x32 = pto.left(dtype="f16", shape=(16, 16), blayout="ColMajor", slayout="RowMajor")
    b_r16x32 = pto.right(dtype="f16", shape=(16, 32))
    c_acc16x32 = pto.acc(dtype="f32", shape=(16, 32))

    a_m32 = pto.mat(dtype="f16", shape=(32, 32))
    b_m32 = pto.mat(dtype="f16", shape=(32, 32))
    a_l32 = pto.left(dtype="f16", shape=(32, 32), blayout="ColMajor", slayout="RowMajor")
    b_r32 = pto.right(dtype="f16", shape=(32, 32))
    c_acc32 = pto.acc(dtype="f32", shape=(32, 32))

    # (16,16)
    a_m16 = pto.load(a16)
    b_m16 = pto.load(b16)
    a_l16 = pto.mov(a_m16)
    b_r16 = pto.mov(b_m16)
    c_acc16 = pto.matmul(a_l16, b_r16)
    pto.store(c16, c_acc16)

    # (32,16)
    a_m32x16 = pto.load(a32x16)
    b_m16_for_32x16 = pto.load(b16_for_32x16)
    a_l32x16 = pto.mov(a_m32x16)
    b_r16_for_32x16 = pto.mov(b_m16_for_32x16)
    c_acc32x16 = pto.matmul(a_l32x16, b_r16_for_32x16)
    pto.store(c32x16, c_acc32x16)

    # (16,32)
    a_m16_for_16x32 = pto.load(a16_for_16x32)
    b_m16x32 = pto.load(b16x32)
    a_l16_for_16x32 = pto.mov(a_m16_for_16x32)
    b_r16x32 = pto.mov(b_m16x32)
    c_acc16x32 = pto.matmul(a_l16_for_16x32, b_r16x32)
    pto.store(c16x32, c_acc16x32)

    # (128,128) tiled matmul using tmatmul_acc.
    for r in range(0, 128, 32):
        for c in range(0, 128, 32):
            for k in range(0, 128, 32):
                a_m32 = pto.load(a128, r, k)
                b_m32 = pto.load(b128, k, c)
                a_l32 = pto.mov(a_m32)
                b_r32 = pto.mov(b_m32)
                if k == 0:
                    c_acc32 = pto.matmul(a_l32, b_r32)
                else:
                    c_acc32 = pto.matmul_acc(c_acc32, a_l32, b_r32)
            pto.store(c128, r, c, c_acc32)

    # (256,256) tiled matmul using tmatmul_acc.
    for r in range(0, 256, 32):
        for c in range(0, 256, 32):
            for k in range(0, 256, 32):
                a_m32 = pto.load(a256, r, k)
                b_m32 = pto.load(b256, k, c)
                a_l32 = pto.mov(a_m32)
                b_r32 = pto.mov(b_m32)
                if k == 0:
                    c_acc32 = pto.matmul(a_l32, b_r32)
                else:
                    c_acc32 = pto.matmul_acc(c_acc32, a_l32, b_r32)
            pto.store(c256, r, c, c_acc32)

    pto.epilogue()
    return pto.program()
