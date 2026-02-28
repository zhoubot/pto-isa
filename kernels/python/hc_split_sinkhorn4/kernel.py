from __future__ import annotations

from dataclasses import dataclass

from pto_as import PTO
from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec


@dataclass(frozen=True)
class HcSplitSinkhorn4Config:
    n: int
    sinkhorn_iters: int = 20
    eps: float = 1.0e-6


def hc_split_sinkhorn4():
    """
    TileLang -> pyPTO port for hc=4 split + Sinkhorn normalization.

    Inputs:
      mixes:     [n, 24] f32   (layout: [pre(4), post(4), comb(16)])
      hc_scale:  [1, 3]  f32   (scale0, scale1, scale2)
      hc_base:   [1, 24] f32   (base for the 24 mix channels)

    Outputs:
      pre:   [n, 4]  f32
      post:  [n, 4]  f32
      comb:  [n, 16] f32  (flattened 4x4)

    Notes:
    - This is written for clarity and correctness. It parallelizes across `n` with `block_dim` blocks.
    - The `comb` sinkhorn operates per-row (one `i`) on a small 4x4 tile.
    """
    hc = 4
    mix_hc = 24

    pto = PTO("hc_split_sinkhorn4")
    pto.prologue()

    bid = pto.get_block_idx()
    bn = pto.get_block_num()

    mixes = pto.tensor(dtype="f32", shape=(n, mix_hc), role="in")
    hc_scale = pto.tensor(dtype="f32", shape=(1, 3), role="in")
    hc_base = pto.tensor(dtype="f32", shape=(1, mix_hc), role="in")

    pre_out = pto.tensor(dtype="f32", shape=(n, hc), role="out")
    post_out = pto.tensor(dtype="f32", shape=(n, hc), role="out")
    comb_out = pto.tensor(dtype="f32", shape=(n, hc * hc), role="out")

    # Scalar params (loaded once). Keep 32B alignment for Vec f32 tiles via cols=8.
    s0 = pto.vec(dtype="f32", shape=(1, 8), valid="1x1")
    s1 = pto.vec(dtype="f32", shape=(1, 8), valid="1x1")
    s2 = pto.vec(dtype="f32", shape=(1, 8), valid="1x1")
    s0 = pto.load(hc_scale, 0, 0)
    s1 = pto.load(hc_scale, 0, 1)
    s2 = pto.load(hc_scale, 0, 2)

    # Broadcasted scales.
    s0_row = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    s1_row = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    s2_row = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    s0_row = pto.rowexpand(s0)
    s1_row = pto.rowexpand(s1)
    s2_row = pto.rowexpand(s2)

    # Bases (loaded once).
    base_pre = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    base_post = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    base_pre = pto.load(hc_base, 0, 0)
    base_post = pto.load(hc_base, 0, 4)

    base_c0 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    base_c1 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    base_c2 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    base_c3 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    base_c0 = pto.load(hc_base, 0, 8)
    base_c1 = pto.load(hc_base, 0, 12)
    base_c2 = pto.load(hc_base, 0, 16)
    base_c3 = pto.load(hc_base, 0, 20)

    # Sigmoid temporaries.
    mix_pre = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    mix_post = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    exp_x = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    denom = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")

    # Sinkhorn row tiles (4x4 in 4x(8,valid=4) rows).
    r0 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    r1 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    r2 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    r3 = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")

    # Row-reduce scratch + scalars.
    tmp = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    row_max = pto.vec(dtype="f32", shape=(1, 8), valid="1x1")
    row_max_b = pto.vec(dtype="f32", shape=(1, 8), valid="1x8")
    row_sum = pto.vec(dtype="f32", shape=(1, 8), valid="1x1")
    row_sum_b = pto.vec(dtype="f32", shape=(1, 8), valid="1x8")
    col_sum = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")

    # Output tiles.
    pre_tile = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")
    post_tile = pto.vec(dtype="f32", shape=(1, 8), valid="1x4")

    for i in range(bid, n, bn):
        # --- pre ---
        mix_pre = pto.load(mixes, i, 0)
        mix_pre = pto.mul(mix_pre, s0_row)
        mix_pre = pto.add(mix_pre, base_pre)
        exp_x = pto.exp(mix_pre)
        denom = pto.adds(exp_x, 1.0)
        pre_tile = pto.div(exp_x, denom)
        pre_tile = pto.adds(pre_tile, eps)
        pto.store(pre_out, i, 0, pre_tile)

        # --- post ---
        mix_post = pto.load(mixes, i, 4)
        mix_post = pto.mul(mix_post, s1_row)
        mix_post = pto.add(mix_post, base_post)
        exp_x = pto.exp(mix_post)
        denom = pto.adds(exp_x, 1.0)
        post_tile = pto.div(exp_x, denom)
        post_tile = pto.muls(post_tile, 2.0)
        pto.store(post_out, i, 0, post_tile)

        # --- comb logits (4x4), expressed as 4 row tiles (1x4 each) ---
        r0 = pto.load(mixes, i, 8)
        r1 = pto.load(mixes, i, 12)
        r2 = pto.load(mixes, i, 16)
        r3 = pto.load(mixes, i, 20)

        r0 = pto.mul(r0, s2_row)
        r1 = pto.mul(r1, s2_row)
        r2 = pto.mul(r2, s2_row)
        r3 = pto.mul(r3, s2_row)
        r0 = pto.add(r0, base_c0)
        r1 = pto.add(r1, base_c1)
        r2 = pto.add(r2, base_c2)
        r3 = pto.add(r3, base_c3)

        # --- sinkhorn ---
        # Per-row softmax + eps: r = softmax(r) + eps
        row_max = pto.rowmax(r0, tmp)
        row_max_b = pto.rowexpand(row_max)
        r0 = pto.rowexpandsub(r0, row_max_b)
        r0 = pto.exp(r0)
        row_sum = pto.rowsum(r0, tmp)
        row_sum_b = pto.rowexpand(row_sum)
        r0 = pto.rowexpanddiv(r0, row_sum_b)
        r0 = pto.adds(r0, eps)

        row_max = pto.rowmax(r1, tmp)
        row_max_b = pto.rowexpand(row_max)
        r1 = pto.rowexpandsub(r1, row_max_b)
        r1 = pto.exp(r1)
        row_sum = pto.rowsum(r1, tmp)
        row_sum_b = pto.rowexpand(row_sum)
        r1 = pto.rowexpanddiv(r1, row_sum_b)
        r1 = pto.adds(r1, eps)

        row_max = pto.rowmax(r2, tmp)
        row_max_b = pto.rowexpand(row_max)
        r2 = pto.rowexpandsub(r2, row_max_b)
        r2 = pto.exp(r2)
        row_sum = pto.rowsum(r2, tmp)
        row_sum_b = pto.rowexpand(row_sum)
        r2 = pto.rowexpanddiv(r2, row_sum_b)
        r2 = pto.adds(r2, eps)

        row_max = pto.rowmax(r3, tmp)
        row_max_b = pto.rowexpand(row_max)
        r3 = pto.rowexpandsub(r3, row_max_b)
        r3 = pto.exp(r3)
        row_sum = pto.rowsum(r3, tmp)
        row_sum_b = pto.rowexpand(row_sum)
        r3 = pto.rowexpanddiv(r3, row_sum_b)
        r3 = pto.adds(r3, eps)

        # Column normalize: r = r / (col_sum + eps)
        col_sum = pto.add(r0, r1)
        col_sum = pto.add(col_sum, r2)
        col_sum = pto.add(col_sum, r3)
        col_sum = pto.adds(col_sum, eps)
        r0 = pto.div(r0, col_sum)
        r1 = pto.div(r1, col_sum)
        r2 = pto.div(r2, col_sum)
        r3 = pto.div(r3, col_sum)

        # Iterate sinkhorn_iters-1:
        for _ in range(0, sinkhorn_iters_m1):
            row_sum = pto.rowsum(r0, tmp)
            row_sum = pto.adds(row_sum, eps)
            row_sum_b = pto.rowexpand(row_sum)
            r0 = pto.rowexpanddiv(r0, row_sum_b)

            row_sum = pto.rowsum(r1, tmp)
            row_sum = pto.adds(row_sum, eps)
            row_sum_b = pto.rowexpand(row_sum)
            r1 = pto.rowexpanddiv(r1, row_sum_b)

            row_sum = pto.rowsum(r2, tmp)
            row_sum = pto.adds(row_sum, eps)
            row_sum_b = pto.rowexpand(row_sum)
            r2 = pto.rowexpanddiv(r2, row_sum_b)

            row_sum = pto.rowsum(r3, tmp)
            row_sum = pto.adds(row_sum, eps)
            row_sum_b = pto.rowexpand(row_sum)
            r3 = pto.rowexpanddiv(r3, row_sum_b)

            col_sum = pto.add(r0, r1)
            col_sum = pto.add(col_sum, r2)
            col_sum = pto.add(col_sum, r3)
            col_sum = pto.adds(col_sum, eps)
            r0 = pto.div(r0, col_sum)
            r1 = pto.div(r1, col_sum)
            r2 = pto.div(r2, col_sum)
            r3 = pto.div(r3, col_sum)

        pto.store(comb_out, i, 0, r0)
        pto.store(comb_out, i, 4, r1)
        pto.store(comb_out, i, 8, r2)
        pto.store(comb_out, i, 12, r3)

    pto.epilogue()
    return pto.program()


def make_hc_split_sinkhorn4_kernel(*, cfg: HcSplitSinkhorn4Config) -> KernelSpec:
    if cfg.n <= 0:
        raise ValueError("n must be > 0")
    if cfg.sinkhorn_iters <= 0:
        raise ValueError("sinkhorn_iters must be > 0")
    return compile_kernel_spec(
        hc_split_sinkhorn4,
        consts={
            "n": int(cfg.n),
            "sinkhorn_iters": int(cfg.sinkhorn_iters),
            "sinkhorn_iters_m1": int(cfg.sinkhorn_iters) - 1,
            "eps": float(cfg.eps),
        },
    )
