from __future__ import annotations

from dataclasses import dataclass

from pto_as import PTO
from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec


@dataclass(frozen=True)
class ActQuant128Config:
    m: int
    n: int
    block_m: int = 32
    group_size: int = 128


def act_quant128():
    """
    A3-friendly activation quant kernel (TileLang -> pyPTO port).

    Differences vs the TileLang reference:
    - Uses f16 for input/output (A3 TMATMUL does not support FP8).
    - Keeps per-row scale in f32 (shape [m, n/group_size]).

    Math (per (row, group)):
      amax = max(abs(x[row, group]), 1e-4)
      scale = amax / 448
      y = clamp(x / scale, -448, 448)
    """
    pto = PTO("act_quant128")
    pto.prologue()

    fp8_min = pto.const("fp8_min", -448.0, scalar("f32"))
    fp8_max = pto.const("fp8_max", 448.0, scalar("f32"))
    fp8_max_inv = pto.const("fp8_max_inv", 1.0 / 448.0, scalar("f32"))
    eps = pto.const("eps", 1.0e-4, scalar("f32"))

    bid = pto.get_block_idx()
    bn = pto.get_block_num()

    X = pto.tensor(dtype="f16", shape=(m, n), role="in")
    Y = pto.tensor(dtype="f16", shape=(m, n), role="out")
    # NOTE: Keep f32 GM tensors 32B-aligned in the inner dim. Store the per-row scale
    # as 8 duplicated f32 values per (row, group): shape [m, (n/group_size)*8].
    S = pto.tensor(dtype="f32", shape=(m, (n // group_size) * 8), role="out")

    # NOTE: On A3, some vector reduce/broadcast paths become fragile at wider tiles.
    # This kernel targets group_size=128 and processes it as two 64-wide halves.
    split_k = 64

    x0_f16 = pto.vec(dtype="f16", shape=(block_m, split_k))
    x1_f16 = pto.vec(dtype="f16", shape=(block_m, split_k))
    x0_f32 = pto.vec(dtype="f32", shape=(block_m, split_k))
    x1_f32 = pto.vec(dtype="f32", shape=(block_m, split_k))
    x0_abs = pto.vec(dtype="f32", shape=(block_m, split_k))
    x1_abs = pto.vec(dtype="f32", shape=(block_m, split_k))
    tmp0 = pto.vec(dtype="f32", shape=(block_m, split_k))
    tmp1 = pto.vec(dtype="f32", shape=(block_m, split_k))
    # Keep per-row scalars as a proper [M, 1] column-major Vec tile (row-reduce
    # output), then broadcast to one 32B block per row for vector binops.
    amax0 = pto.vec(dtype="f32", shape=(block_m, 1), blayout="ColMajor")
    amax1 = pto.vec(dtype="f32", shape=(block_m, 1), blayout="ColMajor")
    amax0_b = pto.vec(dtype="f32", shape=(block_m, 8))
    amax1_b = pto.vec(dtype="f32", shape=(block_m, 8))
    amax_b = pto.vec(dtype="f32", shape=(block_m, 8))
    scale = pto.vec(dtype="f32", shape=(block_m, 8))
    y0_f16 = pto.vec(dtype="f16", shape=(block_m, split_k))
    y1_f16 = pto.vec(dtype="f16", shape=(block_m, split_k))

    tiles_m = m // block_m
    tiles_n = n // group_size
    total_tiles = tiles_m * tiles_n

    for tid in range(bid, total_tiles, bn):
        tm = tid % tiles_m
        tn = tid // tiles_m
        m0 = tm * block_m
        n0 = tn * group_size

        n1 = n0 + split_k

        x0_f16 = pto.load(X, m0, n0)
        x0_f32 = pto.cvt(x0_f16, RoundMode.CAST_NONE)
        x0_abs = pto.abs(x0_f32)
        amax0 = pto.rowmax(x0_abs, tmp0)
        amax0_b = pto.rowexpand(amax0)

        x1_f16 = pto.load(X, m0, n1)
        x1_f32 = pto.cvt(x1_f16, RoundMode.CAST_NONE)
        x1_abs = pto.abs(x1_f32)
        amax1 = pto.rowmax(x1_abs, tmp1)
        amax1_b = pto.rowexpand(amax1)

        amax_b = pto.max(amax0_b, amax1_b)
        amax_b = pto.maxs(amax_b, eps)
        scale = pto.muls(amax_b, fp8_max_inv)
        s_col = tn * 8
        pto.store(S, m0, s_col, scale)

        # Reload X for the actual quantize+store step. This avoids keeping large
        # Vec tiles live across the amax reduction and prevents UB aliasing bugs
        # with in-place rowexpanddiv on some A3 toolchains.
        x0_f16 = pto.load(X, m0, n0)
        x0_f32 = pto.cvt(x0_f16, RoundMode.CAST_NONE)
        x0_f32 = pto.rowexpanddiv(x0_f32, scale)
        x0_f32 = pto.maxs(x0_f32, fp8_min)
        x0_f32 = pto.mins(x0_f32, fp8_max)
        y0_f16 = pto.cvt(x0_f32, RoundMode.CAST_ROUND)
        pto.store(Y, m0, n0, y0_f16)

        x1_f16 = pto.load(X, m0, n1)
        x1_f32 = pto.cvt(x1_f16, RoundMode.CAST_NONE)
        x1_f32 = pto.rowexpanddiv(x1_f32, scale)
        x1_f32 = pto.maxs(x1_f32, fp8_min)
        x1_f32 = pto.mins(x1_f32, fp8_max)
        y1_f16 = pto.cvt(x1_f32, RoundMode.CAST_ROUND)
        pto.store(Y, m0, n1, y1_f16)

    pto.epilogue()
    return pto.program()


def make_act_quant128_kernel(*, cfg: ActQuant128Config) -> KernelSpec:
    if cfg.block_m <= 0 or cfg.group_size <= 0:
        raise ValueError("block_m/group_size must be > 0")
    if cfg.block_m != 32:
        raise ValueError("act_quant128 currently targets block_m=32 only")
    if cfg.m % cfg.block_m != 0:
        raise ValueError(f"m must be divisible by block_m (m={cfg.m}, block_m={cfg.block_m})")
    if cfg.group_size != 128:
        raise ValueError("act_quant128 currently targets group_size=128 only")
    if cfg.n % cfg.group_size != 0:
        raise ValueError(f"n must be divisible by group_size (n={cfg.n}, group_size={cfg.group_size})")
    return compile_kernel_spec(
        act_quant128,
        consts={
            "m": int(cfg.m),
            "n": int(cfg.n),
            "block_m": int(cfg.block_m),
            "group_size": int(cfg.group_size),
        },
    )
