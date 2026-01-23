from __future__ import annotations

from dataclasses import dataclass

from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec_from_source


@dataclass(frozen=True)
class GemmConfig:
    m: int
    n: int
    k: int
    bm: int = 128
    bn: int = 128
    bk: int = 64


def make_gemm_f16f16f32_kernel(*, cfg: GemmConfig) -> KernelSpec:
    if cfg.m % cfg.bm != 0 or cfg.n % cfg.bn != 0 or cfg.k % cfg.bk != 0:
        raise ValueError("m,n,k must be multiples of bm,bn,bk")

    tiles_m = cfg.m // cfg.bm
    tiles_n = cfg.n // cfg.bn

    # Notes:
    # - We use ping-pong Left/Right tiles (`*_0`/`*_1`) to avoid overwriting L0A/L0B while the cube pipe is active.
    # - Work partition: each block computes one [bm x bn] output tile. blockDim should be tiles_m * tiles_n.
    src = f"""
def gemm_f16f16f32():
    prologue()
    bid = get_block_idx()

    A = tensor(dtype="f16", shape=({cfg.m}, {cfg.k}), role="in")
    # Represent B as a DN tensor backed by a physical [n, k] row-major buffer on host.
    # This matches the manual A2/A3 GEMM kernels which typically load B in DN/ZN form.
    B = tensor(dtype="f16", shape=({cfg.k}, {cfg.n}), stride=(1, {cfg.k}), layout="DN", role="in")
    C = tensor(dtype="f32", shape=({cfg.m}, {cfg.n}), role="out")

    # Mat tiles (loaded from GM). Shape matches Left/Right so we can use TMOV (no TEXTRACT here).
    a_mat = tile(loc="Mat", dtype="f16", rows={cfg.bm}, cols={cfg.bk}, blayout="ColMajor", slayout="RowMajor")
    # For DN global tensors, use a ZN Mat tile (DN->ZN is supported by TLOAD(MatTile,...)).
    b_mat = tile(loc="Mat", dtype="f16", rows={cfg.bk}, cols={cfg.bn}, blayout="RowMajor", slayout="ColMajor")

    # Matmul operands / accumulator.
    a_left_0 = tile(loc="Left", dtype="f16", rows={cfg.bm}, cols={cfg.bk}, blayout="RowMajor", slayout="RowMajor")
    a_left_1 = tile(loc="Left", dtype="f16", rows={cfg.bm}, cols={cfg.bk}, blayout="RowMajor", slayout="RowMajor")
    b_right_0 = tile(loc="Right", dtype="f16", rows={cfg.bk}, cols={cfg.bn}, blayout="RowMajor", slayout="ColMajor")
    b_right_1 = tile(loc="Right", dtype="f16", rows={cfg.bk}, cols={cfg.bn}, blayout="RowMajor", slayout="ColMajor")
    c_acc = tile(loc="Acc", dtype="f32", rows={cfg.bm}, cols={cfg.bn}, blayout="ColMajor", slayout="RowMajor")

    tiles_m = {tiles_m}
    tiles_n = {tiles_n}

    m_idx = bid % tiles_m
    n_idx = bid // tiles_m

    if n_idx < tiles_n:
        m0 = m_idx * {cfg.bm}
        n0 = n_idx * {cfg.bn}

        for k0 in range(0, {cfg.k}, {cfg.bk}):
            tload(a_mat, A, m0, k0)
            tload(b_mat, B, k0, n0)
            it0 = k0 // {cfg.bk}
            lane = it0 % 2
            if lane == 0:
                tmov(a_left_0, a_mat)
                tmov(b_right_0, b_mat)
                if k0 == 0:
                    tmatmul(c_acc, a_left_0, b_right_0)
                else:
                    tmatmul_acc(c_acc, c_acc, a_left_0, b_right_0)
            else:
                tmov(a_left_1, a_mat)
                tmov(b_right_1, b_mat)
                tmatmul_acc(c_acc, c_acc, a_left_1, b_right_1)
        tstore(C, m0, n0, c_acc)

    epilogue()
"""
    return compile_kernel_spec_from_source(src, func_name="gemm_f16f16f32")
