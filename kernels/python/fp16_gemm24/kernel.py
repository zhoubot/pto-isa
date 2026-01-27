from __future__ import annotations

from dataclasses import dataclass

from pto_as import PTO
from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec


@dataclass(frozen=True)
class Fp16Gemm24Config:
    m: int
    n: int
    k: int
    grid_m: int = 4
    grid_n: int = 6
    base_m: int = 128
    base_n: int = 256
    base_k: int = 64


def fp16_gemm24():
    """
    High-performance-ish GEMM for A3 cube:
      C[m,n] = A[m,k] @ B[k,n]

    Notes:
    - B is declared as a DN tensor (shape [k,n]) backed by a physical [n,k] row-major buffer on host (B^T contiguous).
    - Launch with block_dim=grid_m*grid_n (defaults to 24 to fill A3 cube).
    """
    pto = PTO("fp16_gemm24")
    pto.prologue()
    bid = pto.get_block_idx()

    A = pto.tensor(dtype="f16", shape=(m, k), role="in")
    B = pto.tensor(dtype="f16", shape=(k, n), stride=(1, k), layout="DN", role="in")
    C = pto.tensor(dtype="f32", shape=(m, n), role="out")

    a_mat0 = pto.mat(dtype="f16", shape=(base_m, base_k))
    a_mat1 = pto.mat(dtype="f16", shape=(base_m, base_k))
    b_mat0 = pto.mat(dtype="f16", shape=(base_k, base_n), blayout="RowMajor", slayout="ColMajor")
    b_mat1 = pto.mat(dtype="f16", shape=(base_k, base_n), blayout="RowMajor", slayout="ColMajor")

    a0 = pto.left(dtype="f16", shape=(base_m, base_k))
    a1 = pto.left(dtype="f16", shape=(base_m, base_k))
    b0 = pto.right(dtype="f16", shape=(base_k, base_n))
    b1 = pto.right(dtype="f16", shape=(base_k, base_n))
    c = pto.acc(dtype="f32", shape=(base_m, base_n))

    single_core_m = m // grid_m
    single_core_n = n // grid_n
    m_loop = single_core_m // base_m
    n_loop = single_core_n // base_n
    k_tiles = k // base_k

    m_core = bid % grid_m
    n_core = bid // grid_m

    if n_core < grid_n:
        base_m0 = m_core * single_core_m
        base_n0 = n_core * single_core_n

        for mi in range(0, m_loop):
            mi_off = mi * base_m
            m0 = base_m0 + mi_off
            for nj in range(0, n_loop):
                nj_off = nj * base_n
                n0 = base_n0 + nj_off

                for kt in range(0, k_tiles):
                    k0 = kt * base_k
                    lane = kt % 2
                    if lane == 0:
                        a_mat0 = pto.load(A, m0, k0)
                        b_mat0 = pto.load(B, k0, n0)
                        a0 = pto.mov(a_mat0)
                        b0 = pto.mov(b_mat0)
                        if kt == 0:
                            c = pto.matmul(a0, b0)
                        else:
                            c = pto.matmul_acc(c, a0, b0)
                    else:
                        a_mat1 = pto.load(A, m0, k0)
                        b_mat1 = pto.load(B, k0, n0)
                        a1 = pto.mov(a_mat1)
                        b1 = pto.mov(b_mat1)
                        c = pto.matmul_acc(c, a1, b1)

                pto.store(C, m0, n0, c)

    pto.epilogue()
    return pto.program()


def make_fp16_gemm24_kernel(*, cfg: Fp16Gemm24Config) -> KernelSpec:
    if cfg.grid_m <= 0 or cfg.grid_n <= 0:
        raise ValueError("grid_m/grid_n must be > 0")
    if cfg.base_m <= 0 or cfg.base_n <= 0 or cfg.base_k <= 0:
        raise ValueError("base_m/base_n/base_k must be > 0")
    if (cfg.m % (cfg.grid_m * cfg.base_m)) != 0 or (cfg.n % (cfg.grid_n * cfg.base_n)) != 0 or (cfg.k % cfg.base_k) != 0:
        raise ValueError("shape must be divisible by grid/base tiles")
    return compile_kernel_spec(
        fp16_gemm24,
        consts={
            "m": int(cfg.m),
            "n": int(cfg.n),
            "k": int(cfg.k),
            "grid_m": int(cfg.grid_m),
            "grid_n": int(cfg.grid_n),
            "base_m": int(cfg.base_m),
            "base_n": int(cfg.base_n),
            "base_k": int(cfg.base_k),
        },
    )
