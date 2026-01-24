from __future__ import annotations

from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec_from_source


def make_gemm_performance_kernel(*, m: int = 6144, k: int = 6144, n: int = 6144) -> KernelSpec:
    """
    High-performance-ish GEMM for A3 cube (fp16 inputs -> fp32 output).

    Design goals:
    - Match the manual GEMM performance example's launch geometry: 24 blocks (4x6 split).
    - Keep the kernel simple and let `ptoas --insert-events` insert the necessary set/wait flags.
    - Use device-side timing + sampled numpy validation in the runner.

    Layout:
    - A: ND [m, k] fp16
    - B: DN [k, n] fp16 backed by a physical [n, k] row-major buffer on host (i.e. B^T contiguous)
    - C: ND [m, n] fp32
    """
    base_m, base_k, base_n = 128, 64, 256
    grid_m, grid_n = 4, 6
    block_dim = grid_m * grid_n
    if (m % (grid_m * base_m)) != 0 or (n % (grid_n * base_n)) != 0 or (k % base_k) != 0:
        raise ValueError("shape must be divisible by grid/base tiles (m%512==0, n%1536==0, k%64==0 for defaults)")

    single_core_m = m // grid_m
    single_core_n = n // grid_n
    m_loop = single_core_m // base_m
    n_loop = single_core_n // base_n
    k_tiles = k // base_k

    src = f"""
def gemm_performance():
    prologue()
    bid = get_block_idx()

    A = tensor(dtype="f16", shape=({m}, {k}), role="in")
    # DN tensor backed by a physical [n, k] row-major buffer (host passes B^T contiguous).
    B = tensor(dtype="f16", shape=({k}, {n}), stride=(1, {k}), layout="DN", role="in")
    C = tensor(dtype="f32", shape=({m}, {n}), role="out")

    # Scalar params kept as SSA values (the AST frontend only supports simple BinOps).
    base_m = {base_m}
    base_k = {base_k}
    base_n = {base_n}

    # Double-buffer GM->L1 (Mat) and L1->L0 (Left/Right) to enable overlap across pipes.
    a_mat0 = tile(loc="Mat", dtype="f16", rows={base_m}, cols={base_k}, blayout="ColMajor", slayout="RowMajor")
    a_mat1 = tile(loc="Mat", dtype="f16", rows={base_m}, cols={base_k}, blayout="ColMajor", slayout="RowMajor")
    b_mat0 = tile(loc="Mat", dtype="f16", rows={base_k}, cols={base_n}, blayout="RowMajor", slayout="ColMajor")
    b_mat1 = tile(loc="Mat", dtype="f16", rows={base_k}, cols={base_n}, blayout="RowMajor", slayout="ColMajor")

    a0 = tile(loc="Left", dtype="f16", rows={base_m}, cols={base_k}, blayout="RowMajor", slayout="RowMajor")
    a1 = tile(loc="Left", dtype="f16", rows={base_m}, cols={base_k}, blayout="RowMajor", slayout="RowMajor")
    b0 = tile(loc="Right", dtype="f16", rows={base_k}, cols={base_n}, blayout="RowMajor", slayout="ColMajor")
    b1 = tile(loc="Right", dtype="f16", rows={base_k}, cols={base_n}, blayout="RowMajor", slayout="ColMajor")

    c = tile(loc="Acc", dtype="f32", rows={base_m}, cols={base_n}, blayout="ColMajor", slayout="RowMajor")

    grid_m = {grid_m}
    grid_n = {grid_n}
    m_core = bid % grid_m
    n_core = bid // grid_m

    if n_core < grid_n:
        base_m0 = m_core * {single_core_m}
        base_n0 = n_core * {single_core_n}

        for mi in range(0, {m_loop}):
            mi_off = mi * base_m
            m0 = base_m0 + mi_off
            for nj in range(0, {n_loop}):
                nj_off = nj * base_n
                n0 = base_n0 + nj_off

                # Prime buffer 0 (k0=0).
                tload(a_mat0, A, m0, 0)
                tload(b_mat0, B, 0, n0)
                tmov(a0, a_mat0)
                tmov(b0, b_mat0)

                # Also prime buffer 1 (k0=base_k) so event insertion can conservatively
                # synchronize both ping-pong buffers through control-flow merges.
                if {k_tiles} > 1:
                    tload(a_mat1, A, m0, base_k)
                    tload(b_mat1, B, base_k, n0)
                    tmov(a1, a_mat1)
                    tmov(b1, b_mat1)

                # Main K loop with software pipelining:
                # - Prefetch (k+1) into the other buffer
                # - Compute current buffer
                for kt in range(0, {k_tiles}):
                    k_next_t = kt + 1
                    if k_next_t < {k_tiles}:
                        k1 = k_next_t * base_k
                        lane = kt % 2
                        if lane == 0:
                            tload(a_mat1, A, m0, k1)
                            tload(b_mat1, B, k1, n0)
                            tmov(a1, a_mat1)
                            tmov(b1, b_mat1)
                        else:
                            tload(a_mat0, A, m0, k1)
                            tload(b_mat0, B, k1, n0)
                            tmov(a0, a_mat0)
                            tmov(b0, b_mat0)

                    lane = kt % 2
                    if lane == 0:
                        if kt == 0:
                            tmatmul(c, a0, b0)
                        else:
                            tmatmul_acc(c, c, a0, b0)
                    else:
                        tmatmul_acc(c, c, a1, b1)

                tstore(C, m0, n0, c)

    epilogue()
"""
    # Kernel name is inferred from the Python function name ("gemm_performance").
    # If you need to override it, the AST frontend supports `pto = PTO("name")`
    # inside the kernel body, which sets an explicit kernel name.
    return compile_kernel_spec_from_source(src, func_name="gemm_performance")
