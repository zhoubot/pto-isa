from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from pto_as import PTO
from ptoas.python.ast_frontend import KernelSpec, compile_kernel_spec


@dataclass(frozen=True)
class SparseAttnTopKSplitConfig:
    q: int  # number of query positions (flattened b*m)
    h: int
    d: int
    topk: int
    tile: int = 16


def sparse_attn_topk_split():
    """
    Sparse-attn compute kernel (TileLang -> pyPTO port), with the gather step moved out of the kernel.

    This kernel assumes K/V are already gathered into per-query contiguous buffers:
      - q:   [q*h, d]   f16
      - kt:  [q*d, topk] f16   (per-query K^T contiguous)
      - v:   [q*topk, d] f16   (per-query V contiguous)

    It is split into three stages (cube/vec/cube) via `pto.stage_*()` markers and
    should be compiled with `ptoas --split-kernels`:
      1) scores = Q @ K^T  (cube) -> scores_gm f32
      2) probs  = softmax(scores * scale) (vec) -> probs_gm f16
      3) out    = probs @ V (cube) -> out f32

    Note:
    - The original TileLang kernel gathers `kv` from a large KV cache using `topk_idxs`.
      A3's current MGATHER prototype is limited by a small tmp UB buffer, so we model the
      gather outside the kernel and focus on the compute pipeline.
    """
    TS = 16

    pto = PTO("sparse_attn_topk_split")
    pto.prologue()

    bid = pto.get_block_idx()
    bn = pto.get_block_num()

    q_in = pto.tensor(dtype="f16", shape=(q * h, d), role="in")          # [q*h, d]
    kt_in = pto.tensor(dtype="f16", shape=(q * d, topk), role="in")      # [q*d, topk]
    v_in = pto.tensor(dtype="f16", shape=(q * topk, d), role="in")       # [q*topk, d]

    # Workspaces (role="in" so regression compares only the final `out`).
    scores_gm = pto.tensor(dtype="f32", shape=(q * h, topk), role="in")  # [q*h, topk]
    probs_gm = pto.tensor(dtype="f16", shape=(q * h, topk), role="in")   # [q*h, topk]

    out = pto.tensor(dtype="f32", shape=(q * h, d), role="out")          # [q*h, d]

    # --- Tiles: QK stage (cube) ---
    q_mat = pto.mat(dtype="f16", shape=(TS, TS))
    kt_mat = pto.mat(dtype="f16", shape=(TS, TS))
    q_left = pto.left(dtype="f16", shape=(TS, TS), blayout="ColMajor", slayout="RowMajor")
    kt_right = pto.right(dtype="f16", shape=(TS, TS))
    scores_acc = pto.acc(dtype="f32", shape=(TS, TS))

    # --- Tiles: softmax stage (vec) ---
    scores_tile = pto.vec(dtype="f32", shape=(TS, TS))
    tmp = pto.vec(dtype="f32", shape=(TS, TS))
    row_max = pto.vec_tile(dtype="f32", shape=(TS, 1), blayout="ColMajor")
    row_sum = pto.vec_tile(dtype="f32", shape=(TS, 1), blayout="ColMajor")
    max_bcast = pto.vec(dtype="f32", shape=(TS, TS))
    sum_bcast = pto.vec(dtype="f32", shape=(TS, TS))
    max_acc = pto.vec(dtype="f32", shape=(TS, TS))
    sum_acc = pto.vec(dtype="f32", shape=(TS, TS))
    centered = pto.vec(dtype="f32", shape=(TS, TS))
    exp_scores = pto.vec(dtype="f32", shape=(TS, TS))
    probs_f32 = pto.vec(dtype="f32", shape=(TS, TS))
    probs_f16 = pto.vec(dtype="f16", shape=(TS, TS))

    scale = pto.const("scale", 1.0 / sqrt(d), scalar("f32"))

    # --- Tiles: PV stage (cube) ---
    p_mat = pto.mat(dtype="f16", shape=(TS, TS))
    v_mat = pto.mat(dtype="f16", shape=(TS, TS))
    p_left = pto.left(dtype="f16", shape=(TS, TS), blayout="ColMajor", slayout="RowMajor")
    v_right = pto.right(dtype="f16", shape=(TS, TS))
    out_acc = pto.acc(dtype="f32", shape=(TS, TS))

    # -------------------------------------------------------------------------
    # Stage 1: scores = Q @ K^T (cube)
    # -------------------------------------------------------------------------
    pto.stage_qk_cube()
    for qid in range(bid, q, bn):
        q_row0 = qid * h
        kt_row0 = qid * d
        for hi in range(0, h, TS):
            qh0 = q_row0 + hi
            for kj in range(0, topk, TS):
                for kk in range(0, d, TS):
                    q_mat = pto.load(q_in, qh0, kk)
                    kt0 = kt_row0 + kk
                    kt_mat = pto.load(kt_in, kt0, kj)
                    # Loop-carried hazard: TMATMUL may still be reading the previous
                    # q_left/kt_right tiles (L0A/L0B) while we overwrite them via TMOV (MTE1).
                    pto.record_event(src_op="TMATMUL", dst_op="TMOV_M2L", token=4)
                    pto.wait_event(src_op="TMATMUL", dst_op="TMOV_M2L", token=4)
                    q_left = pto.mov(q_mat)
                    kt_right = pto.mov(kt_mat)
                    if kk == 0:
                        scores_acc = pto.matmul(q_left, kt_right)
                    else:
                        scores_acc = pto.matmul_acc(scores_acc, q_left, kt_right)
                # Ensure the final TMATMUL completes before storing the Acc tile to GM.
                pto.record_event(src_op="TMATMUL", dst_op="TSTORE_ACC", token=0)
                pto.wait_event(src_op="TMATMUL", dst_op="TSTORE_ACC", token=0)
                pto.store(scores_gm, qh0, kj, scores_acc)

    # -------------------------------------------------------------------------
    # Stage 2: probs = softmax(scores * scale) (vec)
    # -------------------------------------------------------------------------
    pto.stage_softmax_vec()
    for qid in range(bid, q, bn):
        q_row0 = qid * h
        for hi in range(0, h, TS):
            qh0 = q_row0 + hi

            # Pass 1: row_max across all column tiles.
            for kj in range(0, topk, TS):
                scores_tile = pto.load(scores_gm, qh0, kj)
                scores_tile = pto.muls(scores_tile, scale)
                row_max = pto.rowmax(scores_tile, tmp)
                max_bcast = pto.rowexpand(row_max)
                if kj == 0:
                    max_acc = pto.mov(max_bcast)
                else:
                    max_acc = pto.max(max_acc, max_bcast)

            # Pass 2: row_sum across all column tiles.
            for kj in range(0, topk, TS):
                scores_tile = pto.load(scores_gm, qh0, kj)
                scores_tile = pto.muls(scores_tile, scale)
                centered = pto.sub(scores_tile, max_acc)
                exp_scores = pto.exp(centered)
                row_sum = pto.rowsum(exp_scores, tmp)
                sum_bcast = pto.rowexpand(row_sum)
                if kj == 0:
                    sum_acc = pto.mov(sum_bcast)
                else:
                    sum_acc = pto.add(sum_acc, sum_bcast)

            # Pass 3: probs tile-by-tile.
            for kj in range(0, topk, TS):
                scores_tile = pto.load(scores_gm, qh0, kj)
                scores_tile = pto.muls(scores_tile, scale)
                centered = pto.sub(scores_tile, max_acc)
                exp_scores = pto.exp(centered)
                probs_f32 = pto.div(exp_scores, sum_acc)
                probs_f16 = pto.cvt(probs_f32, RoundMode.CAST_ROUND)
                pto.store(probs_gm, qh0, kj, probs_f16)

    # -------------------------------------------------------------------------
    # Stage 3: out = probs @ V (cube)
    # -------------------------------------------------------------------------
    pto.stage_pv_cube()
    for qid in range(bid, q, bn):
        q_row0 = qid * h
        v_row0 = qid * topk
        for hi in range(0, h, TS):
            qh0 = q_row0 + hi
            for dj in range(0, d, TS):
                for kk in range(0, topk, TS):
                    p_mat = pto.load(probs_gm, qh0, kk)
                    v0 = v_row0 + kk
                    v_mat = pto.load(v_in, v0, dj)
                    # Loop-carried hazard: TMATMUL may still be reading p_left/v_right while
                    # we overwrite them via TMOV (MTE1).
                    pto.record_event(src_op="TMATMUL", dst_op="TMOV_M2L", token=5)
                    pto.wait_event(src_op="TMATMUL", dst_op="TMOV_M2L", token=5)
                    p_left = pto.mov(p_mat)
                    v_right = pto.mov(v_mat)
                    if kk == 0:
                        out_acc = pto.matmul(p_left, v_right)
                    else:
                        out_acc = pto.matmul_acc(out_acc, p_left, v_right)
                pto.record_event(src_op="TMATMUL", dst_op="TSTORE_ACC", token=1)
                pto.wait_event(src_op="TMATMUL", dst_op="TSTORE_ACC", token=1)
                pto.store(out, qh0, dj, out_acc)

    pto.epilogue()
    return pto.program()


def make_sparse_attn_topk_split_kernel(*, cfg: SparseAttnTopKSplitConfig) -> KernelSpec:
    if cfg.tile != 16:
        raise ValueError("tile must be 16 for A3 cube/vec compatibility")
    for name, v in (("q", cfg.q), ("h", cfg.h), ("d", cfg.d), ("topk", cfg.topk)):
        if v <= 0 or (v % 16) != 0:
            raise ValueError(f"{name} must be >0 and divisible by 16 (got {v})")
    return compile_kernel_spec(
        sparse_attn_topk_split,
        consts={
            "q": int(cfg.q),
            "h": int(cfg.h),
            "d": int(cfg.d),
            "topk": int(cfg.topk),
            "tile": int(cfg.tile),
        },
    )
