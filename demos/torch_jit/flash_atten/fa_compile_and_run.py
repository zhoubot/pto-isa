#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import random
import torch
import torch_npu
from jit_util_flash import jit_compile_flash

NUM_ITERATIONS = 50
WARMUP = 10
SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.npu.manual_seed(SEED)


def prep_inputs_for_fused(q2d, k2d, v2d):
    """
    Make inputs match default layout BSH: (B,S,H).
    """
    # q2d: (S0,H) -> (1,S0,H)
    q = q2d.unsqueeze(0)

    # k2d: (S1,H) -> (1,S1,H)
    k = k2d.unsqueeze(0)

    # v2d: (S1,H) -> (1,S1,H)
    v = v2d.unsqueeze(0)

    return q, k, v


def fused_only(q_bsh, k_bsh, v_bsh):
    out, _ = torch_npu.npu_fused_infer_attention_score(q_bsh, k_bsh, v_bsh)
    return out  # (B,S,H)


def time_op_npu(fn):
    """
    Accurate device timing:
    - warmup to stabilize
    - synchronize around measurement
    - measure average per-iter ms
    """
    for _ in range(WARMUP):
        _ = fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITERATIONS):
        _ = fn()
    end.record()
    torch.npu.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / NUM_ITERATIONS


def test_flash():
    s0, s1, head = 256, 2048, 128
    num_tiles = s0 // 128

    device = "npu"
    torch.npu.set_device(device)

    dtype = torch.float16
    out_dtype = torch.float32

    # ==========================
    # Inputs
    # ==========================
    q2d = torch.rand((s0, head), device=device, dtype=dtype)
    k2d = torch.rand((s1, head), device=device, dtype=dtype)
    v2d = torch.rand((s1, head), device=device, dtype=dtype)

    # ==========================
    # Reference fused inputs
    # ==========================
    q_bsh, k_bsh, v_bsh = prep_inputs_for_fused(q2d, k2d, v2d)

    # ==========================
    # Flash kernel buffers
    # ==========================
    o_out = torch.empty((s0, head), device=device, dtype=out_dtype)

    out_device = torch.empty((s0, s1), device=device, dtype=torch.float32)
    xexp_device = torch.empty((s0, s1), device=device, dtype=torch.float16)
    pout_fp32_device = torch.empty((s0, s1), device=device, dtype=torch.float32)

    out_2d_device = torch.empty(
        (num_tiles, s0, head), device=device, dtype=torch.float32
    )
    g_sum_device = torch.empty((num_tiles, s0), device=device, dtype=torch.float32)
    exp_max_device = torch.empty((num_tiles, s0), device=device, dtype=torch.float32)
    o_parts_device = torch.empty(
        (num_tiles, s0, head), device=device, dtype=torch.float32
    )

    # ==========================
    # Compile flash ONCE
    # ==========================
    flash = jit_compile_flash(verbose=False)

    # ==========================
    # Benchmark ONLY fused op
    # ==========================
    fused_ms = time_op_npu(lambda: fused_only(q_bsh, k_bsh, v_bsh))

    # ==========================
    # Benchmark flash kernel call
    # ==========================
    flash_ms = time_op_npu(
        lambda: flash(
            q2d,
            k2d,
            v2d,
            o_out,
            out_device,
            xexp_device,
            pout_fp32_device,
            out_2d_device,
            g_sum_device,
            exp_max_device,
            o_parts_device,
        )
    )

    # ==========================
    # Correctness check
    # ==========================

    o_ref_bsh = fused_only(q_bsh, k_bsh, v_bsh)
    o_ref = o_ref_bsh.squeeze(0).to(torch.float32)

    print(f"FlashAttention kernel time : {flash_ms:.3f} ms/iter")
    print(f"Fused attention time       : {fused_ms:.3f} ms/iter")
    print(f"Speedup                    : {fused_ms / flash_ms:.2f}×")

    torch.testing.assert_close(o_out, o_ref, rtol=0.3, atol=1e-1)
    print("FlashAttention test passed!")


if __name__ == "__main__":
    test_flash()
