# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
import torch_npu

from jit_util_gemm import jit_compile_gemm


# GEMM FLOPs: 2 * m * n * k (each of m*n outputs does k muls + k adds)
def gemm_tflops(m: int, n: int, k: int, elapsed_ms: float) -> float:
    flops = 2 * m * n * k
    return (flops / 1e12) / (elapsed_ms / 1000.0)


def run_timed_npu(n_repeat: int, run_one):
    """Run run_one() n_repeat times and return total elapsed ms using NPU events."""
    start_ev = torch.npu.Event(enable_timing=True)
    end_ev = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    start_ev.record()
    for _ in range(n_repeat):
        run_one()
    end_ev.record()
    torch.npu.synchronize()
    return start_ev.elapsed_time(end_ev)


def test_gemm(n_repeat=50):
    # NOTE: shapes are currently hard-coded as in gemm_kernel.cpp
    m = 6144
    n = 6144
    k = 6144

    device = "npu"
    dtype = torch.float16
    out_dtype = torch.float32

    a = torch.rand((m, k), device=device, dtype=dtype)
    b = torch.rand((k, n), device=device, dtype=dtype)
    c = torch.empty((m, n), device=device, dtype=out_dtype)

    gemm_func = jit_compile_gemm(verbose=False)

    # Correctness check (single run)
    gemm_func(c, a, b)
    torch.npu.synchronize()
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c.to(torch.float16), c_ref, rtol=0.1, atol=1e-5)
    print("GEMM test pass!")

    # Benchmark: custom GEMM kernel (float16 input and float32 accumulator)
    # NOTE: `blockDim` in gemm_kernel.cpp is hard-coded to 24 (910B2)
    elapsed_custom_ms = run_timed_npu(n_repeat, lambda: gemm_func(c, a, b))
    tflops_custom = gemm_tflops(m, n, k, elapsed_custom_ms / n_repeat)

    # Benchmark: torch built-in matmul (float16 input and output)
    elapsed_torch_ms = run_timed_npu(n_repeat, lambda: torch.matmul(a, b))
    tflops_torch = gemm_tflops(m, n, k, elapsed_torch_ms / n_repeat)

    print(
        f"Custom GEMM kernel: {elapsed_custom_ms:.2f} ms / {n_repeat} runs "
        f"({elapsed_custom_ms / n_repeat:.3f} ms/iter) -> {tflops_custom:.3f} TFLOPs"
    )

    print(
        f"Torch matmul (float16): {elapsed_torch_ms:.2f} ms / {n_repeat} runs "
        f"({elapsed_torch_ms / n_repeat:.3f} ms/iter) -> {tflops_torch:.3f} TFLOPs"
    )


if __name__ == "__main__":
    test_gemm()
