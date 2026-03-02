#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import op_extension
import numpy as np

S0_BASE = 64
HEAD_SIZE = 128
TILE_S1_DEFAULT = 256

def gen_case(s0, s1, head_size=HEAD_SIZE, cube_s1=128, tile_s1=TILE_S1_DEFAULT, is_causal=False):
    # generate inputs in FP16, compute golden in FP32
    q_fp32 = (np.random.randn(s0, head_size).astype(np.float16) * 1.5).astype(np.float32)
    k_fp32 = (np.random.randn(head_size, s1).astype(np.float16) * 1.5).astype(np.float32)
    q = q_fp32.astype(np.float16)
    k = k_fp32.astype(np.float16)
    golden = (q.astype(np.float32).dot(k.astype(np.float32))).astype(np.float32)

    assert s1 % tile_s1 == 0, "S1 must be divisible by TILE_S1"
    assert tile_s1 % cube_s1 == 0, "TILE_S1 must be divisible by CUBE_S1"

    # write FP16 inputs and FP32 golden
    #q.tofile(os.path.join(path, 'q.bin'))
    q_torch = torch.from_numpy(q)
    #k.tofile(os.path.join(path, 'k.bin'))
    k_torch = torch.from_numpy(k)
    kt = k.T.astype(np.float16)    
    #kt.tofile(os.path.join(path, 'kt.bin')) 
    kt_torch = torch.from_numpy(kt.copy())
    #golden.tofile(os.path.join(path, 'qk.bin'))
    qk_torch = torch.from_numpy(golden)
    # also produce softmax x_exp (per-row) saved as FP16 and tmp_float_exp saved as FP32
    # compute softmax in tiled fashion by TILE_S1 tiles (default 256)
    arr_f32 = golden.astype(np.float32)
    if is_causal:
        mask = np.triu((np.ones(arr_f32.shape) * float(-3.40282e+38)).astype(np.float32), 1)
        arr_f32 += mask
    scale = 1/np.sqrt(head_size)
    num_tiles = s1 // tile_s1

    # allocate full arrays to collect per-tile exponentials and per-tile global sums
    full_exp = np.zeros((s0, s1), dtype=np.float32)
    global_sums = []
    exp_max_parts = []

    # emulate TSOFTMAXFA recurrence across tiles to compute new_global_sum and exp_max per tile
    global_max = None
    global_sum = None

    for ti in range(num_tiles):
        c0 = ti * tile_s1
        c1 = c0 + tile_s1
        tile = arr_f32[:, c0:c1]
        # local max per row for this tile
        local_max = np.max(tile, axis=1, keepdims=True).astype(np.float32)
        if global_max is not None:
            local_max = np.maximum(local_max, global_max).astype(np.float32) 
        if ti == 0:
            new_global_max = local_max
            tmp_float = (tile - new_global_max) * scale
            tmp_float_exp = np.exp(tmp_float).astype(np.float32)
            new_global_sum = (np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32))
            exp_max_tile = np.ones_like(new_global_max).astype(np.float32)
        else:
            # exp_max = exp((global_max - local_max) * scale)
            exp_max = (global_max - local_max).astype(np.float32)
            exp_max = np.exp(exp_max * scale).astype(np.float32)
            new_global_max = local_max
            tmp_float = (tile - new_global_max) * scale
            tmp_float_exp = np.exp(tmp_float).astype(np.float32)
            new_global_sum = exp_max * global_sum + (np.sum(tmp_float_exp, axis=1, keepdims=True).astype(np.float32) )
            exp_max_tile = exp_max

        # record results and update global state
        full_exp[:, c0:c1] = tmp_float_exp
        global_sums.append(new_global_sum.reshape(-1))
        exp_max_parts.append(exp_max_tile.reshape(-1))
        global_max = new_global_max
        global_sum = new_global_sum

    tmp_float_exp = full_exp
    # p saved as FP16 (store raw exponentials per tile as half)
    soft = (full_exp).astype(np.float16)
    p_torch = torch.from_numpy(soft)
    #soft.tofile(os.path.join(path, 'p.bin'))
    #tmp_float_exp.tofile(os.path.join(path, 'p_fp32.bin'))
    p_fp32_torch = torch.from_numpy(tmp_float_exp)

    # generate random V (S1 x HEAD_SIZE) and compute y = soft (S0 x S1) dot V (S1 x HEAD_SIZE)
    v_fp32 = (np.random.randn(s1, head_size).astype(np.float16) * 1.2).astype(np.float32)
    v = v_fp32.astype(np.float16)
    # soft (S0 x S1) as float32
    soft_f32 = soft.astype(np.float32)
    # compute full pv by accumulating per-tile partials
    pv = np.zeros((s0, head_size), dtype=np.float32)
    # compute per-tile partials based on TILE_S1
    num_tiles = s1 // tile_s1
    pv_tile_fifo_parts = []
    for ti in range(num_tiles):
        c0 = ti * tile_s1
        soft_tile = soft_f32[:, c0:c0+tile_s1]
        v_tile = v[c0:c0+tile_s1, :].astype(np.float32)
        pv_tile_fifo = (soft_tile.dot(v_tile)).astype(np.float32)
        pv_tile_fifo_parts.append(pv_tile_fifo)
        pv += pv_tile_fifo

    #v.tofile(os.path.join(path, 'v.bin'))
    v_torch = torch.from_numpy(v)
    vt = v.T.astype(np.float16)    
    #vt.tofile(os.path.join(path, 'vt.bin'))       
    #pv.tofile(os.path.join(path, 'pv.bin'))
    pv_torch = torch.from_numpy(pv)
    # write per-tile partials as pv_tile_fifo0.bin, pv_tile_fifo1.bin
    """
    for idx, part in enumerate(pv_tile_fifo_parts):
        part.tofile(os.path.join(path, f'pv_tile_fifo{idx}.bin'))
    # write per-tile global_sum and exp_max parts
    for idx, g in enumerate(global_sums):
        g.astype(np.float32).tofile(os.path.join(path, f'global_sum_part{idx}.bin'))
    for idx, e in enumerate(exp_max_parts):
        e.astype(np.float32).tofile(os.path.join(path, f'exp_max_part{idx}.bin'))
    """

    # compute running output o: use exp_max per-tile for accumulation and divide by new_global_sum on last tile
    o_running = np.zeros((s0, head_size), dtype=np.float32)
    for ti, part in enumerate(pv_tile_fifo_parts):
        if ti == 0:
            o_running = part.copy()
        else:
            exp_max_tile = exp_max_parts[ti].reshape((s0, 1)).astype(np.float32)
            o_running = exp_max_tile * o_running + part
            if ti == num_tiles - 1:
                new_global_sum_tile = global_sums[ti].reshape((s0, 1)).astype(np.float32)
                o_running = o_running / new_global_sum_tile
        # write per-iteration o
        # o_running.astype(np.float32).tofile(os.path.join(path, f'o_part{ti}.bin'))
    # write final running output
    #o_running.astype(np.float32).tofile(os.path.join(path, 'o.bin'))
    o_torch = torch.from_numpy(o_running.astype(np.float32))

    return q_torch, kt_torch, v_torch, o_torch, [qk_torch, p_torch, p_fp32_torch, pv_torch]


class TestCustomFA(TestCase):

    def test_fa_custom_ops(self):

        for seq_len in [1024, 2048, 4096, 8192, 16384, 32768]:
            for head_size in [64, 128]:
                for is_causal in [False, True]:
                    print(f"Testing for sequence length: {seq_len}, head_size: {head_size}, causal: {is_causal}")
                    # Define the tensor size
                    s0 = seq_len
                    s1 = seq_len
                    q, k, v, cpuout, intermediate_result = gen_case(s0, s1, head_size=head_size, is_causal=is_causal)

                    q_npu = q.npu()
                    k_npu = k.npu()
                    v_npu = v.npu()
                    # Call the custom my_fa operator
                    output = torch.ops.npu.my_fa(q_npu, k_npu, v_npu, is_causal)

                    # Validate the results
                    self.assertRtolEqual(output, cpuout, prec=1.e-3)


if __name__ == "__main__":
    run_tests()
