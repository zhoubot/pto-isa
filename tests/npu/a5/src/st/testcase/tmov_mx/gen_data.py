#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import math
import numpy as np
import ml_dtypes
import en_dtypes

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
fp4_e1m2x2 = en_dtypes.float4_e1m2
fp4_e2m1x2 = en_dtypes.float4_e2m1

def convert_x1_scale_format(x1_mx_gm, block_size=16, c0_size_mx=2):
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(x1_mx_gm, 
                       ((0, pad_m), (0, pad_k)), 
                       mode='constant',
                       constant_values=0)
    else:
        padded = x1_mx_gm
    
    m_padded = m + pad_m
    k_padded = k + pad_k

    x1_scale_gm = padded.reshape((int(m_padded / block_size), block_size, 
                                 int(k_padded / c0_size_mx), c0_size_mx))
    x1_scale_gm = x1_scale_gm.transpose(0, 2, 1, 3)
    x1_scale_gm = x1_scale_gm.reshape(x1_scale_gm.shape[0] * x1_scale_gm.shape[1], 
                                     x1_scale_gm.shape[2] * x1_scale_gm.shape[3])

    return x1_scale_gm


def convert_x2_scale_format(x2_mx_gm, block_size=16, c0_size_mx=2):
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(x2_mx_gm, 
                       ((0, pad_k), (0, pad_n)),
                       mode='constant',
                       constant_values=0)
    else:
        padded = x2_mx_gm
    
    k_padded, n_padded = padded.shape
    
    x2_scale_gm = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    x2_scale_gm = x2_scale_gm.reshape(x2_scale_gm.shape[1] * x2_scale_gm.shape[3], 
                                      x2_scale_gm.shape[0] * x2_scale_gm.shape[2])

    return x2_scale_gm


def pack_two_fp4(scale_matrix):
    scale_matrix_row = scale_matrix.shape[0]
    scale_matrix_col = scale_matrix.shape[1]
    scale_matrix_bin = scale_matrix.flatten()
    scale_matrix_high = scale_matrix_bin[::2].view(np.uint8)
    scale_matrix_low = scale_matrix_bin[1::2].view(np.uint8)
    low_bits = (scale_matrix_low & 0x0F) << 4
    high_bits = scale_matrix_high & 0x0F
    combined = low_bits | high_bits
    scale_matrix_bin = combined.reshape(scale_matrix_row, scale_matrix_col // 2)
    return scale_matrix_bin


def align_to_multiple(k, alignment=64):
    return (k + alignment - 1) // alignment * alignment


def gen_golden_data(case_name, param):
    a_type, b_type, dst_type, src_format = param.atype, param.btype, param.ctype, param.src_format
    m, k, n, is_atrans, is_btrans = param.m, param.k, param.n, False, False
    start_m, start_k, start_n = param.start_m, param.start_k, param.start_n
    base_m, base_k, base_n = param.base_m, param.base_k, param.base_n

    if a_type == fp4_e2m1x2:
        x1_gm = np.random.randint(-6, 6, [m, k]).astype(a_type)
    elif a_type == fp4_e1m2x2:
        x1_gm = np.random.randint(-1, 2, [m, k]).astype(a_type)
    else:
        x1_gm = np.random.randint(-10, 10, [m, k]).astype(a_type)

    if b_type == fp4_e2m1x2:
        x2_gm = np.random.randint(-6, 6, [k, n]).astype(b_type)
    elif b_type == fp4_e1m2x2:
        x2_gm = np.random.randint(-1, 2, [k, n]).astype(b_type)
    else:
        x2_gm = np.random.randint(-10, 10, [k, n]).astype(b_type)

    k_aligned = align_to_multiple(k, 64)
    #compact cases
    if base_m != 0: 
        x1_pad = np.random.randint(0, 2, [base_m, base_k]).astype(a_type)
        x1_pad[:m, :k_aligned] = 0
        x1_pad[:m, :k] = x1_gm
        if a_type == fp4_e2m1x2 or a_type == fp4_e1m2x2:
            x1_pad = pack_two_fp4(x1_pad)
        x2_pad = np.random.randint(0, 2, [base_k, base_n]).astype(b_type)
        x2_pad[:k_aligned, :n] = 0
        x2_pad[:k, :n] = x2_gm
        if b_type == fp4_e2m1x2 or b_type == fp4_e1m2x2:
            x2_pad = pack_two_fp4(x2_pad)
        x1_pad.tofile("./x1_gm.bin")
        x2_pad.tofile("./x2_gm.bin")
    else:
        if a_type == fp4_e2m1x2 or a_type == fp4_e1m2x2:
            x1_gm_bin = pack_two_fp4(x1_gm)
            x2_gm_bin = pack_two_fp4(x2_gm)
            x1_gm_bin.tofile("./x1_gm.bin")
            x2_gm_bin.tofile("./x2_gm.bin")
        else:
            x1_gm.tofile("./x1_gm.bin")
            x2_gm.tofile("./x2_gm.bin")

    k_mx = k_aligned // 32
    x1_mx_gm = np.random.randint(127, 130, [m, k_mx]).astype(np.uint8)
    x2_mx_gm = np.random.randint(127, 130, [k_mx, n]).astype(np.uint8)

    ###################### compute ########################
    x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
    x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
    x1 = np.zeros([m, k], dtype=np.float64)
    x2 = np.zeros([k, n], dtype=np.float64)

    for i in range(x1_gm.shape[1]):
        x1[:, i] = x1_gm[:, i] * x1_mx[:, i // 32]
        x2[i, :] = x2_gm[i, :] * x2_mx[i // 32, :]

    x1_chunk = x1[start_m:, start_k:]
    x2_chunk = x2[start_k:, start_n:]
    
    golden = np.matmul(x1_chunk.astype(np.float64), x2_chunk.astype(np.float64)).astype(dst_type)
    golden.tofile("./golden.bin")

    if base_m != 0:
        x1_mx_pad = np.random.randint(127, 130, [base_m, base_k // 32]).astype(np.uint8)
        x2_mx_pad = np.random.randint(127, 130, [base_k // 32, base_n]).astype(np.uint8)
        x1_mx_pad[:m, :k_mx] = x1_mx_gm
        x2_mx_pad[:k_mx, :n] = x2_mx_gm
        x1_mx_gm = x1_mx_pad
        x2_mx_gm = x2_mx_pad

    if src_format == 'zznn':
        # x1_scale_gm, convert to zZ format
        x1_scale_gm = convert_x1_scale_format(x1_mx_gm, 16, 2)
        # x1_scale_gm, convert to nN format
        x2_scale_gm = convert_x2_scale_format(x2_mx_gm, 16, 2)
    elif src_format == 'dndn':
        # x1_scale_gm, convert to dn format
        x1_scale_gm = x1_mx_gm.reshape((x1_mx_gm.shape[0], x1_mx_gm.shape[1] // 2, 2)).transpose(1, 0, 2)
        x2_scale_gm = x2_mx_gm.transpose()
    else:
        x1_scale_gm = x1_mx_gm
        # x2_scale_gm, convert to nd format
        x2_scale_gm = x2_mx_gm.reshape((x2_mx_gm.shape[0] // 2, 2, x2_mx_gm.shape[1])).transpose(0, 2, 1)
        
    x1_scale_gm.tofile("./x1_mx_gm.bin")
    x2_scale_gm.tofile("./x2_mx_gm.bin")


class TMovmxParams:
    def __init__(self, atype, btype, ctype, m, k, n, src_format='zznn', start_m=0, start_k=0, start_n=0, 
        base_m=0, base_k=0, base_n=0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.src_format = src_format
        self.start_m = start_m
        self.start_k = start_k
        self.start_n = start_n
        self.base_m = base_m
        self.base_k = base_k
        self.base_n = base_n

if __name__ == "__main__":
    case_name_list = [
        # normal
        "TMOVMXTest.case1",
        "TMOVMXTest.case2",
        "TMOVMXTest.case3",
        "TMOVMXTest.case4",
        "TMOVMXTest.case5",
        "TMOVMXTest.case6",
        "TMOVMXTest.case7",
        "TMOVMXTest.case8",
        "TMOVMXTest.case9",
        # startIdx != 0
        "TMOVMXTest.case10",
        "TMOVMXTest.case11",
        "TMOVMXTest.case12",
        "TMOVMXTest.case13",
        "TMOVMXTest.case14",
        "TMOVMXTest.case15",
        # TExtractCompact
        "TMOVMXTest.case16",
        "TMOVMXTest.case17",
        "TMOVMXTest.case18",
        "TMOVMXTest.case19",
        "TMOVMXTest.case20",
        "TMOVMXTest.case21",
        "TMOVMXTest.case22",
        "TMOVMXTest.case23",
        "TMOVMXTest.case24",
    ]

    case_params_list = [
        # TExtract
        # normal
        TMovmxParams(fp8_e5m2, fp8_e5m2, np.float32, 128, 64, 64, 'zznn'),
        TMovmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 32, 128, 64, 'zznn'),
        TMovmxParams(fp8_e5m2, fp8_e5m2, np.float32, 64, 128, 80, 'zznn'),  # need to use compact mode.

        TMovmxParams(fp8_e5m2, fp8_e5m2, np.float32, 115, 64, 30, 'ndnd'),
        TMovmxParams(fp8_e5m2, fp8_e4m3fn, np.float32, 64, 120, 64, 'ndnd'),
        TMovmxParams(fp4_e2m1x2, fp4_e2m1x2, np.float32, 48, 192, 96, 'ndnd'),

        TMovmxParams(fp8_e5m2, fp8_e5m2, np.float32, 128, 64, 64, 'dndn'),
        TMovmxParams(fp4_e2m1x2, fp4_e2m1x2, np.float32, 95, 12, 90, 'dndn'),
        TMovmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 4, 30, 8, 'dndn'),
        # startIdx != 0
        TMovmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 128, 32, 64, 'zznn', 64, 0, 32),
        TMovmxParams(fp4_e2m1x2, fp4_e2m1x2, np.float32, 128, 98, 64, 'zznn', 32, 64, 0),

        TMovmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 128, 60, 254, 'ndnd', 16, 0, 64),
        TMovmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 48, 180, 96, 'ndnd', 16, 64, 32),

        TMovmxParams(fp8_e5m2, fp8_e5m2, np.float32, 95, 120, 89, 'dndn', 16, 64, 32),
        TMovmxParams(fp4_e1m2x2, fp4_e2m1x2, np.float32, 48, 190, 98, 'dndn', 16, 0, 64),

        # TExtractCompact
        TMovmxParams(fp8_e5m2, fp8_e5m2, np.float32, 46, 66, 45, 'zznn', 0, 0, 0, 128, 256, 128),
        TMovmxParams(fp8_e5m2, fp8_e5m2, np.float32, 68, 130, 80, 'zznn', 16, 64, 32, 128, 256, 128),
        TMovmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 127, 126, 130, 'zznn', 32, 64, 64, 256, 128, 256),
        TMovmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 80, 96, 192, 'ndnd', 48, 0, 64, 128, 256, 256),
        TMovmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 98, 126, 108, 'ndnd', 32, 64, 32, 128, 256, 128),
        TMovmxParams(fp4_e1m2x2, fp4_e2m1x2, np.float32, 68, 196, 80, 'ndnd', 0, 64, 64, 128, 256, 128),
        TMovmxParams(fp8_e5m2, fp8_e4m3fn, np.float32, 32, 64, 108, 'dndn', 16, 0, 32, 128, 256, 128),
        TMovmxParams(fp8_e5m2, fp8_e4m3fn, np.float32, 196, 146, 96, 'dndn', 64, 64, 32, 256, 256, 128),
        TMovmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 97, 96, 122, 'dndn', 32, 0, 64, 128, 256, 128),
    ]


    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)