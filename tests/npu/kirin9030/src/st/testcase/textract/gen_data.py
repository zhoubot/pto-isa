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
import numpy as np

np.random.seed(19)
def check(x,n):
    if len(x) < n:
        x = '0' * (n-len(x)) + x
    elif len(x) > n:
        x = x[1:]
    return x

def cast(c, dtype):
    if dtype == 'fp16':
        c = np.array(c).astype(np.float16)
    elif dtype == 'fp32':
        c = np.array(c).astype(np.float32)
    return c

def HF8(input):
    if len(input) != 8:
        print("The input must be 8-bit. Please check the input.")
        exit(-1)
    d, e = '', ''
    s, m = input[0], input[5:]
    m1, m2, m3 = int(input[5]), int(input[6]), int(input[7])
    if input[1] == '1' or input[2] == '1':
        d, e = input[1:3], input[3:5]
    elif input[3] == '1':
        d, e = input[1:4], input[4]
    else:
        d, e = input[1:5], ''
    f1 = -1 if s == '1' else 1
    f2 = 1
    if d == '0000':
        if s == '1':
            if m == '000':
                return np.nan
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23) * f1
        else:
            if m == '000':
                return 0
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23)
        return input
    elif d == '0001':
        f2 = 0
        input = (1 + (m1 * 4 + m2 * 2 + m3)/8) * 2 ** f2 * f1
        return input
    elif d == '001':
        f2 = -1 if e == '1' else 1
        input = (1 + (m1 * 4 + m2 * 2 + m3)/8) * 2 ** f2 * f1
        return input
    elif d == '01':
        f2 = -1 if int(input[3]) == 1 else 1
        input = (1 + (m1 * 4 + m2 * 2 + m3)/8) * 2 ** (f2 * (2 + int(input[4]))) * f1
        return input
    elif d == '10':
        f2 = -1 if int(input[3]) == 1 else 1
        input = (1 + (m2 * 2 + m3)/4) * 2 ** (f2 * (4 + int(input[4]) * 2 + int(input[5]))) * f1
        return input
    elif d == '11':
        f2 = -1 if int(input[3]) == 1 else 1
        if e == '01' and m == '111':
            return f1 * np.inf
        input = (1 + m3/2) * 2 ** (f2 * (8 + int(input[4]) * 4 + int(input[5]) * 2 + int(input[6]))) * f1
        return input


def create_padded_tensors(
    x1_gm, x2_gm, m, n, k, target_m, target_n, target_k, src_type = np.int8,
    rand_range_right = (1,5),
    rand_range_down = (1,5),
    rand_range_corner = (1,5)):
    assert target_m >= m, f"target_m ({target_m}) mast be >= m ({m})"
    assert target_n >= n, f"target_n ({target_n}) mast be >= n ({n})"
    assert target_k >= k, f"target_k ({target_k}) mast be >= k ({k})"
    # x1_gm_padded：target_m, target_k
    x1_gm_padded = np.zeros((target_m, target_k), dtype=np.int32).astype(src_type)
    # origin data
    x1_gm_padded[:m, :k] = x1_gm
    # Right-side random value padding (k-direction extension)
    right_fill = np.random.randint(rand_range_right[0], rand_range_right[1],
                                    size=(m, target_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[:m, k:target_k] = right_fill
    # Add 0 to the bottom (extended in the m direction)
    x1_gm_padded[m:target_m, :k] = 0

    # Add random value in the bottom right corner
    corner_fill = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                    size=(target_m - m, target_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[m:target_m, k:target_k] = corner_fill
    #x2_gm_padded：target_k, target_n
    x2_gm_padded = np.zeros((target_k, target_n), dtype=np.int32).astype(src_type)
    x2_gm_padded[:k, :n] = x2_gm
    down_fill = np.random.randint(rand_range_down[0], rand_range_down[1],
                                    size=(target_k - k, n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:target_k, :n] = down_fill
    x2_gm_padded[:k, n:target_n] = 0
    corner_fill2 = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                     size=(target_k - k, target_n - n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:target_k, n:target_n] = corner_fill2
    return x1_gm_padded, x2_gm_padded


def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, start_m, start_k, start_n, is_bias, is_atrans, is_btrans, target_m, target_k, target_n = \
        param.m, param.k, param.n, param.start_m, param.start_k, param.start_n, False, param.is_atrans, \
        param.is_btrans, param.target_m, param.target_k, param.target_n

    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(src_type)
    # get slice
    x1_slice = x1_gm[start_m:, start_k:]  # from (rowIdx1, colIdx1) to the end
    x2_slice = x2_gm[start_k:, start_n:]  # from (rowIdx2, colIdx2) to the end
    golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type)
    # padding for unaligned data
    if target_m > 0 or target_n > 0 or target_k > 0:
        target_m = target_m if target_m > 0 else m
        target_n = target_n if target_n > 0 else n
        target_k = target_k if target_k > 0 else k
        x1_gm, x2_gm = create_padded_tensors(x1_gm, x2_gm, m, n, k, target_m, target_n, target_k, src_type, \
                    rand_range_right=(1,5), rand_range_down=(1,5), rand_range_corner=(1,5))

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if not is_btrans:
        x2_gm = x2_gm.transpose()#[N,K]

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class textractParams:
    def __init__(self, atype, btype, ctype, m, k, n, start_m, start_k, start_n,  \
        is_atrans=0, is_btrans=0, target_m = 0, target_k = 0, target_n = 0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.start_m = start_m
        self.start_k = start_k
        self.start_n = start_n
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans
        self.target_m = target_m
        self.target_k = target_k
        self.target_n = target_n

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TEXTRACTTest.case1",
        "TEXTRACTTest.case2",
        "TEXTRACTTest.case3",
        "TEXTRACTTest.case4",
        "TEXTRACTTest.case5",
        "TEXTRACTTest.case6",
        "TEXTRACTTest.case7",
        "TEXTRACTTest.case8",
    ]

    case_params_list = [
        # TExtract
        # normal
        textractParams(np.float16, np.float16, np.float16, 32, 96, 64, 0, 0, 0, 0, 0),
        textractParams(np.int8, np.int8, np.int32, 128, 128, 64, 0, 0, 0, 0, 0),
        # startIdx 
        textractParams(np.float16, np.float16, np.float16, 64, 96, 64, 32, 16, 16, 0, 0),
        textractParams(np.int8, np.int8, np.int32, 128, 128, 64, 32, 64, 32, 0, 0),
        # transpose，startIdx 
        textractParams(np.float16, np.float16, np.float16, 64, 128, 64, 0, 64, 0, 1, 1),
        textractParams(np.int8, np.int8, np.int32, 128, 64, 128, 32, 0, 0, 1, 1),
        # dynamic shape
        textractParams(np.int8, np.int8, np.int32, 64, 96, 32, 32, 0, 0, 1, 0),
        textractParams(np.float16, np.float16, np.float16, 64, 48, 96, 16, 16, 0, 1, 0),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
