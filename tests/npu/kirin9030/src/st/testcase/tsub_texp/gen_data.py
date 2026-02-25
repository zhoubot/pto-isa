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

CASENAME = "TSUB_TEXP"


class TestParams:
    def __init__(self, dtype, gm_row, gm_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.tile_row = gm_row
        self.tile_col = gm_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def gen_golden_data(case_name, param):
    dtype = param.dtype
    rows = param.tile_row
    cols = param.tile_col
 
    h, w = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]
 
    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
 
    temp1 = input1 - input2
    golden = np.exp(temp1)
 
    # Apply valid region constraints
    output = np.zeros([h, w]).astype(dtype)
    for h in range(h):
        for w in range(w):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]
 
    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")
 
    return output, input1, input2, golden
 
def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half'
    }[param.dtype]
    return f"{CASENAME}Test.case_{dtype_str}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"
 
if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")
 
    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)
 
    case_params_list = [
        TestParams(np.float32, 64, 64, 64, 64, 64, 64),
        TestParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]
 
    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)