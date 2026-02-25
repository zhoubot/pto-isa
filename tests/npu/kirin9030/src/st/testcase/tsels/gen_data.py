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

PAD_VALUE_NULL = "PAD_VAL_NULL"
PAD_VALUE_MAX = "PAD_VAL_MAX"


def gen_golden_data_tsels(case_name, param):
    dtype = param.dtype

    if param.pad_value == PAD_VALUE_MAX:
        height, width = [param.global_row, param.global_col]
    else:
        height, width = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    if dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        input1 = np.random.randint(1, 10, size=[height, width]).astype(dtype)
        input2 = np.random.randint(10, 20, size=[height, width]).astype(dtype)
    else:
        input1 = np.random.uniform(low=-1303.033, high=33003.033, size=[height, width]).astype(dtype)
        input2 = np.random.uniform(low=-1303.033, high=33003.033, size=[height, width]).astype(dtype)
    select_mode = np.random.randint(0, 2, dtype=np.uint8)

    golden = np.empty_like(input1)

    for i in range(height):
        for j in range(width):
            golden[i, j] = input1[i, j] if select_mode == 1 else input2[i, j]

    # Apply valid region constraints
    output = np.zeros([height, width]).astype(dtype)
    for h in range(height):
        for w in range(width):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    select_mode.tofile("input_scalar.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class TestParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col,
                 valid_row, valid_col,
                 pad_value=PAD_VALUE_NULL):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.pad_value = pad_value

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.uint32: 'uint32',
        np.int16: 'int16',
        np.uint16: 'uint16',
        np.int8: 'int8',
        np.uint8: 'uint8',
    }[param.dtype]
    return f"TSELSTest.case_{dtype_str}_{param.global_row}x{param.global_col}"\
        f"_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TestParams(np.float32, 64, 64, 32, 32, 64, 64),
        TestParams(np.float32, 128, 128, 64, 64, 128, 128),
        TestParams(np.float32, 2, 32, 2, 32, 2, 32),
        TestParams(np.int32, 2, 32, 2, 32, 2, 32),
        TestParams(np.uint32, 2, 32, 2, 32, 2, 32),
        TestParams(np.float16, 2, 32, 2, 32, 2, 32),
        TestParams(np.int16, 2, 32, 2, 32, 2, 32),
        TestParams(np.int8, 2, 32, 2, 32, 2, 32),
        TestParams(np.uint8, 2, 32, 2, 32, 2, 32),

        TestParams(np.float32, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX),
        TestParams(np.float32, 16, 200, 20, 224, 16, 200, PAD_VALUE_MAX),
        TestParams(np.float32, 16, 200, 20, 256, 16, 200, PAD_VALUE_MAX),
        TestParams(np.float32, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX),
        TestParams(np.uint16, 16, 200, 20, 224, 16, 200, PAD_VALUE_MAX),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tsels(case_name, param)
        os.chdir(original_dir)