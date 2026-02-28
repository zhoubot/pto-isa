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


def gen_golden_data_trowexpandop(param, element_op: str):
    dtype = param.dtype
    row, col = [param.tile_row, param.tile_col]

    input1 = np.random.uniform(low=-2, high=2, size=[row, col]).astype(dtype)
    input2 = np.random.uniform(low=1, high=2, size=[row, 1]).astype(dtype)

    if element_op == "div":
        golden = input1 / input2
    elif element_op == "mul":
        golden = input1 * input2
    elif element_op == "sub":
        golden = input1 - input2
    elif element_op == "add":
        golden = input1 + input2
    elif element_op == "min":
        golden = np.minimum(input1, input2)
    elif element_op == "max":
        golden = np.maximum(input1, input2)
    elif element_op == "expdif":
        golden = np.exp(input1 - input2)
    else:
        raise ValueError(element_op)

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TRowExpandOpParams:
    def __init__(self, dtype, tile_row, tile_col):
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col


def generate_case_name(param, element_op: str):
    dtype_str = {np.float32: "float", np.float16: "half"}[param.dtype]

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TROWEXPANDOPTest.case_{element_op}_{dtype_str}"
    name += substring(param.tile_row, param.tile_col)
    return name


if __name__ == "__main__":
    case_params_list = [
        TRowExpandOpParams(np.float32, 64, 64),
        TRowExpandOpParams(np.float16, 16, 256),
    ]
    operations_list = ["div", "mul", "sub", "add", "min", "max", "expdif"]

    combinations = [(param, element_op) for param in case_params_list for element_op in operations_list]

    for param, element_op in combinations:
        case_name = generate_case_name(param, element_op)
        os.makedirs(case_name, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trowexpandop(param, element_op)
        os.chdir(original_dir)
