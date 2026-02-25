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


def gen_golden_data_ttri(case_name, param):
    row, col = [param.row, param.col]
    is_upper = param.is_upper
    diagonal = param.diagonal

    golden = np.zeros((row, col), dtype=np.int32)

    for i in range(row):
        for j in range(col):
            if is_upper:
                golden[i][j] = (0 if j < diagonal + i else 1)
            else:
                golden[i][j] = (1 if j <= diagonal + i else 0)

    # Save the input and golden data to binary files
    
    golden.tofile("golden.bin")


class TTRIParams:
    def __init__(self, row, col, is_upper, diagonal):
        self.row = row
        self.col = col
        self.is_upper = is_upper
        self.diagonal = diagonal


def generate_case_name(param):
    name = f"TTRITest.case_ttri_{param.row}x{param.col}_"
    name += f"{'upper' if param.is_upper else 'lower'}_"
    name += f"diag{'_' if param.diagonal < 0 else ''}{abs(param.diagonal)}"

    return name


if __name__ == "__main__":
    test_params = [
        TTRIParams(64, 64, True, 0),
        TTRIParams(100, 64, True, -2),
        TTRIParams(128, 32, False, 1),
        TTRIParams(256, 16, False, -1),
        TTRIParams(200, 48, True, 2),
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    
    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    for param in test_params:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        print(f"Generating data for case: {case_name}")
        gen_golden_data_ttri(case_name, param)
        os.chdir(original_dir)
