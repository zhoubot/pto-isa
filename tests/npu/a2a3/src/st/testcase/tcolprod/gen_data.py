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

import os
import numpy as np

np.random.seed(23)


def gen_golden_data(param):
    data_type = param.data_type
    cols = param.col
    src_row = param.src_row
    src_valid_row = param.src_valid_row

    input_arr = np.random.rand(src_valid_row, cols) * 10
    input_arr = input_arr.astype(data_type)
    golden = np.zeros((1, cols), dtype=data_type)
    golden[0] = np.prod(input_arr, axis=0)
    input_arr.tofile("input.bin")
    golden.tofile("golden.bin")


class TColProd:
    def __init__(self, data_type, col, src_row, src_valid_row):
        self.data_type = data_type
        self.col = col
        self.src_row = src_row
        self.src_valid_row = src_valid_row


if __name__ == "__main__":
    case_name_list = [
        "TCOLPRODTest.case1",
        "TCOLPRODTest.case2",
        "TCOLPRODTest.case3",
        "TCOLPRODTest.case4",
        "TCOLPRODTest.case5",
        "TCOLPRODTest.case6",
        "TCOLPRODTest.case7",
        "TCOLPRODTest.case8",
        "TCOLPRODTest.case9",
        "TCOLPRODTest.case10",
        "TCOLPRODTest.case11",
        "TCOLPRODTest.case12",
        "TCOLPRODTest.case13",
        "TCOLPRODTest.case14",
        "TCOLPRODTest.case15",
        "TCOLPRODTest.case16",
    ]

    case_params_list = [
        TColProd(np.int16, 16, 16, 8),
        TColProd(np.int32, 16, 16, 8),
        TColProd(np.float32, 16, 16, 8),
        TColProd(np.int16, 128, 16, 8),
        TColProd(np.int32, 64, 16, 8),
        TColProd(np.float32, 64, 16, 8),
        TColProd(np.int16, 512, 16, 8),
        TColProd(np.int32, 256, 16, 8),
        TColProd(np.float32, 256, 16, 8),
        TColProd(np.int16, 512, 16, 7),
        TColProd(np.int32, 256, 32, 31),
        TColProd(np.float32, 256, 32, 31),
        TColProd(np.float32, 256, 16, 1),
        TColProd(np.float16, 256, 1, 1),
        TColProd(np.float16, 64, 5, 3),
        TColProd(np.float16, 16, 8, 4),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_params_list[i])

        os.chdir(original_dir)
