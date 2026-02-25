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
import struct
import ctypes
import numpy as np
np.random.seed(2025)


def gen_golden_data(param):
    data_type = param.data_type
    row = param.row
    src_col = param.src_col
    dst_col = param.dst_col
    dst_valid_col = param.dst_valid_col

    if np.issubdtype(data_type, np.integer):
        value_max = np.iinfo(data_type).max
        value_min = np.iinfo(data_type).min
    else:
        value_max = np.finfo(data_type).max
        value_min = np.finfo(data_type).min
    if param.is_brcb:
        input_arr = np.random.uniform(low=value_min, high=value_max, size=(row * src_col)).astype(data_type)
        golden = np.zeros((row * src_col, dst_col))
        for i in range(row * src_col):
            for j in range(dst_col):
                golden[i][j] = input_arr[i]
    else:
        input_arr = np.random.uniform(low=value_min, high=value_max, size=(row, src_col)).astype(data_type)
        golden = np.zeros((row, dst_col))
        for i in range(row):
            for j in range(dst_valid_col):
                golden[i][j] = input_arr[i][0]
    golden = golden.astype(data_type)
    input_arr.tofile("./input.bin")
    golden.tofile("./golden.bin")


class TRowExpand:
    def __init__(self, name, data_type, row, src_col, src_validcol, dst_col, dst_valid_col, is_brcb = False):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.src_col = src_col
        self.src_validcol = src_validcol
        self.dst_col = dst_col
        self.dst_valid_col = dst_valid_col
        self.is_brcb = is_brcb


if __name__ == "__main__":
    # 用例名称
    case_params_list = [
        TRowExpand("TROWEXPANDTest.case0", np.uint16, 16, 16, 16, 512, 512),
        TRowExpand("TROWEXPANDTest.case1", np.uint8, 16, 32, 32, 256, 256),
        TRowExpand("TROWEXPANDTest.case2", np.uint32, 16, 8, 8, 128, 128),
        TRowExpand("TROWEXPANDTest.case3", np.float32, 16, 32, 32, 512, 512),
        TRowExpand("TROWEXPANDTest.case4", np.uint16, 16, 16, 1, 256, 255),
        TRowExpand("TROWEXPANDTest.case5", np.uint8, 16, 32, 1, 512, 511),
        TRowExpand("TROWEXPANDTest.case6", np.uint32, 16, 8, 1, 128, 127),
        TRowExpand("TROWEXPANDTest.case7", np.uint16, 16, 16, 1, 128, 127),
        TRowExpand("TROWEXPANDTest.case8", np.uint8, 2, 32, 1, 64, 63),
        TRowExpand("TROWEXPANDTest.case9", np.uint16, 4080, 1, 1, 16, 16, True),
        TRowExpand("TROWEXPANDTest.case10", np.uint16, 16, 1, 1, 16, 16, True),
        TRowExpand("TROWEXPANDTest.case11", np.uint32, 4080, 1, 1, 8, 8, True),
        TRowExpand("TROWEXPANDTest.case12", np.uint32, 16, 1, 1, 8, 8, True),
        TRowExpand("TROWEXPANDTest.case13", np.float32, 4080, 1, 1, 8, 8, True),
        TRowExpand("TROWEXPANDTest.case14", np.float32, 16, 1, 1, 8, 8, True),
        TRowExpand("TROWEXPANDTest.case15", np.int16, 16, 16, 16, 512, 512),
        TRowExpand("TROWEXPANDTest.case16", np.int8, 16, 32, 32, 256, 256),
        TRowExpand("TROWEXPANDTest.case17", np.int32, 16, 8, 8, 128, 128),
        TRowExpand("TROWEXPANDTest.case18", np.int16, 16, 16, 1, 256, 255),
        TRowExpand("TROWEXPANDTest.case19", np.int8, 16, 32, 1, 512, 511),
        TRowExpand("TROWEXPANDTest.case20", np.int32, 16, 8, 1, 128, 127),
        TRowExpand("TROWEXPANDTest.case21", np.int16, 16, 16, 1, 128, 127),
        TRowExpand("TROWEXPANDTest.case22", np.int8, 2, 32, 1, 64, 63),
    ]

    for _, param in enumerate(case_params_list):
        if not os.path.exists(param.name):
            os.makedirs(param.name)
        original_dir = os.getcwd()
        os.chdir(param.name)
        gen_golden_data(param)
        os.chdir(original_dir)