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
import ml_dtypes
bfloat16 = ml_dtypes.bfloat16
np.random.seed(23)


def recalculate_indices(indices, rows, cols):
    for row in range(indices.shape[0]):
        for col in range(indices.shape[1]):
            indices[row, col] = indices[row, col] * cols + col
    return indices


def scatter(src, indices):
    dst = np.zeros_like(src, dtype=src.dtype).flatten()
    for row in range(indices.shape[0]):
        for col in range(indices.shape[1]):
            idx = indices[row, col]
            dst[idx] = src[row, col]
    return dst


class TScatterParams:
    def __init__(self, name, src0_type, src1_type, src0_row, src0_col, src1_row, src1_col):
        self.name = name
        self.src0_type = src0_type
        self.src1_type = src1_type
        self.src0_row = src0_row
        self.src0_col = src0_col
        self.src1_row = src1_row
        self.src1_col = src1_col


def gen_golden_data(param: TScatterParams):
    src0_type = param.src0_type
    src1_type = param.src1_type
    src0_row = param.src0_row
    src0_col = param.src0_col
    src1_row = param.src1_row  # index
    src1_col = param.src1_col  # index

    src_data = np.random.randint(0, 20, (src0_row * src0_col)).astype(src0_type)
    src_data = src_data.reshape((src0_row, src0_col))

    indices = np.random.randint(0, 2, (src1_row * src1_col)).astype(src1_type)
    indices = indices.reshape((src1_row, src1_col))
    indices = recalculate_indices(indices, src0_row, src0_col)

    golden = scatter(src_data, indices)

    src_data.tofile('input.bin')
    indices.tofile('indexes.bin')
    golden.tofile('golden.bin')
    os.chdir(original_dir)


if __name__ == "__main__":
    case_params_list = [
        TScatterParams("TSCATTERTest.case1", np.int16, np.uint16, 2, 32, 1, 32),
        TScatterParams("TSCATTERTest.case2", np.float16, np.uint16, 63, 64, 63, 64),
        TScatterParams("TSCATTERTest.case3", np.int32, np.uint32, 31, 128, 31, 128),
        TScatterParams("TSCATTERTest.case4", np.int16, np.int16, 15, 64 * 3, 15, 192),
        TScatterParams("TSCATTERTest.case5", np.float32, np.int32, 7, 64 * 7, 7, 448),
        TScatterParams("TSCATTERTest.case6", np.int8, np.uint16, 256, 32, 256, 32),
        TScatterParams("TSCATTERTest.case7", np.float32, np.uint32, 32, 64, 32, 64),
        TScatterParams("TSCATTERTest.case8", bfloat16, np.int16, 32, 64, 32, 64)
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)