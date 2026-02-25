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

np.random.seed(42)


class MScatterParams:
    def __init__(self, name, dtype, src_rows, src_cols, out_size):
        self.name = name
        self.dtype = dtype
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.out_size = out_size


def gen_golden_data(param: MScatterParams):
    dtype = param.dtype
    src_rows = param.src_rows
    src_cols = param.src_cols
    out_size = param.out_size

    src_size = src_rows * src_cols

    # Generate 1D source data - use integers first, then convert
    src = ((np.arange(1, src_size + 1) % 256) + 1).astype(dtype)

    # Generate 1D indices (each index points to a valid position in output)
    indices = (np.arange(0, src_size) % out_size).astype(np.int32)

    # Compute golden output: out[indices[i]] = src[i]
    golden = np.zeros(out_size, dtype=dtype)
    for i in range(src_size):
        golden[indices[i]] = src[i]

    src.tofile("src.bin")
    indices.tofile("indices.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    case_params_list = [
        MScatterParams("MSCATTERTest.case_half_8x32_1024", np.float16, 8, 32, 1024),
        MScatterParams("MSCATTERTest.case_half_16x64_2048", np.float16, 16, 64, 2048),
        MScatterParams("MSCATTERTest.case_float_8x32_512", np.float32, 8, 32, 512),
        MScatterParams("MSCATTERTest.case_float_16x32_1024", np.float32, 16, 32, 1024),
        MScatterParams("MSCATTERTest.case_float_16x64_2048", np.float32, 16, 64, 2048),
        MScatterParams("MSCATTERTest.case_float_8x8_128", np.float32, 8, 8, 128),
        MScatterParams("MSCATTERTest.case_int32_8x16_256", np.int32, 8, 16, 256),
        MScatterParams("MSCATTERTest.case_int32_16x32_1024", np.int32, 16, 32, 1024),
        MScatterParams("MSCATTERTest.case_int32_16x16_512", np.int32, 16, 16, 512),
        MScatterParams("MSCATTERTest.case_uint8_16x32_1024", np.uint8, 16, 32, 1024),
        MScatterParams("MSCATTERTest.case_uint8_16x64_2048", np.uint8, 16, 64, 2048),
    ]

    for param in case_params_list:
        if not os.path.exists(param.name):
            os.makedirs(param.name)
        original_dir = os.getcwd()
        os.chdir(param.name)
        gen_golden_data(param)
        os.chdir(original_dir)
        print(f"Generated {param.name}")

    print("All MSCATTER test data generated successfully")
