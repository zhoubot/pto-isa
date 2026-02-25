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

np.random.seed(20260127)


def gen_golden_data(case_name, param):
    a_type = param.atype
    b_type = param.btype
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = (
        param.m,
        param.k,
        param.n,
        param.is_bias,
        False,
        False,
    )

    x1_gm = np.random.randint(-10, 10, [m, k]).astype(a_type)
    x2_gm = np.random.randint(-10, 10, [k, n]).astype(b_type)
    bias_gm = np.random.randint(-1000, 1000, [n]).astype(bias_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    if is_bias:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(
            dst_type
        ) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(
            dst_type
        )

    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.is_bias = is_bias
        if bias_type:
            self.bias_type = bias_type
        else:
            self.bias_type = ctype


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMATMULTest.case_norm_1",
        "TMATMULTest.case_norm_2",
        "TMATMULTest.case_norm_3",
        "TMATMULTest.case_norm_4",
        "TMATMULTest.case_norm_5",

        "TMATMULTest.case_bias_1",
        "TMATMULTest.case_bias_2",
        "TMATMULTest.case_bias_3",
        "TMATMULTest.case_bias_4",
        "TMATMULTest.case_bias_5",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float16, 40, 50, 60, False),
        tmatmulParams(np.int8, np.int8, np.int32, 6, 7, 8, False),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 16, 1026, False),
        tmatmulParams(np.int8, np.int8, np.int32, 26, 15, 27, False),
        tmatmulParams(np.int8, np.int8, np.int32, 101, 1, 99, False),


        tmatmulParams(np.int8, np.int8, np.int32, 8, 7, 6, True),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 15, 16, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 66, 11, 1, True),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 29, 11, 41, True, np.float16),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
