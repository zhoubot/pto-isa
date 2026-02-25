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


def gen_expand():
    H, W = 64, 64
    x = np.random.uniform(-2.0, 2.0, size=(H, W)).astype(np.float32)
    golden = np.zeros((H, W), dtype=np.float32)
    golden[:, :] = x[:, [0]]
    x.tofile("input.bin")
    golden.tofile("golden.bin")


def gen_vec_op(kind: str):
    H, W = 64, 64
    x = np.random.uniform(0.5, 2.0, size=(H, W)).astype(np.float32)
    s = np.random.uniform(0.5, 2.0, size=(H, 1)).astype(np.float32)
    if kind == "div":
        golden = x / s
    elif kind == "mul":
        golden = x * s
    elif kind == "sub":
        golden = x - s
    elif kind == "add":
        golden = x + s
    else:
        raise ValueError(kind)
    x.tofile("input1.bin")
    s.tofile("input2.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "testcases"), exist_ok=True)

    cases = [
        ("TROWEXPAND_Test.case_expand_float_64x64", gen_expand),
        ("TROWEXPAND_Test.case_div_float_64x64", lambda: gen_vec_op("div")),
        ("TROWEXPAND_Test.case_mul_float_64x64", lambda: gen_vec_op("mul")),
        ("TROWEXPAND_Test.case_sub_float_64x64", lambda: gen_vec_op("sub")),
        ("TROWEXPAND_Test.case_add_float_64x64", lambda: gen_vec_op("add")),
    ]

    cwd = os.getcwd()
    for name, fn in cases:
        os.makedirs(name, exist_ok=True)
        os.chdir(name)
        fn()
        os.chdir(cwd)

