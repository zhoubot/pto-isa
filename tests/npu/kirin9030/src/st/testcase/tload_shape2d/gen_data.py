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
from enum import Enum
np.random.seed(19)
# np.set_printoptions(threshold=np.inf)


class DataFormat(Enum):
    ND2NZ = 1
    DN2NZ = 2
    ND2ND = 3
    NZ2NZ = 4
    DN2DN = 5


def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    shape0 = param.shape0
    shape1 = param.shape1
    shape2 = param.shape2
    whole_shape0 = param.ws0
    whole_shape1 = param.ws1
    whole_shape2 = param.ws2
    whole_shape3 = param.ws3
    whole_shape4 = param.ws4

    M, K, BASEM, BASEK, is_atrans = param.m, param.k, param.basem, param.basek, False
    x1_gm = np.random.randint(1, 5, [M, K]).astype(src_type)
    golden = np.zeros([BASEM, BASEK]).astype(src_type)

    if param.load_type == DataFormat['ND2NZ'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape3, whole_shape4]).astype(src_type)
        golden = np.zeros([BASEM, BASEK]).astype(src_type)  # L1中Tile大小
        # 先对golden赋值
        min_m = min(M, golden.shape[0])
        min_k = min(K, golden.shape[1])
        golden[:min_m, :min_k] = x1_gm[:min_m, :min_k]
    elif param.load_type == DataFormat['DN2NZ'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape4, whole_shape3]).astype(src_type)
        golden = np.zeros([BASEK, BASEM]).astype(src_type)  # L1中Tile大小
        min_k = min(K, golden.shape[0])
        min_m = min(M, golden.shape[1])
        golden[:min_k, :min_m] = x1_gm[:min_k, :min_m]
    elif param.load_type == DataFormat['ND2ND'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4]).astype(src_type)
        golden = np.zeros([BASEM, BASEK]).astype(src_type)  # L1中Tile大小
        print(f"origin x1_gm shape: {x1_gm.shape}")

        # 先对golden赋值
        submatrix = x1_gm[
            0:shape0,          # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,          # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,          # d2: 截取前shape2个元素（对应目标 d2=3）
            0:M,         # d3: 截取前M个元素（对应目标 d3=64）
            0:K         # d4: 截取K个元素（对应目标 d4=128）
        ]
        # 输出：(1, 2, 3, 64, 128)
        print(f"select real global shape: {submatrix.shape}")
        flattened_submatrix = submatrix.reshape(BASEM, K)
        # 输出：(384, 128)
        print(f"flattened submatrix shape: {flattened_submatrix.shape}")

        min_m = min(flattened_submatrix.shape[0], golden.shape[0])
        min_k = min(flattened_submatrix.shape[1], golden.shape[1])
        golden[:min_m, :min_k] = flattened_submatrix[:min_m, :min_k]
    elif param.load_type == DataFormat['DN2DN'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape4, whole_shape3]).astype(src_type)
        golden = np.zeros([BASEK, BASEM]).astype(src_type)
        print(f"origin x1_gm shape: {x1_gm.shape}")

        # 先对golden赋值
        submatrix = x1_gm[
            0:shape0,          # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,          # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,          # d2: 截取前shape2个元素（对应目标 d2=3）
            0:K,         # d3: 截取前M个元素（对应目标 d3=64）
            0:M         # d4: 截取K个元素（对应目标 d4=128）
        ]
        # 输出：(1, 2, 3, 128, 64)
        print(f"select real global shape: {submatrix.shape}")
        flattened_submatrix = submatrix.reshape(BASEK, M)
        # 输出：(768, 64)
        print(f"flattened submatrix shape: {flattened_submatrix.shape}")

        min_k = min(flattened_submatrix.shape[0], golden.shape[0])
        min_m = min(flattened_submatrix.shape[1], golden.shape[1])
        golden[:min_k, :min_m] = flattened_submatrix[:min_k, :min_m]
    elif param.load_type == DataFormat['NZ2NZ'].value:
        x1_gm = np.random.randint(
            1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4]).astype(src_type)

        submatrix = x1_gm[
            0:shape0,          # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,          # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,          # d2: 截取前shape2个元素（对应目标 d2=4）
            0:M,         # d3: 截取前M个元素（对应目标 d3=16）
            0:K         # d4: 截取K个元素（对应目标 d4=8）
        ]
        # 输出：(2, 2, 4, 16, 8)
        print(f"select real global shape: {submatrix.shape}")
        new_submatrix = submatrix.reshape(
            submatrix.shape[0] * submatrix.shape[1], submatrix.shape[2], submatrix.shape[3], submatrix.shape[4])  # (4, 4, 16, 8)

        golden = np.zeros([BASEM, BASEK]).astype(src_type)  # L1中Tile大小 [80,48]
        c0Size = 16
        if src_type == np.float32:
            c0Size = 8
        elif src_type == np.int8:
            c0Size = 32
        print("ND2NZ, c0Size=", c0Size)
        assert (
            BASEK % c0Size) == 0, "BASEK should be c0Size aligned when matrix is NZ format"
        assert (BASEM %
                16) == 0, "BASEM should be 16 aligned when matrix is NZ format"
        golden = golden.reshape((int(BASEM / 16), 16, int(BASEK / c0Size), c0Size)
                                ).transpose(2, 0, 1, 3).astype(src_type)  # [80,48] -> [6,5,16,8]

        golden[:new_submatrix.shape[0], :new_submatrix.shape[1],
               :new_submatrix.shape[2], :new_submatrix.shape[3]] = new_submatrix

    x2_gm = np.random.randint(1, 5, [M, K]).astype(src_type)
    # print("============x1_gm======",x1_gm)
    print("============golden.shape======", golden.shape)
    if param.load_type == DataFormat['ND2NZ'].value:
        assert (BASEM %
                16) == 0, "BASEM should be 16 aligned when matrix A is NZ format"
        c0Size = 16
        if src_type == np.float32:
            c0Size = 8
        elif src_type == np.int8:
            c0Size = 32
        print("ND2NZ, c0Size=", c0Size)
        assert (
            BASEK % c0Size) == 0, "BASEK should be c0Size aligned when matrix A is NZ format"
        golden = golden.reshape(
            (int(BASEM / 16), 16, int(BASEK / c0Size), c0Size)).transpose(2, 0, 1, 3).astype(src_type)
    elif param.load_type == DataFormat['DN2NZ'].value:
        golden = golden.transpose()
        # print("=====after transpose=======golden.shape======",golden.shape)
        assert (BASEK %
                16) == 0, "BASEK should be 16 aligned when matrix A is NZ format"
        c0Size = 16
        if src_type == np.float32:
            c0Size = 8
        elif src_type == np.int8:
            c0Size = 32
        print("DN2NZ, c0Size=", c0Size)
        assert (
            BASEM % c0Size) == 0, "BASEM should be c0Size aligned when matrix A is NZ format"
        golden = golden.reshape(
            (int(BASEM / 16), 16, int(BASEK / c0Size), c0Size)).transpose(2, 0, 1, 3).astype(src_type)

    # print("============golden======",golden)
    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, shape0, shape1, shape2, m, k,  ws0, ws1, ws2, ws3, ws4,  basem, basek, load_type):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2

        self.ws0 = ws0
        self.ws1 = ws1
        self.ws2 = ws2
        self.ws3 = ws3
        self.ws4 = ws4

        self.basem = basem  # L1 row
        self.basek = basek  # L1 col
        self.load_type = load_type


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        # 此名称需要和 TEST_F(TMATMULTest, case1)定义的名称一致
        "TLOADSHAPE2DTest.1_1_1_128_128_half_ND2NZ",
        "TLOADSHAPE2DTest.1_1_1_128_128_int8_t_ND2NZ",
        "TLOADSHAPE2DTest.1_1_1_128_128_float_ND2NZ",
        "TLOADSHAPE2DTest.1_1_1_64_128_half_DN2NZ",
        "TLOADSHAPE2DTest.1_1_1_63_127_half_ND2NZ",
        "TLOADSHAPE2DTest.1_1_1_128_128_float_ND2ND",
        "TLOADSHAPE2DTest.1_1_1_37_126_int8_t_ND2ND",

        "TLOADSHAPE2DTest.1_1_1_33_99_1_1_1_64_128_48_112_half_ND2NZ",
        "TLOADSHAPE2DTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ",

        "TLOADSHAPE2DTest.1_1_1_51_123_1_1_1_64_128_64_128_float_DN2NZ",
        "TLOADSHAPE2DTest.1_1_1_63_127_1_1_1_63_127_64_128_half_DN2NZ",

        "TLOADSHAPE2DTest.1_1_1_128_128_1_1_1_128_128_128_128_float_DN2DN",
        "TLOADSHAPE2DTest.1_1_1_37_126_1_1_1_37_126_64_126_int8_t_DN2DN",

        "TLOADSHAPE2DTest.1_10_8_16_16_1_11_9_16_16_128_160_half_NZ2NZ",
        "TLOADSHAPE2DTest.1_8_4_16_32_1_9_4_16_32_80_256_int8_t_NZ2NZ",

        "TLOADSHAPE2DTest.1_1_1_59_119_1_1_1_59_124_59_120_int64_t_ND2ND",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float32, 1, 1, 1, 128,
                      128, 1, 1, 1, 128, 128, 128, 128, DataFormat['ND2NZ'].value),
        tmatmulParams(np.int8, np.int8, np.int32, 1, 1, 1, 128, 128,
                      1, 1, 1, 128, 128, 128, 128, DataFormat['ND2NZ'].value),
        tmatmulParams(np.float32, np.float32,  np.float32, 1, 1, 1, 128,
                      128, 1, 1, 1, 128, 128, 128, 128, DataFormat['ND2NZ'].value),
        tmatmulParams(np.float16, np.float16,  np.float32, 1, 1, 1, 64,
                      128, 1, 1, 1, 64, 128, 64, 128, DataFormat['DN2NZ'].value),
        tmatmulParams(np.float16, np.float16,  np.float32, 1, 1, 1, 63,
                      127, 1, 1, 1, 63, 127, 64, 128,  DataFormat['ND2NZ'].value),
        tmatmulParams(np.float32, np.float32,  np.float32, 1, 1, 1, 128,
                      128, 1, 1, 1, 128, 128,  128, 128,  DataFormat['ND2ND'].value),
        tmatmulParams(np.int8, np.int8,  np.int32, 1, 1, 1, 37, 126,
                      1, 1, 1, 37, 126, 37, 128, DataFormat['ND2ND'].value),

        tmatmulParams(np.float16, np.float16,  np.float32, 1, 1, 1, 33,
                      99, 1, 1, 1, 64, 128, 48, 112, DataFormat['ND2NZ'].value),
        tmatmulParams(np.int8, np.int8,  np.int32, 1, 1, 1, 59, 119,
                      1, 1, 1, 64, 128, 64, 128, DataFormat['ND2NZ'].value),

        tmatmulParams(np.float32, np.float32,  np.float32, 1, 1, 1, 51,
                      123, 1, 1, 1, 64, 128, 64, 128, DataFormat['DN2NZ'].value),
        tmatmulParams(np.float16, np.float16,  np.float32, 1, 1, 1, 63,
                      127, 1, 1, 1, 63, 127, 64, 128,  DataFormat['DN2NZ'].value),

        tmatmulParams(np.float32, np.float32,  np.float32, 1, 1, 1, 128,
                      128, 1, 1, 1, 128, 128, 128, 128, DataFormat['DN2DN'].value),
        tmatmulParams(np.int8, np.int8,  np.int32, 1, 1, 1, 37, 126,
                      1, 1, 1, 37, 126, 64, 126, DataFormat['DN2DN'].value),

        tmatmulParams(np.float16, np.float16,  np.float32, 1, 10, 8, 16,
                      16, 1, 11, 9, 16, 16, 128, 160, DataFormat['NZ2NZ'].value),
        tmatmulParams(np.int8, np.int8, np.int32, 1, 8, 4, 16, 32,
                      1, 9, 4, 16, 32, 80, 256, DataFormat['NZ2NZ'].value),

        tmatmulParams(np.int64, np.int64,  np.int64, 1, 1, 1, 59,
                      119, 1, 1, 1, 59, 124, 59, 120, DataFormat['ND2ND'].value),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
