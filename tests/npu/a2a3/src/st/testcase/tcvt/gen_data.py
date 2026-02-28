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

def gen_golden(case_name, param):
    srctype = param.srctype
    dsttype = param.dsttype
    m, n = param.m, param.n

    # Generate input data with reasonable ranges
    if srctype == np.float32 or srctype == np.float16:
        # Floating point: range [-100, 100]
        x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.int8:
        # int8: full range [-128, 127]
        x1_gm = np.random.randint(-128, 128, [m, n]).astype(srctype)
    elif srctype == np.uint8:
        # uint8: full range [0, 255]
        x1_gm = np.random.randint(0, 256, [m, n]).astype(srctype)
    elif srctype == np.int16:
        # int16: reasonable range [-1000, 1000]
        x1_gm = np.random.randint(-1000, 1000, [m, n]).astype(srctype)
    elif srctype == np.uint16:
        # uint16: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int32:
        # int32: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    elif srctype == np.uint32:
        # uint32: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int64:
        # int64: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    else:
        # Default: signed int range
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)

    # Apply rounding mode for conversions
    mode = param.mode
    
    # Perform conversion first
    if np.issubdtype(srctype, np.floating):
        if np.issubdtype(dsttype, np.integer):
            # Floating point to integer conversion
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        elif srctype == np.float32 and dsttype == np.float32:
            # FP32 to FP32 conversion - apply rounding to integer values but keep as float
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        else:
            # Other float to float conversions - no rounding applied
            converted_golden = x1_gm
    else:
        # Integer to any type conversion
        converted_golden = x1_gm

    # Clamp the result to the destination type's representable range
    if np.issubdtype(dsttype, np.integer):
        info = np.iinfo(dsttype)
        # Clip in float64 to avoid overflow/casting issues (e.g. int64 -> int32)
        golden = np.clip(converted_golden.astype(np.float64), float(info.min), float(info.max)).astype(dsttype)
    elif np.issubdtype(dsttype, np.floating):
        info = np.finfo(dsttype)
        golden = np.clip(converted_golden.astype(np.float64), info.min, info.max).astype(dsttype)
    else:
        golden = converted_golden.astype(dsttype)
            
    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")
                
class tcvtParams:
    def __init__(self, srctype, dsttype, m, n, mode):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode

if __name__ == "__main__":
    # Type conversion pairs: (name_suffix, source_type, destination_type)
    type_pairs = [
        # FP32 Source
        ("fp32_fp32", np.float32, np.float32),
        ("fp32_fp16", np.float32, np.float16),
        ("fp32_int32", np.float32, np.int32),
        ("fp32_int16", np.float32, np.int16),
        ("fp32_int64", np.float32, np.int64),
        
        # FP16 Source
        ("fp16_fp32", np.float16, np.float32),
        ("fp16_int32", np.float16, np.int32),
        ("fp16_int16", np.float16, np.int16),
        ("fp16_int8", np.float16, np.int8),
        ("fp16_uint8", np.float16, np.uint8),

        # INT32 Source
        ("int32_fp32", np.int32, np.float32),
        ("int32_int16", np.int32, np.int16),
        ("int32_int64", np.int32, np.int64),

        # INT16 Source
        ("int16_fp16", np.int16, np.float16),
        ("int16_fp32", np.int16, np.float32),

        # INT8 Source
        ("int8_fp16", np.int8, np.float16),

        # UINT8 Source
        ("uint8_fp16", np.uint8, np.float16),

        # INT64 Source
        ("int64_fp32", np.int64, np.float32),
        ("int64_int32", np.int64, np.int32),
    ]

    # Different shape configurations (m, n)
    shapes = [
        (2, 128),
        (2, 32),
        (1, 64),
        (4, 64),
    ]

    case_name_list = []
    case_params_list = []

    # Generate test cases for each type pair and shape combination
    for type_name, src, dst in type_pairs:
        for m, n in shapes:
            case_name = f"case_{type_name}_{m}x{n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i])

        os.chdir(original_dir)