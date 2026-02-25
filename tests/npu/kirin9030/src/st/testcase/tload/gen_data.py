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


if __name__ == "__main__":
   # 用例名称
    case_name_list = [
        "TLOADTest.case_float_GT_128_128_VT_128_128_BLK1", # 此名称需要和 TEST_F(TMATMULTest, xxxx)定义的名称一致
        "TLOADTest.case_float_GT_2_2_2_256_64_VT_256_64_BLK8", 
        "TLOADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX", 
        "TLOADTest.case_s16_GT_128_127_VT_128_128_BLK1_PADMAX", 
        "TLOADTest.case_u8_GT_128_127_VT_128_128_BLK1_PADMIN", 
        "TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_DYN", 
        "TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_STC", 
        "TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX", 
        "TLOADTest.case_int64_GT_128_128_VT_128_128_BLK1",
        "TLOADTest.case_uint64_GT_128_125_VT_128_128_BLK1_PADZERO",
        "TLOADTest.case_int64_GT_2_2_2_256_62_VT_256_64_BLK8_PADZERO",
        "TLOADTest.case_uint64_GT_2_2_2_256_64_VT_256_64_BLK8",
    ]
    
    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        pass
        os.chdir(original_dir)

    pass
    