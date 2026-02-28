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

np.random.seed(19)

def gen_golden_data(param):
    dtype = param.dtype
    shape = param.shape
    rank = param.rank
    size = shape[0] * shape[1] * shape[2] * shape[3] * shape[4]

    #Generate random input arrays
    input1 = np.random.randint(1, 10, size=[size]).astype(dtype)
    golden = np.tile(input1, rank)

    #Save the input and golden data to binary files
    input1.tofile("input.bin") 
    golden.tofile("golden.bin")


class TBroadCastParams:
    def __init__(self, dtype, shape, rank):
        self.dtype = dtype 
        self.shape = shape
        self.rank = rank


def generate_case_name(param, index):
    dtype_str = {
        np.float32: 'float', 
        np.float16: 'half', 
        np.int8: 'int8', 
        np.int32: 'int32', 
        np.int16: 'int16' 
    }[param.dtype]

    name = f"TBroadCastTest.case_{dtype_str}_{index}"
    return name

if __name__ == "__main__":
    #Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    testcases_dir = os.path.join(script_dir, "testcases")

    #Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TBroadCastParams(np.float32, (1, 2, 4, 64, 64), 5), 
        TBroadCastParams(np.int32, (1, 2, 4, 64, 64), 3), 
        TBroadCastParams(np.int16, (2, 2, 3, 64, 64), 2), 
        TBroadCastParams(np.float16, (1, 2, 1, 16, 256), 1)
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param, i + 1) 
        if not os.path.exists(case_name):
            os.makedirs(case_name) 
        original_dir = os.getcwd() 
        os.chdir(case_name) 
        gen_golden_data(param) 
        os.chdir(original_dir)
