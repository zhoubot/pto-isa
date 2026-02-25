#Copyright(c) 2026 Huawei Technologies Co., Ltd.
#This program is free software, you can redistribute it and / or modify it under the terms and conditions of
#CANN Open Software License Agreement Version 2.0(the "License").
#Please refer to the License for details.You may not use this file except in compliance with the License.
#THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO NON - INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#See LICENSE in the root of the software repository for the full text of the License.
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

import os 
import struct
import numpy as np 

np.random.seed(19)

op_types = [
    "Add",
    "Set"
]

def gen_golden_data(param):
    op_type = param.operation_type
    
    shared_memory = np.random.randint(1, 100, size=[1]).astype(np.int32)
    signal = np.random.randint(1, 100, size=[1]).astype(np.int32)

    if op_type == "Add":
        golden = shared_memory + signal
    else:
        golden = signal

    signal.tofile("signal.bin")
    shared_memory.tofile("sharedMemory.bin")
    golden.tofile("golden.bin")


class TNotifyParams:
    def __init__(self, operation_type, case_name):
        self.operation_type = operation_type
        self.case_name = case_name

if __name__ == "__main__":
    #Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    testcases_dir = os.path.join(script_dir, "testcases")

    #Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TNotifyParams("Add", "TNOTIFY.case1"), 
        TNotifyParams("Set", "TNOTIFY.case2")
    ]

    for i, param in enumerate(case_params_list):
        case_name = param.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name) 
        original_dir = os.getcwd() 
        os.chdir(case_name) 
        gen_golden_data(param) 
        os.chdir(original_dir)
