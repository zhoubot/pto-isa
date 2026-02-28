# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import glob
import os
import sys
import inspect
import subprocess
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that processes optional arguments.")
    parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-b","--build-folder", type=str, default="build_tests", help="Set the build folder path")
    parser.add_argument("-c", "--compiler", required=False, help="Compiler for CPU-SIM (e.g. g++ or /usr/bin/g++)", default="g++")
    args = parser.parse_args()
    return args

def red(st):
    return f"\033[31m{st}\033[0m"
def green(st):
    return f"\033[32m{st}\033[0m"


def main():
    args=parse_arguments()
    cmd_suffix = "" if args.verbose else ">>/dev/null"
    try:
        os.mkdir(args.build_folder)
    except:
        pass
    tests_path = os.path.dirname(os.path.dirname(inspect.getfile(sys.modules[__name__])))+"/cpu/st/"
    build_dir = args.build_folder
    cxx = args.compiler
    if os.system(f"export CXX={cxx}; cmake -S {tests_path} -B {build_dir} {cmd_suffix} && cd {build_dir} && make -j8 {cmd_suffix}")==0:
        os.chdir(build_dir)
        py_files = glob.glob(f"{tests_path}/testcase/*/gen_data.py", recursive=False)
        for f in py_files:
            os.system(f"{sys.executable} {f} {cmd_suffix}")

        os.chdir("bin")
        exe_files = glob.glob("./*", recursive=False)
        total_tests=0
        successful_tests=0
        for f in exe_files:
            try:
                print(f"--- {f} ------------------------------------------------")
                total_tests+=1                
                ret = subprocess.check_output(f)
                split_out = str(ret).split("\\n")
                #print(split_out)
                passed = [x for x in split_out if x.startswith("[  PASSED  ]")]
                failed = [x for x in split_out if x.startswith("[  FAILED  ]")]
                total = [x for x in split_out if x.startswith("[==========]") and "ran" in x ]
                print(*passed)
                if len(failed)>0:
                    print(*[red(x) for x in failed])
                else:
                    successful_tests+=1
                print(*total)
                print()
                
            except:
                print(red(f"ERROR: failed to run {f} test\n"))

        res = f"SUCCESSFULLY EXECUTED {successful_tests} OF {total_tests} TEST SUITES. FAILED:{total_tests-successful_tests}"
        print(green(res) if total_tests==successful_tests else red(res))

if __name__ == "__main__":
    main()
