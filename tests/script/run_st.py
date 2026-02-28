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

import os
import sys
import subprocess
import shutil
import argparse

def run_command(command, cwd=None, check=True):
    try:
        print(f"run command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=check,
            stdout=None,
            stderr=None,
            text=True
        )
        return ""
    except subprocess.CalledProcessError as e:
        print(f"run command failed with return code {e.returncode}")
        raise

def set_env_variables(run_mode, soc_version):
    if run_mode == "sim":
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_lib_path:
            filtered_paths = [
                path for path in ld_lib_path.split(':')
                if '/runtime/lib64' not in path
            ]
            new_ld_lib = ':'.join(filtered_paths)
            os.environ["LD_LIBRARY_PATH"] = new_ld_lib

        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            raise EnvironmentError("ASCEND_HOME_PATH is not set")

        os.environ["LD_LIBRARY_PATH"] = f"{ascend_home}/runtime/lib64/stub:{os.environ.get('LD_LIBRARY_PATH', '')}"

        setenv_path = os.path.join(ascend_home, "bin", "setenv.bash")
        if os.path.exists(setenv_path):
            print(f"run env shell: {setenv_path}")
            result = subprocess.run(
                f"source {setenv_path} && env",
                shell=True,
                executable=shutil.which("bash") or "bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        else:
            print(f"warning: not found {setenv_path}")

        simulator_lib_path = os.path.join(ascend_home, "tools", "simulator", soc_version, "lib")
        os.environ["LD_LIBRARY_PATH"] = f"{simulator_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

def build_project(run_mode, soc_version, testcase = "all", debug_enable = False):
    original_dir = os.getcwd()
    # 清理并创建build目录
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        cmake_cmd = [
            "cmake",
            "-DCMAKE_C_COMPILER=bisheng",
            "-DCMAKE_CXX_COMPILER=bisheng",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={soc_version}",
            f"-DTEST_CASE={testcase}",
            ".."
        ]
        if debug_enable :
            cmake_cmd.append("-DDEBUG_MODE=ON")

        subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # make_cmd = ["make", "VERBOSE=1"] # print compile log for debug
        make_cmd = ["make"]
        cpu_count = os.cpu_count() or 4
        make_cmd.extend(["-j", str(cpu_count)])

        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("compile process:\n", result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"build failed: {e.stdout}")
        raise
    finally:
        os.chdir(original_dir)

def run_gen_data(golden_path):
    original_dir = os.getcwd()
    try:
        cmd = ["cp", golden_path, "build/gen_data.py"]
        run_command(cmd)

        build_dir = "build/"
        os.chdir(build_dir)

        gloden_gen_cmd = [sys.executable, "gen_data.py"]
        output = run_command(gloden_gen_cmd)
        print(output)
    except Exception as e:
        print(f"gen golden failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

def run_binary(testcase, run_mode, args="all"):
    original_dir = os.getcwd()
    try:
        build_dir = "build/bin/"
        os.chdir(build_dir)

        if args != "all":
            if run_mode == "sim":
                os.environ["CAMODEL_LOG_PATH"] = f"../{args}"
            single_case = "--gtest_filter=" + args
            cmd = ["./" + testcase, single_case]
            print(f"run single testcase : {args}")
            output = run_command(cmd)
            print(output)
        else : # all
            cmd = ["./" + testcase]
            print(f"run testcase : {testcase}")
            output = run_command(cmd)
            print(output)

    except Exception as e:
        print(f"run binary failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="执行st脚本")
    parser.add_argument("-r", "--run-mode", required=True, help="运行模式（如 sim or npu)")
    parser.add_argument("-v", "--soc-version", required=True, help="SOC版本 只支持 a3 or a5")
    parser.add_argument("-t", "--testcase", required=True, help="需要执行的用例")
    parser.add_argument("-g", "--gtest_filter", required=False, help="可选 需要执行的具体case名")
    parser.add_argument("-d", "--debug-enable", action='store_true', help="开启debug检查")

    args = parser.parse_args()
    default_soc_version = "Ascend910B1"
    if args.soc_version == "a5":
        default_soc_version = "Ascend910_9599"
    default_cases = "all"
    if args.gtest_filter != None:
        default_cases = args.gtest_filter

    original_dir = os.getcwd()
    try:
        # 获取当前脚本（run_st.py）的绝对路径
        script_path = os.path.abspath(__file__)

        if args.soc_version == "a3":
            target_dir = os.path.dirname(os.path.dirname(script_path))
            target_dir = target_dir + "/npu/a2a3/src/st"
        else : # a5
            target_dir = os.path.dirname(os.path.dirname(script_path))
            target_dir = target_dir + "/npu/a5/src/st"

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        # 设置环境变量
        set_env_variables(args.run_mode, default_soc_version)

        # 执行构建
        build_project(args.run_mode, default_soc_version, args.testcase, args.debug_enable)

        # 生成标杆
        golden_path = "testcase/" + args.testcase + "/gen_data.py"
        run_gen_data(golden_path)

        # 执行二进制文件
        run_binary(args.testcase, args.run_mode, default_cases)

    except Exception as e:
        print(f"run failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    os.chdir(original_dir)

if __name__ == "__main__":
    main()