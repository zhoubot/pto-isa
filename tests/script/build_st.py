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
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)

def run_command(command, cwd=None, check=True):
    try:
        log(f"run command: {' '.join(command)}")
        subprocess.run(
            command,
            cwd=cwd,
            check=check,
            stdout=None,
            stderr=None,
            text=True
        )
        return ""
    except subprocess.CalledProcessError as e:
        log(f"run command failed with return code {e.returncode}")
        raise


def resolve_ascend_home() -> str:
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        candidate = Path.home() / "Ascend" / "ascend-toolkit" / "latest"
        if candidate.exists():
            ascend_home = str(candidate)
            os.environ["ASCEND_HOME_PATH"] = ascend_home
        else:
            raise EnvironmentError(
                "ASCEND_HOME_PATH is not set and default ~/Ascend/ascend-toolkit/latest does not exist"
            )
    if not Path(ascend_home).exists():
        raise EnvironmentError(f"ASCEND_HOME_PATH does not exist: {ascend_home}")
    return ascend_home


def source_setenv(ascend_home: str) -> None:
    setenv_path = Path(ascend_home) / "bin" / "setenv.bash"
    if not setenv_path.exists():
        log(f"warning: not found {setenv_path}")
        return

    log(f"run env shell: {setenv_path}")
    result = subprocess.run(
        ["bash", "-lc", f"source '{setenv_path}' >/dev/null 2>&1 && env -0"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to source {setenv_path} (rc={result.returncode}): {result.stderr.decode(errors='ignore')}"
        )
    for item in result.stdout.split(b"\x00"):
        if not item:
            continue
        if b"=" not in item:
            continue
        key, _, value = item.partition(b"=")
        key_str = key.decode(errors="ignore")
        if not key_str:
            continue
        os.environ[key_str] = value.decode(errors="ignore")

    os.environ["ASCEND_HOME_PATH"] = ascend_home


def set_env_variables(run_mode: str) -> None:
    ascend_home = resolve_ascend_home()
    source_setenv(ascend_home)

def resolve_bisheng() -> str:
    return shutil.which("bisheng") or "bisheng"


def build_project(run_mode, soc_version, testcase = "all"):
    original_dir = os.getcwd()
    # 清理并创建build目录
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        bisheng = resolve_bisheng()
        cmake_cmd = [
            "cmake",
            f"-DCMAKE_C_COMPILER={bisheng}",
            f"-DCMAKE_CXX_COMPILER={bisheng}",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={soc_version}",
        ]
        # When TEST_CASE is omitted, CMakeLists adds *all* testcases.
        if testcase and testcase.lower() != "all":
            cmake_cmd.append(f"-DTEST_CASE={testcase}")
        cmake_cmd.append("..")

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
        log("compile process:\n" + result.stdout)

    except subprocess.CalledProcessError as e:
        log(f"build failed: {e.stdout}")
        raise
    finally:
        os.chdir(original_dir)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="执行st脚本")
    parser.add_argument("-r", "--run-mode", required=True, help="运行模式（如 sim or npu)")
    parser.add_argument("-v", "--soc-version", required=True, help="SOC版本 只支持 a3 or a5")
    parser.add_argument("-t", "--testcase", required=True, help="需要执行的用例")
    parser.add_argument(
        "-g",
        "--gtest_filter",
        required=False,
        help="可选 需要执行的具体case名（build_st.py 中忽略；仅为与 run_st.py 参数兼容）",
    )

    args = parser.parse_args()
    default_soc_version = "Ascend910B1"
    if args.soc_version == "a5":
        default_soc_version = "Ascend910_9599"

    original_dir = os.getcwd()
    try:
        # 获取当前脚本（run_st.py）的绝对路径
        script_path = os.path.abspath(__file__)

        if args.soc_version == "a3":
            target_dir = os.path.dirname(os.path.dirname(script_path))
            target_dir = target_dir + "/npu/a2a3/src/st"
        else:  # a5
            target_dir = os.path.dirname(os.path.dirname(script_path))
            target_dir = target_dir + "/npu/a5/src/st"

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        # 设置环境变量（确保 ccec/acl 等可用）
        set_env_variables(args.run_mode)

        # 执行构建
        build_project(args.run_mode, default_soc_version, args.testcase)

    except Exception as e:
        log(f"run failed: {str(e)}")
        sys.exit(1)
    os.chdir(original_dir)

if __name__ == "__main__":
    main()
