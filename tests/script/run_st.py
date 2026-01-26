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
from typing import List, Optional, Set


def log(msg: str) -> None:
    print(msg, flush=True)

def _is_truthy(v: str) -> bool:
    return v in ("1", "true", "True", "yes", "YES", "on", "ON")

def run_command(command, cwd=None, check=True, *, verbose: bool = False):
    try:
        if verbose:
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

        p = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if check and p.returncode != 0:
            if p.stdout:
                log(p.stdout)
            raise subprocess.CalledProcessError(p.returncode, command, output=p.stdout)
        return p.stdout or ""
    except subprocess.CalledProcessError as e:
        log(f"run command failed with return code {e.returncode}")
        raise

def _split_paths(path_value: str) -> List[str]:
    return [p for p in path_value.split(":") if p]


def _dedupe_paths(paths: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


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
    # Use a login shell; parse via NUL separators to avoid newline issues.
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

    # Ensure the caller-specified ASCEND_HOME_PATH survives setenv overrides.
    os.environ["ASCEND_HOME_PATH"] = ascend_home


def resolve_simulator_lib_path(ascend_home: str, soc_version: str) -> str:
    candidates = [
        Path(ascend_home) / "tools" / "simulator" / soc_version / "lib",
        Path(ascend_home) / "tools" / "simulator" / soc_version / "lib64",
        Path(ascend_home) / "simulator" / soc_version / "lib",
        Path(ascend_home) / "simulator" / soc_version / "lib64",
        Path(ascend_home) / "aarch64-linux" / "simulator" / soc_version / "lib",
        Path(ascend_home) / "aarch64-linux" / "simulator" / soc_version / "lib64",
        Path(ascend_home) / "x86_64-linux" / "simulator" / soc_version / "lib",
        Path(ascend_home) / "x86_64-linux" / "simulator" / soc_version / "lib64",
    ]
    for p in candidates:
        if p.is_dir():
            return str(p)
    raise EnvironmentError(
        f"cannot find simulator lib dir for SOC={soc_version} under ASCEND_HOME_PATH={ascend_home}"
    )


def resolve_runtime_stub_path(ascend_home: str) -> str:
    candidates = [
        Path(ascend_home) / "runtime" / "lib64" / "stub",
        Path(ascend_home) / "acllib" / "lib64" / "stub",
    ]
    for p in candidates:
        if p.is_dir():
            return str(p)
    raise EnvironmentError(f"cannot find runtime stub dir under ASCEND_HOME_PATH={ascend_home}")

def resolve_bisheng() -> str:
    # ST builds rely on Ascend CCE compilation; prefer the toolchain-provided `bisheng`.
    # CMake only honors compiler selection at *configure time*, so pass it via -DCMAKE_*_COMPILER.
    return shutil.which("bisheng") or "bisheng"


def set_env_variables(run_mode: str, soc_version: str) -> None:
    ascend_home = resolve_ascend_home()
    source_setenv(ascend_home)

    if run_mode != "sim":
        return

    sim_lib = resolve_simulator_lib_path(ascend_home, soc_version)
    ld_paths = _split_paths(os.environ.get("LD_LIBRARY_PATH", ""))

    # NOTE: On some toolkit builds, putting runtime stub libs first can crash during
    # static initialization (observed in libnnopbase.so). Default to *not* forcing
    # runtime stubs, and only enable them if explicitly requested.
    use_runtime_stub = os.environ.get("PTO_USE_RUNTIME_STUB", "0") == "1"
    if use_runtime_stub:
        stub_lib = resolve_runtime_stub_path(ascend_home)
        new_ld = _dedupe_paths([sim_lib, stub_lib] + ld_paths)
        log(f"runtime stub:  {stub_lib} (PTO_USE_RUNTIME_STUB=1)")
    else:
        new_ld = _dedupe_paths([sim_lib] + ld_paths)
        log("runtime stub:  (disabled; set PTO_USE_RUNTIME_STUB=1 to enable)")
    os.environ["LD_LIBRARY_PATH"] = ":".join(new_ld)
    log(f"simulator lib: {sim_lib}")

def build_project(run_mode, soc_version, testcase = "all", debug_enable = False):
    original_dir = os.getcwd()
    # 清理并创建build目录
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        verbose = _is_truthy(os.environ.get("PTO_ST_VERBOSE", "")) or debug_enable
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
        if debug_enable :
            cmake_cmd.append("-DDEBUG_MODE=ON")

        cmake_p = subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if cmake_p.returncode != 0:
            log("cmake failed:")
            if cmake_p.stdout:
                log(cmake_p.stdout)
            raise subprocess.CalledProcessError(cmake_p.returncode, cmake_cmd, output=cmake_p.stdout)

        # make_cmd = ["make", "VERBOSE=1"] # print compile log for debug
        make_cmd = ["make"]
        cpu_count = os.cpu_count() or 4
        make_cmd.extend(["-j", str(cpu_count)])

        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            log("build failed:")
            if result.stdout:
                log(result.stdout)
            raise subprocess.CalledProcessError(result.returncode, make_cmd, output=result.stdout)
        if verbose and result.stdout:
            log("compile process:\n" + result.stdout)

    except subprocess.CalledProcessError as e:
        log(f"build failed: {e.stdout}")
        raise
    finally:
        os.chdir(original_dir)

def run_gen_data(golden_path):
    original_dir = os.getcwd()
    try:
        verbose = _is_truthy(os.environ.get("PTO_ST_VERBOSE", ""))
        cmd = ["cp", golden_path, "build/gen_data.py"]
        run_command(cmd, verbose=verbose)

        build_dir = "build/"
        os.chdir(build_dir)

        golden_gen_cmd = [sys.executable, "gen_data.py"]
        output = run_command(golden_gen_cmd, verbose=verbose)
        if verbose and output:
            log(output)
    except Exception as e:
        log(f"gen golden failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

def _summarize_set_wait_flags(log_dir: Path) -> None:
    if not log_dir.exists():
        log(f"[sim log check] CAMODEL_LOG_PATH not found: {log_dir}")
        return

    set_count = 0
    wait_count = 0
    samples: List[str] = []
    for f in sorted(log_dir.rglob("*.instr_log.dump")):
        try:
            with f.open("r", errors="ignore") as fp:
                for line in fp:
                    if "SET_FLAG" in line:
                        set_count += 1
                        if len(samples) < 6:
                            samples.append(f"{f.name}: {line.strip()}")
                    if "WAIT_FLAG" in line:
                        wait_count += 1
                        if len(samples) < 6:
                            samples.append(f"{f.name}: {line.strip()}")
        except OSError:
            continue

    log(f"[sim log check] log dir: {log_dir}")
    log(f"[sim log check] SET_FLAG lines: {set_count}, WAIT_FLAG lines: {wait_count}")
    if samples:
        log("[sim log check] sample lines:")
        for s in samples:
            log("  " + s)

def run_binary(testcase, run_mode, args="all"):
    original_dir = os.getcwd()
    try:
        verbose = _is_truthy(os.environ.get("PTO_ST_VERBOSE", ""))
        logs_on_pass = _is_truthy(os.environ.get("PTO_ST_LOGS", "")) or verbose
        build_dir = "build/bin/"
        os.chdir(build_dir)

        if args != "all":
            log_dir: Optional[Path] = None
            if run_mode == "sim":
                # Default: avoid heavy simulator dumps on passing runs.
                # Enable with PTO_ST_LOGS=1, or we will auto-enable on failure.
                if logs_on_pass:
                    log_dir = (Path.cwd().parent / args).resolve()
                    log_dir.mkdir(parents=True, exist_ok=True)
                    os.environ["CAMODEL_LOG_PATH"] = str(log_dir)
            single_case = "--gtest_filter=" + args
            cmd = ["./" + testcase, single_case]
            log(f"run single testcase : {args}")
            try:
                output = run_command(cmd, verbose=verbose)
                if verbose and output:
                    log(output)
            except Exception:
                # Re-run once with simulator logs enabled to help debug.
                if run_mode == "sim" and log_dir is None:
                    log_dir = (Path.cwd().parent / args).resolve()
                    log_dir.mkdir(parents=True, exist_ok=True)
                    os.environ["CAMODEL_LOG_PATH"] = str(log_dir)
                    log(f"[sim] re-run with CAMODEL_LOG_PATH={log_dir}")
                    run_command(cmd, verbose=True)
                    _summarize_set_wait_flags(log_dir)
                raise
            if run_mode == "sim" and log_dir is not None:
                _summarize_set_wait_flags(log_dir)
        else:  # all
            cmd = ["./" + testcase]
            log(f"run testcase : {testcase}")
            output = run_command(cmd, verbose=verbose)
            if verbose and output:
                log(output)

    except Exception as e:
        log(f"run binary failed: {e}")
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
        else:  # a5
            target_dir = os.path.dirname(os.path.dirname(script_path))
            target_dir = target_dir + "/npu/a5/src/st"

        log(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        # 设置环境变量
        set_env_variables(args.run_mode, default_soc_version)

        # 执行构建
        build_project(args.run_mode, default_soc_version, args.testcase, args.debug_enable)

        if args.testcase.lower() == "all":
            tc_root = Path("testcase")
            testcases = sorted(
                p.name for p in tc_root.iterdir() if p.is_dir() and (p / "gen_data.py").exists()
            )
            if not testcases:
                raise RuntimeError(f"no testcases found under {tc_root.resolve()}")
            log(f"run all testcases: {len(testcases)}")
            for tc in testcases:
                golden_path = str(Path("testcase") / tc / "gen_data.py")
                run_gen_data(golden_path)
                run_binary(tc, args.run_mode, default_cases)
        else:
            # 生成标杆
            golden_path = "testcase/" + args.testcase + "/gen_data.py"
            run_gen_data(golden_path)

            # 执行二进制文件
            run_binary(args.testcase, args.run_mode, default_cases)

    except Exception as e:
        log(f"run failed: {str(e)}")
        sys.exit(1)
    os.chdir(original_dir)

if __name__ == "__main__":
    main()
