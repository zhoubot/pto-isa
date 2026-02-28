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
import shutil
from pathlib import Path
from typing import Dict, List, Optional

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that processes optional arguments.")
    parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-b","--build-folder", type=str, default="build_tests", help="Set the build folder path")
    parser.add_argument(
        "-c",
        "--compiler",
        required=False,
        help="Compiler for CPU-SIM (name in PATH or absolute path). Use 'auto' to pick a working default.",
        default="auto",
    )
    args = parser.parse_args()
    return args

def red(st):
    return f"\033[31m{st}\033[0m"
def green(st):
    return f"\033[32m{st}\033[0m"

def _resolve_cxx_candidates(compiler: str) -> List[str]:
    cands: List[str] = []
    if compiler and compiler != "auto":
        if "/" in compiler:
            p = Path(compiler)
            if p.exists():
                return [str(p)]
            raise FileNotFoundError(f"--compiler path not found: {compiler}")
        w = shutil.which(compiler)
        if w:
            return [w]
        raise FileNotFoundError(f"--compiler not found in PATH: {compiler}")

    env_cxx = os.environ.get("CXX")
    if env_cxx:
        if "/" in env_cxx and Path(env_cxx).exists():
            cands.append(env_cxx)
        else:
            w = shutil.which(env_cxx)
            if w:
                cands.append(w)

    # Try newer compilers first (CPU ST targets C++23).
    for cand in ("g++-14", "g++-13", "g++-12", "clang++", "g++"):
        w = shutil.which(cand)
        if w:
            cands.append(w)

    # De-dupe while preserving order.
    out: List[str] = []
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    if not out:
        raise FileNotFoundError("No C++ compiler found (tried g++-14/g++/clang++). Install one or pass --compiler.")
    return out

def _run(cmd: List[str], *, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None, verbose: bool = False) -> None:
    if verbose:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode == 0:
        return
    sys.stdout.write(p.stdout or "")
    raise subprocess.CalledProcessError(p.returncode, cmd)

def main():
    args=parse_arguments()
    try:
        os.mkdir(args.build_folder)
    except:
        pass
    tests_path = os.path.dirname(os.path.dirname(inspect.getfile(sys.modules[__name__])))+"/cpu/st/"
    build_dir = args.build_folder
    last_err: Optional[Exception] = None
    last_out: str = ""
    selected: Optional[str] = None
    for cxx in _resolve_cxx_candidates(args.compiler):
        env = dict(os.environ)
        env["CXX"] = cxx
        try:
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
            os.makedirs(build_dir, exist_ok=True)
            if args.verbose:
                print(f"[cpu-st] trying CXX={cxx}")
                _run(["cmake", "-S", tests_path, "-B", build_dir], env=env, verbose=True)
            else:
                p = subprocess.run(
                    ["cmake", "-S", tests_path, "-B", build_dir],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if p.returncode != 0:
                    last_out = p.stdout or ""
                    raise subprocess.CalledProcessError(p.returncode, ["cmake", "-S", tests_path, "-B", build_dir])
            selected = cxx
            break
        except Exception as e:
            last_err = e
            # Keep trying other candidates in auto mode.
            if args.compiler != "auto":
                raise
            # Heuristic: GCC too old for C++23 is common; try next compiler.
            continue

    if not selected:
        if last_out:
            sys.stdout.write(last_out)
        raise last_err if last_err else RuntimeError("failed to configure CPU ST (no working compiler found)")

    env = dict(os.environ)
    env["CXX"] = selected

    _run(["cmake", "--build", build_dir, "-j", str(os.cpu_count() or 8)], env=env, verbose=args.verbose)

    os.chdir(build_dir)
    py_files = glob.glob(f"{tests_path}/testcase/*/gen_data.py", recursive=False)
    for f in py_files:
        _run([sys.executable, f], verbose=args.verbose)

    os.chdir("bin")
    exe_files = sorted(glob.glob("./*", recursive=False))
    total_tests=0
    successful_tests=0
    for f in exe_files:
        if os.path.isdir(f):
            continue
        try:
            print(f"--- {f} ------------------------------------------------")
            total_tests += 1
            p = subprocess.run([f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            split_out = (p.stdout or "").splitlines()
            passed = [x for x in split_out if x.startswith("[  PASSED  ]")]
            failed = [x for x in split_out if x.startswith("[  FAILED  ]")]
            total = [x for x in split_out if x.startswith("[==========]") and "ran" in x]
            print(*passed)
            if p.returncode != 0 or failed:
                if failed:
                    print(*[red(x) for x in failed])
                else:
                    print(red("ERROR: test process returned non-zero"))
            else:
                successful_tests += 1
            print(*total)
            print()
        except Exception:
            print(red(f"ERROR: failed to run {f} test\n"))

    res = f"SUCCESSFULLY EXECUTED {successful_tests} OF {total_tests} TEST SUITES. FAILED:{total_tests-successful_tests}"
    print(green(res) if total_tests==successful_tests else red(res))

if __name__ == "__main__":
    main()
