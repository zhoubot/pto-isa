#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import ctypes
import os
import subprocess

import torch

PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]
NPU_ARCH = os.environ.get("NPU_ARCH", "dav-2201")

BLOCK_DIM = 24  # hard-coded in gemm_kernel.cpp


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 120) -> str:

    # output .so next to kernel_cpp
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "gemm_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        f"--npu-arch={NPU_ARCH}",
        "-O2",
        "-std=c++17",
        # "-Wno-ignored-attributes", # suppress warnings from PTO headers
        f"-I{PTO_LIB_PATH}/include",
    ]

    cmd = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:\n", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def load_lib(lib_path: str, check_type: bool = True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    if check_type:
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # out
            ctypes.c_void_p,  # src0
            ctypes.c_void_p,  # src1
        ]
        lib.call_kernel.restype = None

    default_block_dim = BLOCK_DIM
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def gemm(
        c: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        block_dim: int = default_block_dim,
        stream_ptr=default_stream_ptr,
    ):
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(c),
            torch_to_ctypes(a),
            torch_to_ctypes(b),
        )

    return gemm


def jit_compile_gemm(verbose: bool = False, clean_up: bool = True):
    """
    Builds gemm_kernel.cpp into gemm_jit.so,
    loads call_kernel, and returns gemm(c, a, b).
    """
    kernel_cpp = "gemm_kernel.cpp"

    lib_path = compile_cpp(kernel_cpp, verbose=verbose)
    func = load_lib(lib_path)

    if clean_up:
        try:
            os.remove(lib_path)
        except OSError:
            pass

    return func
