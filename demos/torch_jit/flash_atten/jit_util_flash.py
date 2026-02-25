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

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _npu_arch_flag() -> str:
    return os.environ.get("NPU_ARCH", "dav-2201").strip()


def compile_flash(kernel_cpp: str, verbose: bool = False, timeout: int = 300) -> str:
    """
    Compile a FlashAttention/TFA kernel cpp into a shared library with a call_kernel symbol.

    Output library is placed next to kernel_cpp, named: flash_jit.so
    """
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "flash_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        f"--npu-arch={_npu_arch_flag()}",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{PTO_LIB_PATH}/kernels/manual/a2a3/flash_atten",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
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


def load_flash_lib(lib_path: str, check_type: bool = True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    if check_type:
        lib.call_kernel.argtypes = [
            ctypes.c_void_p,  # stream
            ctypes.c_int,  # head_size
            ctypes.c_int,  # s0
            ctypes.c_int,  # s1
            ctypes.c_bool,  # is_causal
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o_out
            ctypes.c_void_p,  # qk_out fp32
            ctypes.c_void_p,  # p_out fp16
            ctypes.c_void_p,  # p_out fp32
            ctypes.c_void_p,  # pv_out tiles
            ctypes.c_void_p,  # global_sum
            ctypes.c_void_p,  # exp_max
            ctypes.c_void_p,  # o_parts
        ]
        lib.call_kernel.restype = None

    default_causal = False
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def flash(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o_out: torch.Tensor,
        out_device: torch.Tensor,
        xexp_device: torch.Tensor,
        p_out_fp32_device: torch.Tensor,
        out_2device: torch.Tensor,
        g_sum_device: torch.Tensor,
        exp_max_device: torch.Tensor,
        o_parts_device: torch.Tensor,
        stream_ptr=default_stream_ptr,
        is_causal=default_causal,
    ):

        lib.call_kernel(
            stream_ptr,
            q.shape[1],  # head_size
            q.shape[0],  # s0
            k.shape[0],  # s1
            is_causal,
            torch_to_ctypes(q),
            torch_to_ctypes(k),
            torch_to_ctypes(v),
            torch_to_ctypes(o_out),
            torch_to_ctypes(out_device),  # qk_out fp32
            torch_to_ctypes(xexp_device),  # p_out fp16
            torch_to_ctypes(p_out_fp32_device),  # p_out fp32
            torch_to_ctypes(out_2device),  # pv_out tiles
            torch_to_ctypes(g_sum_device),  # global_sum
            torch_to_ctypes(exp_max_device),  # exp_max
            torch_to_ctypes(o_parts_device),  # o snapshots
        )

    return flash


def jit_compile_flash(
    verbose: bool = False,
    clean_up: bool = True,
    kernel_cpp: str = "fa_kernel.cpp",
):
    """
    Builds the Flash/TFA kernel cpp into flash_jit.so,
    loads call_kernel, and returns flash(...) wrapper.

    Expected C++ export:
        extern "C" void call_kernel(uint32_t blockDim, void* stream, ...)
    """
    lib_path = compile_flash(kernel_cpp, verbose=verbose)
    func = load_flash_lib(lib_path)

    if clean_up:
        try:
            os.remove(lib_path)
        except OSError:
            pass

    return func
