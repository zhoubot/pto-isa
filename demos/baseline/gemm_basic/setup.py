#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import multiprocessing
import os
import shutil
import stat
import subprocess
from distutils.version import LooseVersion

import torch
import torch_npu
from setuptools import find_packages, setup
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from torch_npu.utils.cpp_extension import NpuExtension


BUILD_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def which(filename):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for directory in path:
        fname = os.path.join(directory, filename)
        if os.access(fname, os.F_OK | os.X_OK) and not os.path.isdir(fname):
            return fname
    return None


def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')

    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.18.0"):
        return 'cmake3'
    if cmake is not None and _get_version(cmake) >= LooseVersion("3.18.0"):
        return 'cmake'
    raise RuntimeError('no cmake or cmake3 with version >= 3.18.0 found')


class CPPLibBuild(build_clib, object):
    def run(self):
        cmake = get_cmake_command()
        self.cmake = cmake

        build_py = self.get_finalized_command("build_py")
        extension_dir = os.path.join(BASE_DIR, build_py.build_lib, build_py.get_package_dir("op_extension"))

        build_dir = os.path.join(BASE_DIR, "build")
        build_type_dir = os.path.join(build_dir)
        output_lib_path = os.path.join(build_type_dir, "lib")
        os.makedirs(build_type_dir, exist_ok=True)
        os.chmod(build_type_dir, mode=BUILD_PERMISSION)
        os.makedirs(output_lib_path, exist_ok=True)

        self.build_lib = os.path.relpath(os.path.join(build_dir))
        self.build_temp = os.path.relpath(build_type_dir)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            '-DTORCH_PATH=' + os.path.realpath(os.path.dirname(torch.__file__)),
            '-DTORCH_NPU_PATH=' + os.path.realpath(os.path.dirname(torch_npu.__file__)),
        ]
        cmake_args.append('-DGLIBCXX_USE_CXX11_ABI=' + ('1' if torch.compiled_with_cxx11_abi() else '0'))

        max_jobs = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))
        build_args = ['-j', max_jobs]

        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        for base_dir, dirs, files in os.walk(build_type_dir):
            for dir_name in dirs:
                os.chmod(os.path.join(base_dir, dir_name), mode=BUILD_PERMISSION)
            for file_name in files:
                os.chmod(os.path.join(base_dir, file_name), mode=BUILD_PERMISSION)

        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)
        dst_dir = os.path.join(extension_dir, 'lib')
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(output_lib_path, dst_dir)


class Build(build_ext, object):
    def run(self):
        self.run_command('build_clib')
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, "build/temp"))
        self.library_dirs.append(os.path.relpath(os.path.join(BASE_DIR, "build/lib")))
        super(Build, self).run()


setup(
    name='op_extension',
    description='PTO GEMM (CUBE) op extension for torch_npu',
    packages=find_packages(),
    ext_modules=[NpuExtension("op_extension._C", sources=[])],
    cmdclass={
        'build_clib': CPPLibBuild,
        'build_ext': Build,
    },
)
