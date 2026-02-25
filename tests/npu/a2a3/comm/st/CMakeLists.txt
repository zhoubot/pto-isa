# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(pto_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# shmem 安装根目录（需在环境中提供）
if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please export SHMEM_HOME_PATH to shmem install root.")
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
    message(STATUS "SHMEM_HOME_PATH: ${SHMEM_HOME_PATH}")
endif()

if(DEFINED ENV{SHMEM_BACKEND})
    set(SHMEM_BACKEND $ENV{SHMEM_BACKEND})
endif()
if(NOT DEFINED SHMEM_BACKEND)
    # 自动检测: 优先查找新版 shmem.h
    if(EXISTS "${SHMEM_HOME_PATH}/shmem/include/shmem.h")
        set(SHMEM_BACKEND "CANN_SHMEM")
    else()
        message(FATAL_ERROR "Cannot find shmem.h or shmem_api.h in ${SHMEM_HOME_PATH}/shmem/include/")
    endif()
endif()
message(STATUS "SHMEM_BACKEND: ${SHMEM_BACKEND}")


set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

find_package(GTest REQUIRED)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
    )

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

# 公共 include 路径
include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../../include
    ${PROJECT_SOURCE_DIR}/../../../../..
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# 根据 SHMEM 版本设置不同的 include 路径和编译宏

add_compile_definitions(CANN_SHMEM)
include_directories(
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/shmem/include/device
    ${SHMEM_HOME_PATH}/../src/device
)



add_subdirectory(testcase)