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

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)


# Prefer in-tree googletest to avoid ABI mismatch with preinstalled libgtest
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/../../../../../build_tests/_deps/googletest-src" CACHE PATH "Path to googletest source")
if(EXISTS "${GTEST_SRC_DIR}/CMakeLists.txt")
    message(STATUS "Using in-tree googletest: ${GTEST_SRC_DIR}")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GTEST_SRC_DIR}" "${CMAKE_BINARY_DIR}/_deps/googletest" EXCLUDE_FROM_ALL)
else()
    find_package(GTest REQUIRED)
endif()

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

set(PTO_REPO_ROOT "${PROJECT_SOURCE_DIR}/../../../../../")

# IMPORTANT: for strict cross-checking we must build STs against the *repo* PTO headers,
# not the installed CANN copy. Keep repo include first.
include_directories(
    ${PTO_REPO_ROOT}/include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)
if (DEFINED ENV{ACCURATE_RSQRT})
    add_compile_definitions(ACCURATE_RSQRT)
endif ()
if (DEFINED ENV{ACCURATE_RECIP})
    add_compile_definitions(ACCURATE_RECIP)
endif ()

add_subdirectory(testcase)