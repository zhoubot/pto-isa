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
project(pto_test_cpu_sim)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-D__CPU_SIM)

# GTest is required for CPU STs.
# Prefer building from source to avoid ABI mismatch with preinstalled libgtest
# (and to match the selected C++ compiler).
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/../../../build_tests/_deps/googletest-src" CACHE PATH "Path to googletest source")
if(EXISTS "${GTEST_SRC_DIR}/CMakeLists.txt")
    message(STATUS "Using in-tree googletest: ${GTEST_SRC_DIR}")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GTEST_SRC_DIR}" "${CMAKE_BINARY_DIR}/_deps/googletest" EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    message(STATUS "Fetching googletest via FetchContent")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

if (NOT TARGET GTest::gtest)
    add_library(GTest::gtest ALIAS gtest)
endif()
if (NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
endif()
    if (NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -std=c++23
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_options(-O2)
endif()

# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
if (UNIX AND NOT APPLE)
    add_link_options(
        -Wl,-z,relro
        -Wl,-z,now
    )
endif()


set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

add_subdirectory(testcase)
