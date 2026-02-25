# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

function(pto_comm_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -std=c++17
        -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init)
    target_include_directories(${NAME}_kernel PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../../include
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
        $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
        # $ENV{ASCEND_HOME_PATH}/pkg_inc/
        # $ENV{ASCEND_HOME_PATH}/pkg_inc/profiling/
        # $ENV{ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_directories(${NAME}_kernel PRIVATE
        ${SHMEM_HOME_PATH}/shmem/lib
        ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
        ${ASCEND_HOME_PATH}/lib64
    )
    target_link_libraries(${NAME}_kernel PRIVATE
        shmem
        runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS}
        -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../../include
        ${PROJECT_SOURCE_DIR}/../../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
    )
endfunction()


# Test case list - add new test directories here
# Aligned with SPEC: docs/isa/comm/README.md
set(ALL_TESTCASES
# Point-to-Point Communication (Synchronous)
tput
tget
# Signal-Based Synchronization
tnotify
twait
ttest
# Collective Communication
tgather
tscatter
treduce
tbroadcast
)

if(TEST_CASE IN_LIST ALL_TESTCASES)
message(STATUS "run : ${TEST_CASE}")
else()
message(FATAL_ERROR "not found TEST_CASE: ${TEST_CASE}, support case is : ${ALL_TESTCASES}")
endif()


foreach(TESTCASE ${ALL_TESTCASES})
if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE))
    add_subdirectory(${TESTCASE})
endif()
endforeach()
