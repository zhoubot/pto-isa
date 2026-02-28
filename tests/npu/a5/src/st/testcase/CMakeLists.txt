# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

function(pto_vec_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE 
        ${PROJECT_SOURCE_DIR}/../../../../../include
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)
    
    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../include
        ${PROJECT_SOURCE_DIR}/../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    )


    if(TARGET GTest::gtest)
        set(_GTEST_LIB GTest::gtest)
        set(_GTEST_MAIN GTest::gtest_main)
    else()
        set(_GTEST_LIB gtest)
        set(_GTEST_MAIN gtest_main)
    endif()
    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase ${_GTEST_LIB} ${_GTEST_MAIN} pthread
    )
endfunction()

function(pto_cube_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-cube -DREGISTER_BASE -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE 
        ${PROJECT_SOURCE_DIR}/../../../../../include
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)
    
    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../include 
        ${PROJECT_SOURCE_DIR}/../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    )


    if(TARGET GTest::gtest)
        set(_GTEST_LIB GTest::gtest)
        set(_GTEST_MAIN GTest::gtest_main)
    else()
        set(_GTEST_LIB gtest)
        set(_GTEST_MAIN gtest_main)
    endif()
    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase ${_GTEST_LIB} ${_GTEST_MAIN} pthread
    )
endfunction()

function(pto_mix_st NAME)
    add_library(${NAME}_kernel_mix SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel_mix PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310 -DREGISTER_BASE -std=c++17)

    target_include_directories(${NAME}_kernel_mix PRIVATE 
        ${PROJECT_SOURCE_DIR}/../../../../../include
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel_mix PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE 
        ${PROJECT_SOURCE_DIR}/../../../../../include
        ${PROJECT_SOURCE_DIR}/../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )


    if(TARGET GTest::gtest)
        set(_GTEST_LIB GTest::gtest)
        set(_GTEST_MAIN GTest::gtest_main)
    else()
        set(_GTEST_LIB gtest)
        set(_GTEST_MAIN gtest_main)
    endif()
    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel_mix
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase ${_GTEST_LIB} ${_GTEST_MAIN} pthread
    )
endfunction()

# 定义测试用例列表 testcase下如有新增用例目录需要在此加
set(ALL_TESTCASES
    tci
    tmatmul
    tmatmul_mx
    tload_mix
    tload_shape2d
    tload
    tload_mx_ND_DN
    tload_mx_NZ
    tload_mx_gmtensor
    tmrgsort
    ttrans
    tstore
    textract
    textract_compact
    trowsum
    trowmax
    trowmin
    tcolsum
    tcolmax
    tcolmin
    tsort32
    trowexpand
    trowexpandadd
    trowexpanddiv
    trowexpandmax
    trowexpandmin
    trowexpandmul
    trowexpandsub
    tgather
    tcvt
    tsub
    tadd
    tpartadd
    tfillpad
    tmin
    tmax
    tmins
    tmov
    tmov_mx
    tmul
    tdiv
    tcmps
    tmov_acc2vec
    tsel
    tsels
    trsqrt
    tsqrt
    texp
    tabs
    tlog
    trecip
    tstore_acc2gm
    tdivs
    tmuls
    tadds
    tgatherb
    tpartmin
    tpartmax
    tmov_vect
    texpands
    tmov_acc2mat
    tassign
    tmov_ub2l1
    tcmp
    tadd_tdiv
    tmul_tadds
    tsub_texp
    tmuls_trowsum
    trowexpand_tsqrt
    trowexpand_trowsum
    trowsum_trowexpand
    trowexpand_tdiv
    tscatter
    tneg
    tcolexpanddiv
    tcolexpandmul
    tcolexpandsub
    trowexpandbrcb
    ttril
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
