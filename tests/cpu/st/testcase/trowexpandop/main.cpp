/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TROWEXPANDOPTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T, int kTRows_, int kTCols_>
void LaunchTROWEXPANDDIV(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_>
void LaunchTROWEXPANDMUL(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_>
void LaunchTROWEXPANDSUB(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_>
void LaunchTROWEXPANDADD(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_>
void LaunchTROWEXPANDMAX(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_>
void LaunchTROWEXPANDMIN(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_>
void LaunchTROWEXPANDEXPDIF(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_, typename LaunchFn>
void run_vec_op(LaunchFn fn)
{
    const size_t matSize = kTRows_ * kTCols_;
    size_t matFileSize = matSize * sizeof(T);
    size_t vecFileSize = kTCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), matFileSize);
    aclrtMallocHost((void **)(&src0Host), matFileSize);
    aclrtMallocHost((void **)(&src1Host), vecFileSize);

    aclrtMalloc((void **)&dstDevice, matFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, matFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, matFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", matFileSize, src0Host, matFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", vecFileSize, src1Host, vecFileSize));
    aclrtMemcpy(src0Device, matFileSize, src0Host, matFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, vecFileSize, src1Host, vecFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    fn(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, matFileSize, dstDevice, matFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, matFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(matSize);
    std::vector<T> devFinal(matSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", matFileSize, golden.data(), matFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", matFileSize, devFinal.data(), matFileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDOPTest, case_div_float_64x64)
{
    run_vec_op<float, 64, 64>([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDDIV<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_div_half_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream) {
        LaunchTROWEXPANDDIV<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_float_64x64)
{
    run_vec_op<float, 64, 64>([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDMUL<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_half_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream) {
        LaunchTROWEXPANDMUL<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_float_64x64)
{
    run_vec_op<float, 64, 64>([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDSUB<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_half_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream) {
        LaunchTROWEXPANDSUB<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_float_64x64)
{
    run_vec_op<float, 64, 64>([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDADD<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_half_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream) {
        LaunchTROWEXPANDADD<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_float_64x64)
{
    run_vec_op<float, 64, 64>([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDMAX<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_half_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream) {
        LaunchTROWEXPANDMAX<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_float_64x64)
{
    run_vec_op<float, 64, 64>([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDMIN<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_half_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream) {
        LaunchTROWEXPANDMIN<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_expdif_float_64x64)
{
    run_vec_op<float, 64, 64>([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDEXPDIF<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_expdif_half_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream) {
        LaunchTROWEXPANDEXPDIF<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}
