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

using namespace PtoTestCommon;

template <int kRows, int kCols>
void LaunchTROWEXPAND(float *out, float *src, void *stream);
template <int kRows, int kCols>
void LaunchTROWEXPANDDIV(float *out, float *src0, float *src1, void *stream);
template <int kRows, int kCols>
void LaunchTROWEXPANDMUL(float *out, float *src0, float *src1, void *stream);
template <int kRows, int kCols>
void LaunchTROWEXPANDSUB(float *out, float *src0, float *src1, void *stream);
template <int kRows, int kCols>
void LaunchTROWEXPANDADD(float *out, float *src0, float *src1, void *stream);

class TROWEXPAND_Test : public testing::Test {
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

static void setup_stream(aclrtStream &stream)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&stream);
}

static void teardown_stream(aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TROWEXPAND_Test, case_expand_float_64x64)
{
    constexpr int kRows = 64;
    constexpr int kCols = 64;
    const size_t size = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(float);

    aclrtStream stream;
    setup_stream(stream);

    float *dstHost, *srcHost;
    float *dstDevice, *srcDevice;
    aclrtMallocHost((void **)(&dstHost), size);
    aclrtMallocHost((void **)(&srcHost), size);
    aclrtMalloc((void **)&dstDevice, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, size, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, size));
    aclrtMemcpy(srcDevice, size, srcHost, size, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTROWEXPAND<kRows, kCols>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, size, dstDevice, size, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, size);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    teardown_stream(stream);

    std::vector<float> golden(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<float> out(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), size));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), size));
    EXPECT_TRUE(ResultCmp<float>(golden, out.data(), 0.0f));
}

template <typename LaunchFn>
static void run_vec_op(LaunchFn fn)
{
    constexpr int kRows = 64;
    constexpr int kCols = 64;
    const size_t matSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(float);
    const size_t vecSize = static_cast<size_t>(kRows) * sizeof(float);

    aclrtStream stream;
    setup_stream(stream);

    float *dstHost, *src0Host, *src1Host;
    float *dstDevice, *src0Device, *src1Device;
    aclrtMallocHost((void **)(&dstHost), matSize);
    aclrtMallocHost((void **)(&src0Host), matSize);
    aclrtMallocHost((void **)(&src1Host), vecSize);
    aclrtMalloc((void **)&dstDevice, matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, matSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, vecSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", readSize, src0Host, matSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", readSize, src1Host, vecSize));
    aclrtMemcpy(src0Device, matSize, src0Host, matSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, vecSize, src1Host, vecSize, ACL_MEMCPY_HOST_TO_DEVICE);

    fn(dstDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, matSize, dstDevice, matSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, matSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    teardown_stream(stream);

    std::vector<float> golden(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<float> out(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), matSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), matSize));
    EXPECT_TRUE(ResultCmp<float>(golden, out.data(), 0.0f));
}

TEST_F(TROWEXPAND_Test, case_div_float_64x64)
{
    run_vec_op([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDDIV<64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPAND_Test, case_mul_float_64x64)
{
    run_vec_op([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDMUL<64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPAND_Test, case_sub_float_64x64)
{
    run_vec_op([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDSUB<64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPAND_Test, case_add_float_64x64)
{
    run_vec_op([](float *out, float *src0, float *src1, void *stream) {
        LaunchTROWEXPANDADD<64, 64>(out, src0, src1, stream);
    });
}
