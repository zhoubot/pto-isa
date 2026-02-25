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
#include "acl/acl.h"
#include <gtest/gtest.h>

#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

class TSELSTest : public testing::Test {
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
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kPadValue_>
void LaunchTSels(T *out, T *src0, T *src1, uint8_t selectMode, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kPadValue_>
void test_tsels()
{
    size_t fileSize = kTRows_ * kTCols_ * sizeof(T);
    if (kPadValue_ == PAD_VALUE_MAX) {
        fileSize = kGRows_ * kGCols_ * sizeof(T);
    }
    size_t scalarFileSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;
    uint8_t selectMode, *srcScalarHost;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);
    aclrtMallocHost((void **)(&srcScalarHost), scalarFileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);
    ReadFile(GetGoldenDir() + "/input_scalar.bin", scalarFileSize, srcScalarHost, scalarFileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    selectMode = srcScalarHost[0];
    LaunchTSels<T, kGRows_, kGCols_, kTRows_, kTCols_, kPadValue_>(dstDevice, src0Device, src1Device, selectMode,
                                                                   stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(srcScalarHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret);
}

TEST_F(TSELSTest, case_float_64x64_32x32_64x64)
{
    test_tsels<float, 64, 64, 32, 32, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_float_128x128_64x64_128x128)
{
    test_tsels<float, 128, 128, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_float_2x32_2x32_2x32)
{
    test_tsels<float, 2, 32, 2, 32, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_int32_2x32_2x32_2x32)
{
    test_tsels<int32_t, 2, 32, 2, 32, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_uint32_2x32_2x32_2x32)
{
    test_tsels<uint32_t, 2, 32, 2, 32, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_int16_2x32_2x32_2x32)
{
    test_tsels<int16_t, 2, 32, 2, 32, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_int8_2x32_2x32_2x32)
{
    test_tsels<int8_t, 2, 32, 2, 32, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_uint8_2x32_2x32_2x32)
{
    test_tsels<uint8_t, 2, 32, 2, 32, PAD_VALUE_NULL>();
}
TEST_F(TSELSTest, case_float_60x60_64x64_60x60)
{
    test_tsels<float, 60, 60, 64, 64, PAD_VALUE_MAX>();
}
TEST_F(TSELSTest, case_float_16x200_20x224_16x200)
{
    test_tsels<float, 16, 200, 20, 224, PAD_VALUE_MAX>();
}
TEST_F(TSELSTest, case_float_16x200_20x256_16x200)
{
    test_tsels<float, 16, 200, 20, 256, PAD_VALUE_MAX>();
}
TEST_F(TSELSTest, case_float_1x3600_2x4096_1x3600)
{
    test_tsels<float, 1, 3600, 2, 4096, PAD_VALUE_MAX>();
}
TEST_F(TSELSTest, case_uint16_16x200_20x224_16x200)
{
    test_tsels<uint16_t, 16, 200, 20, 224, PAD_VALUE_MAX>();
}