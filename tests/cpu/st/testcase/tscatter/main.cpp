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

class TSCATTERTest : public testing::Test {
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

template <int kTRows_, int kTCols_>
void LaunchTScatter(float *out, float *src, uint16_t *idx, void *stream);

template <int kTRows_, int kTCols_>
void test_tscatter()
{
    const size_t tileBytes = kTRows_ * kTCols_ * sizeof(float);
    const size_t idxBytes = kTRows_ * kTCols_ * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *srcHost;
    uint16_t *idxHost;
    float *dstDevice, *srcDevice;
    uint16_t *idxDevice;

    aclrtMallocHost((void **)(&dstHost), tileBytes);
    aclrtMallocHost((void **)(&srcHost), tileBytes);
    aclrtMallocHost((void **)(&idxHost), idxBytes);

    aclrtMalloc((void **)&dstDevice, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t tileSize = tileBytes;
    size_t idxSize = idxBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", tileSize, srcHost, tileBytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", idxSize, idxHost, idxBytes));

    aclrtMemcpy(srcDevice, tileBytes, srcHost, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxBytes, idxHost, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTScatter<kTRows_, kTCols_>(dstDevice, srcDevice, idxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, tileBytes, dstDevice, tileBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, tileBytes);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(idxDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(idxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(tileBytes / sizeof(float));
    std::vector<float> devFinal(tileBytes / sizeof(float));
    tileSize = tileBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", tileSize, golden.data(), tileBytes));
    tileSize = tileBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", tileSize, devFinal.data(), tileBytes));
    EXPECT_TRUE(ResultCmp<float>(golden, devFinal, 0.001f));
}

TEST_F(TSCATTERTest, case_float_16x16_16x16_16x16)
{
    test_tscatter<16, 16>();
}
