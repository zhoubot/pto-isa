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
#include "pto/pto-inst.hpp"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTTRI_demo(uint8_t *out, uint8_t *src, void *stream);

class TTRITest : public testing::Test {
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

template <int row, int col, int isUpperOrLower, int diagonal>
void LaunchTTRI(int32_t *out, void *stream);

template <int row, int col, int isUpperOrLower, int diagonal>
void test_ttri()
{
    size_t fileSize = row * col * 4;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *dstHost;
    int32_t *dstDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchTTRI<row, col, isUpperOrLower, diagonal>(dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);

    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int32_t> golden(fileSize);
    std::vector<int32_t> devFinal(fileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRITest, case_ttri_64x64_upper_diag0)
{
    test_ttri<64, 64, 1, 0>();
}
TEST_F(TTRITest, case_ttri_100x64_upper_diag_2)
{
    test_ttri<100, 64, 1, -2>();
}
TEST_F(TTRITest, case_ttri_128x32_lower_diag1)
{
    test_ttri<128, 32, 0, 1>();
}
TEST_F(TTRITest, case_ttri_200x48_upper_diag2)
{
    test_ttri<200, 48, 1, 2>();
}
TEST_F(TTRITest, case_ttri_256x16_lower_diag_1)
{
    test_ttri<256, 16, 0, -1>();
}