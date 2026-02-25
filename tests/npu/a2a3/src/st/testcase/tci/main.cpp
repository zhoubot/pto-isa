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
#include <gtest/gtest.h>
#include "tci_common.h"

using namespace std;
using namespace PtoTestCommon;

template <uint32_t descending>
void launchTCI_demo_b32_case1(int32_t *out, void *stream);
template <uint32_t descending>
void launchTCI_demo_b32_case2(int32_t *out, void *stream);
template <uint32_t descending>
void launchTCI_demo_b32_case3(int32_t *out, void *stream);
template <uint32_t descending>
void launchTCI_demo_b32_case4(int32_t *out, void *stream);
template <uint32_t descending>
void launchTCI_demo_b16_case1(int16_t *out, void *stream);
template <uint32_t descending>
void launchTCI_demo_b16_case2(int16_t *out, void *stream);
template <uint32_t descending>
void launchTCI_demo_b16_case3(int16_t *out, void *stream);
template <uint32_t descending>
void launchTCI_demo_b16_case4(int16_t *out, void *stream);

class TCITest : public testing::Test {
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
    std::cout << fullPath << std::endl;
    return fullPath;
}

template <typename T, uint32_t ROW, uint32_t COL, uint32_t descending, uint32_t start>
void test_vci_b32()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t FileSize = ROW * COL * sizeof(T);

    int32_t *dstHost;
    int32_t *dstDevice;

    aclrtMallocHost((void **)(&dstHost), FileSize);
    aclrtMalloc((void **)&dstDevice, FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (COL == FLOAT_T1_COL) {
        launchTCI_demo_b32_case1<descending>(dstDevice, stream);
    } else if (COL == FLOAT_T2_COL) {
        launchTCI_demo_b32_case2<descending>(dstDevice, stream);
    } else if (COL == FLOAT_T3_COL) {
        launchTCI_demo_b32_case3<descending>(dstDevice, stream);
    } else {
        launchTCI_demo_b32_case4<descending>(dstDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, FileSize, dstDevice, FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, FileSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int32_t> golden(FileSize);
    std::vector<int32_t> devFinal(FileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", FileSize, golden.data(), FileSize);
    ReadFile(GetGoldenDir() + "/output.bin", FileSize, devFinal.data(), FileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, uint32_t ROW, uint32_t COL, uint32_t descending, uint32_t start>
void test_vci_b16()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t FileSize = ROW * COL * sizeof(T);

    int16_t *dstHost;
    int16_t *dstDevice;

    aclrtMallocHost((void **)(&dstHost), FileSize);
    aclrtMalloc((void **)&dstDevice, FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (COL == HALF_T1_COL) {
        launchTCI_demo_b16_case1<descending>(dstDevice, stream);
    } else if (COL == HALF_T2_COL) {
        launchTCI_demo_b16_case2<descending>(dstDevice, stream);
    } else if (COL == HALF_T3_COL) {
        launchTCI_demo_b16_case3<descending>(dstDevice, stream);
    } else {
        launchTCI_demo_b16_case4<descending>(dstDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, FileSize, dstDevice, FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, FileSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int16_t> golden(FileSize);
    std::vector<int16_t> devFinal(FileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", FileSize, golden.data(), FileSize);
    ReadFile(GetGoldenDir() + "/output.bin", FileSize, devFinal.data(), FileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCITest, case1_int32)
{
    test_vci_b32<int32_t, FLOAT_ROW, FLOAT_T1_COL, ASCEND, START>();
}

TEST_F(TCITest, case2_int32)
{
    test_vci_b32<int32_t, FLOAT_ROW, FLOAT_T2_COL, ASCEND, START>();
}

TEST_F(TCITest, case3_int32)
{
    test_vci_b32<int32_t, FLOAT_ROW, FLOAT_T3_COL, DESCEND, START>();
}

TEST_F(TCITest, case4_int32)
{
    test_vci_b32<int32_t, FLOAT_ROW, FLOAT_T4_COL, DESCEND, START>();
}

TEST_F(TCITest, case5_int16)
{
    test_vci_b16<int16_t, HALF_ROW, HALF_T1_COL, ASCEND, START>();
}

TEST_F(TCITest, case6_int16)
{
    test_vci_b16<int16_t, HALF_ROW, HALF_T2_COL, DESCEND, START>();
}

TEST_F(TCITest, case7_int16)
{
    test_vci_b16<int16_t, HALF_ROW, HALF_T3_COL, ASCEND, START>();
}

TEST_F(TCITest, case8_int16)
{
    test_vci_b16<int16_t, HALF_ROW, HALF_T4_COL, DESCEND, START>();
}
