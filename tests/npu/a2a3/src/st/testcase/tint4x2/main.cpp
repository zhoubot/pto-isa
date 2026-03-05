/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.

This program is free software, you can redistribute it and/or modify it under the
terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

class TINT4X2Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <int TileH, int TileW, int VRows, int VCols>
void LaunchInt4x2Copy(uint8_t *out, uint8_t *src, void *stream);

static void Int4x2CopyTest(size_t bytes, void (*launcher)(uint8_t*, uint8_t*, void*), const char* caseDir)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost = nullptr;
    uint8_t *dstHost = nullptr;
    uint8_t *srcDev = nullptr;
    uint8_t *dstDev = nullptr;

    aclrtMallocHost((void **)&srcHost, bytes);
    aclrtMallocHost((void **)&dstHost, bytes);

    aclrtMalloc((void **)&srcDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fs = bytes;
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/input.bin", fs, srcHost, bytes));

    aclrtMemcpy(srcDev, bytes, srcHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    launcher(dstDev, srcDev, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, bytes, dstDev, bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, bytes);

    std::vector<uint8_t> golden(bytes);
    std::vector<uint8_t> out(bytes);
    fs = bytes;
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/golden.bin", fs, golden.data(), bytes));
    fs = bytes;
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/output.bin", fs, out.data(), bytes));

    EXPECT_EQ(0, std::memcmp(golden.data(), out.data(), bytes));

    (void)caseDir;
    aclrtFree(dstDev);
    aclrtFree(srcDev);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TINT4X2Test, case_copy_64x64)
{
    auto launcher = [](uint8_t* out, uint8_t* src, void* stream){
        LaunchInt4x2Copy<64,64,64,64>(out, src, stream);
    };
    Int4x2CopyTest(64*64, launcher, "case_copy_64x64");
}

TEST_F(TINT4X2Test, case_copy_32x128)
{
    auto launcher = [](uint8_t* out, uint8_t* src, void* stream){
        LaunchInt4x2Copy<32,128,32,128>(out, src, stream);
    };
    Int4x2CopyTest(32*128, launcher, "case_copy_32x128");
}

TEST_F(TINT4X2Test, case_copy_32x96_v32x95)
{
    auto launcher = [](uint8_t* out, uint8_t* src, void* stream){
        LaunchInt4x2Copy<32,96,32,95>(out, src, stream);
    };
    Int4x2CopyTest(32*96, launcher, "case_copy_32x96_v32x95");
}
