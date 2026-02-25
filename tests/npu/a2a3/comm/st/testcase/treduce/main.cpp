/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// G.EXP.05-CPP: Include shared header instead of forward declarations
#include "treduce_kernel.h"

// ============================================================================
// TREDUCE Tests - Basic (data fits in single UB Tile)
// ============================================================================
TEST(TReduce, FloatSmall_Sum)
{
    ASSERT_TRUE((RunReduceFloat256Sum(4, 4, 0, 0)));
}
TEST(TReduce, FloatSmall_Sum_8Ranks)
{
    ASSERT_TRUE((RunReduceFloat256Sum(8, 8, 0, 0)));
}
TEST(TReduce, Int32Large_Sum)
{
    ASSERT_TRUE((RunReduceInt32_4096_Sum(2, 2, 0, 0)));
}
TEST(TReduce, Int32Large_Sum_8Ranks)
{
    ASSERT_TRUE((RunReduceInt32_4096_Sum(8, 8, 0, 0)));
}
TEST(TReduce, Int32Small_Sum)
{
    ASSERT_TRUE((RunReduceInt32_512_Sum(2, 2, 0, 0)));
}
TEST(TReduce, Int32Small_Sum_8Ranks)
{
    ASSERT_TRUE((RunReduceInt32_512_Sum(8, 8, 0, 0)));
}
TEST(TReduce, Int32Small_Max)
{
    ASSERT_TRUE((RunReduceInt32_256_Max(2, 2, 0, 0)));
}
TEST(TReduce, Int32Small_Max_8Ranks)
{
    ASSERT_TRUE((RunReduceInt32_256_Max(8, 8, 0, 0)));
}
TEST(TReduce, Int32Small_Min)
{
    ASSERT_TRUE((RunReduceInt32_256_Min(2, 2, 0, 0)));
}
TEST(TReduce, Int32Small_Min_8Ranks)
{
    ASSERT_TRUE((RunReduceInt32_256_Min(8, 8, 0, 0)));
}
TEST(TReduce, SingleRank_Sum)
{
    ASSERT_TRUE((RunReduceFloat256Sum(1, 1, 0, 0)));
}
TEST(TReduce, Root1_FloatSmall_Sum)
{
    ASSERT_TRUE((RunReduceFloat256SumWithRoot(2, 2, 0, 0, 1)));
}
TEST(TReduce, EmptyRows_FloatSmall_Sum)
{
    ASSERT_TRUE((RunReduceEmptyFloat256Sum(2, 2, 0, 0, 0)));
}

// ============================================================================
// TREDUCE Tests - Large Shape Chunked (GlobalTensor > UB Tile, auto-chunked)
// ============================================================================
// int32: 128x32, tile 16 rows → 8 chunks, Sum, 2 ranks
TEST(TReduce, LargeShape_Int32_128x32_tile16_Sum)
{
    ASSERT_TRUE((RunReduceLargeShape_Int32_128x32_tile16_Sum(2, 2, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, Sum, 4 ranks
TEST(TReduce, LargeShape_Int32_128x32_tile16_Sum_4Ranks)
{
    ASSERT_TRUE((RunReduceLargeShape_Int32_128x32_tile16_Sum(4, 4, 0, 0)));
}
// float: 256x64, tile 32 rows → 8 chunks, Sum, 2 ranks
TEST(TReduce, LargeShape_Float_256x64_tile32_Sum)
{
    ASSERT_TRUE((RunReduceLargeShape_Float_256x64_tile32_Sum(2, 2, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, Max, 2 ranks
TEST(TReduce, LargeShape_Int32_128x32_tile16_Max)
{
    ASSERT_TRUE((RunReduceLargeShape_Int32_128x32_tile16_Max(2, 2, 0, 0)));
}
// int32: 512x32, tile 64 rows → 8 chunks, Sum, 2 ranks (larger data)
TEST(TReduce, LargeShape_Int32_512x32_tile64_Sum)
{
    ASSERT_TRUE((RunReduceLargeShape_Int32_512x32_tile64_Sum(2, 2, 0, 0)));
}
// int32: 512x32, tile 64 rows → 8 chunks, Sum, 8 ranks
TEST(TReduce, LargeShape_Int32_512x32_tile64_Sum_8Ranks)
{
    ASSERT_TRUE((RunReduceLargeShape_Int32_512x32_tile64_Sum(8, 8, 0, 0)));
}

// ============================================================================
// TREDUCE Tests - Ping-Pong Double Buffering (3 UB Tiles: acc + ping + pong)
// ============================================================================
// int32: 128x32, tile 16 rows → 8 chunks, Sum, 2 ranks
TEST(TReduce, PingPong_Int32_128x32_tile16_Sum)
{
    ASSERT_TRUE((RunReducePingPong_Int32_128x32_tile16_Sum(2, 2, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, Sum, 4 ranks
TEST(TReduce, PingPong_Int32_128x32_tile16_Sum_4Ranks)
{
    ASSERT_TRUE((RunReducePingPong_Int32_128x32_tile16_Sum(4, 4, 0, 0)));
}
// float: 256x64, tile 32 rows → 8 chunks, Sum, 2 ranks
TEST(TReduce, PingPong_Float_256x64_tile32_Sum)
{
    ASSERT_TRUE((RunReducePingPong_Float_256x64_tile32_Sum(2, 2, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, Max, 2 ranks
TEST(TReduce, PingPong_Int32_128x32_tile16_Max)
{
    ASSERT_TRUE((RunReducePingPong_Int32_128x32_tile16_Max(2, 2, 0, 0)));
}
// float: 256x64, tile 32 rows → 8 chunks, Sum, 8 ranks
TEST(TReduce, PingPong_Float_256x64_tile32_Sum_8Ranks)
{
    ASSERT_TRUE((RunReducePingPong_Float_256x64_tile32_Sum(8, 8, 0, 0)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
