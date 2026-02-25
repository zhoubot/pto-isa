/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

template <int row, int col, int isUpperOrLower, int diagonal>
AICORE void runTTRI(__gm__ int32_t __out__ *out)
{
    using DynShapeDim5 = Shape<1, 1, 1, row, col>;
    using DynStridDim5 = Stride<1, 1, 1, col, 1>;
    using GlobalData = GlobalTensor<int32_t, DynShapeDim5, DynStridDim5>;

    using TileT = Tile<TileType::Vec, int32_t, row, col, BLayout::RowMajor, -1, -1>;
    TileT dstTile(row, col);
    GlobalData dstGlobal(out);

    TTRI<TileT, isUpperOrLower>(dstTile, diagonal);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int row, int col, int isUpperOrLower, int diagonal>
void LaunchTTRI(int32_t *out, void *stream)
{
    (void)stream;
    runTTRI<row, col, isUpperOrLower, diagonal>(out);
}

template void LaunchTTRI<64, 64, 1, 0>(int32_t *out, void *stream);
template void LaunchTTRI<100, 64, 1, -2>(int32_t *out, void *stream);
template void LaunchTTRI<128, 32, 0, 1>(int32_t *out, void *stream);
template void LaunchTTRI<200, 48, 1, 2>(int32_t *out, void *stream);
template void LaunchTTRI<256, 16, 0, -1>(int32_t *out, void *stream);
