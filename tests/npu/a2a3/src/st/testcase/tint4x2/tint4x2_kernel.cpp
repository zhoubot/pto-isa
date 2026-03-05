/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.

This program is free software, you can redistribute it and/or modify it under the
terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <int TileH, int TileW, int VRows, int VCols>
__global__ AICORE void RunInt4x2Copy(__gm__ int4x2_t __out__ *out, __gm__ int4x2_t __in__ *src)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<int4x2_t, DynShape, DynStride>;

    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, VRows, VCols),
                         pto::Stride(TileH * TileW, TileH * TileW, TileH * TileW, TileW, 1));
    GlobalData srcGlobal(src, pto::Shape(1, 1, 1, VRows, VCols),
                         pto::Stride(TileH * TileW, TileH * TileW, TileH * TileW, TileW, 1));

    using TileData = Tile<TileType::Vec, int4x2_t, TileH, TileW, BLayout::RowMajor, -1, -1>;
    TileData tile(VRows, VCols);

    // Place tile in UB at base.
    TASSIGN(tile, 0x0);

    TLOAD(tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, tile);
    out = dstGlobal.data();
}

template <int TileH, int TileW, int VRows, int VCols>
void LaunchInt4x2Copy(uint8_t *out, uint8_t *src, void *stream)
{
    RunInt4x2Copy<TileH, TileW, VRows, VCols><<<1, nullptr, stream>>>(reinterpret_cast<int4x2_t *>(out),
                                                                      reinterpret_cast<int4x2_t *>(src));
}

// instantiations
template void LaunchInt4x2Copy<64, 64, 64, 64>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchInt4x2Copy<32, 128, 32, 128>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchInt4x2Copy<32, 96, 32, 95>(uint8_t *out, uint8_t *src, void *stream);
