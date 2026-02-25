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

template <int kRows, int kCols>
AICORE void runTROWEXPAND(__gm__ float __out__ *out, __gm__ float __in__ *src)
{
    using ShapeMat = Shape<1, 1, 1, kRows, kCols>;
    using StrideMat = Stride<1, 1, 1, kCols, 1>;
    using GlobalMat = GlobalTensor<float, ShapeMat, StrideMat>;

    using TileMat = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    TileMat srcTile(kRows, kCols);
    TileMat dstTile(kRows, kCols);

    GlobalMat srcGlobal(src);
    GlobalMat dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TROWEXPAND(dstTile, srcTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
AICORE void runTROWEXPANDDIV(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1)
{
    using ShapeMat = Shape<1, 1, 1, kRows, kCols>;
    using StrideMat = Stride<1, 1, 1, kCols, 1>;
    using GlobalMat = GlobalTensor<float, ShapeMat, StrideMat>;

    using ShapeVec = Shape<1, 1, 1, kRows, 1>;
    using StrideVec = Stride<1, 1, 1, 1, 1>;
    using GlobalVec = GlobalTensor<float, ShapeVec, StrideVec>;

    using TileMat = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileVec = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor, -1, -1>;
    TileMat src0Tile(kRows, kCols);
    TileVec src1Tile(kRows, 1);
    TileMat dstTile(kRows, kCols);

    GlobalMat src0Global(src0);
    GlobalVec src1Global(src1);
    GlobalMat dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TROWEXPANDDIV(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
AICORE void runTROWEXPANDMUL(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1)
{
    using ShapeMat = Shape<1, 1, 1, kRows, kCols>;
    using StrideMat = Stride<1, 1, 1, kCols, 1>;
    using GlobalMat = GlobalTensor<float, ShapeMat, StrideMat>;

    using ShapeVec = Shape<1, 1, 1, kRows, 1>;
    using StrideVec = Stride<1, 1, 1, 1, 1>;
    using GlobalVec = GlobalTensor<float, ShapeVec, StrideVec>;

    using TileMat = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileVec = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor, -1, -1>;
    TileMat src0Tile(kRows, kCols);
    TileVec src1Tile(kRows, 1);
    TileMat dstTile(kRows, kCols);

    GlobalMat src0Global(src0);
    GlobalVec src1Global(src1);
    GlobalMat dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TROWEXPANDMUL(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
AICORE void runTROWEXPANDSUB(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1)
{
    using ShapeMat = Shape<1, 1, 1, kRows, kCols>;
    using StrideMat = Stride<1, 1, 1, kCols, 1>;
    using GlobalMat = GlobalTensor<float, ShapeMat, StrideMat>;

    using ShapeVec = Shape<1, 1, 1, kRows, 1>;
    using StrideVec = Stride<1, 1, 1, 1, 1>;
    using GlobalVec = GlobalTensor<float, ShapeVec, StrideVec>;

    using TileMat = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileVec = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor, -1, -1>;
    TileMat src0Tile(kRows, kCols);
    TileVec src1Tile(kRows, 1);
    TileMat dstTile(kRows, kCols);

    GlobalMat src0Global(src0);
    GlobalVec src1Global(src1);
    GlobalMat dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TROWEXPANDSUB(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
AICORE void runTROWEXPANDADD(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1)
{
    using ShapeMat = Shape<1, 1, 1, kRows, kCols>;
    using StrideMat = Stride<1, 1, 1, kCols, 1>;
    using GlobalMat = GlobalTensor<float, ShapeMat, StrideMat>;

    using ShapeVec = Shape<1, 1, 1, kRows, 1>;
    using StrideVec = Stride<1, 1, 1, 1, 1>;
    using GlobalVec = GlobalTensor<float, ShapeVec, StrideVec>;

    using TileMat = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileVec = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor, -1, -1>;
    TileMat src0Tile(kRows, kCols);
    TileVec src1Tile(kRows, 1);
    TileMat dstTile(kRows, kCols);

    GlobalMat src0Global(src0);
    GlobalVec src1Global(src1);
    GlobalMat dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TROWEXPANDADD(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
void LaunchTROWEXPAND(float *out, float *src, void *stream)
{
    (void)stream;
    runTROWEXPAND<kRows, kCols>(out, src);
}

template <int kRows, int kCols>
void LaunchTROWEXPANDDIV(float *out, float *src0, float *src1, void *stream)
{
    (void)stream;
    runTROWEXPANDDIV<kRows, kCols>(out, src0, src1);
}

template <int kRows, int kCols>
void LaunchTROWEXPANDMUL(float *out, float *src0, float *src1, void *stream)
{
    (void)stream;
    runTROWEXPANDMUL<kRows, kCols>(out, src0, src1);
}

template <int kRows, int kCols>
void LaunchTROWEXPANDSUB(float *out, float *src0, float *src1, void *stream)
{
    (void)stream;
    runTROWEXPANDSUB<kRows, kCols>(out, src0, src1);
}

template <int kRows, int kCols>
void LaunchTROWEXPANDADD(float *out, float *src0, float *src1, void *stream)
{
    (void)stream;
    runTROWEXPANDADD<kRows, kCols>(out, src0, src1);
}

template void LaunchTROWEXPAND<64, 64>(float *out, float *src, void *stream);
template void LaunchTROWEXPANDDIV<64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTROWEXPANDMUL<64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTROWEXPANDSUB<64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTROWEXPANDADD<64, 64>(float *out, float *src0, float *src1, void *stream);
