/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/npu/a5/TQuant.hpp>
#include "acl/acl.h"
#include <type_traits>

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))

namespace TQuantTest {

// FP32 --> MXFP8
// Quantize fp32 tile to fp8 (e4m3) and exponent-only (e8m0).
// Pad columns to multiples of 32 using min fill to avoid reading garbage.
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuant(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp8,
                                 __gm__ float __in__ *src)
{
    // pad each row to multiple of 32 elements
    constexpr int paddedCols = PTO_CEIL(validCols, 32);
    constexpr int groupedCols_flattened = validRows * (paddedCols / 32);
    constexpr int groupedCols_valid = paddedCols / 32;
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, groupedCols_flattened>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstFP8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;

    // define tile layout based on mode, 0 - ND, 1 - NZ
    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstE8Tile = Tile<TileType::Vec, uint8_t, 1, groupedCols_flattened, BLayout::RowMajor, -1, -1,
                           SLayout::NoneBox, 512, PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, uint8_t, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                            512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, float, 1, groupedCols_flattened, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    SrcTile scalingTile(validRows, validCols);
    DstFP8Tile fp8Tile(validRows, paddedCols);
    DstE8Tile e8Tile(1, groupedCols_flattened);
    MaxTile maxPerGpTile(1, groupedCols_flattened);

    SrcGlobal srcGlobal(src);
    DstE8Global e8Global(out_e8m0);
    DstFP8Global fp8Global(out_fp8);

    TASSIGN(srcTile, 0x0);          // 64  KB = 0x10000
    TASSIGN(maxPerGpTile, 0x10100); // 1.5 KB = 0x1800 (Max and Scaling can overlap)
    TASSIGN(scalingTile, 0x21820);  // 3 KB   = 0xC00
    TASSIGN(e8Tile, 0x30100);       // 0.5 KB = 0x600
    TASSIGN(fp8Tile, 0x38160);      // 16  KB = 0x4000
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TQUANT<pto::QuantType::MXFP8, DstFP8Tile, SrcTile, DstE8Tile, MaxTile>(fp8Tile, srcTile, &e8Tile, &maxPerGpTile,
                                                                           &scalingTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(e8Global, e8Tile);
    TSTORE(fp8Global, fp8Tile);
}

// FP32 --> INT8 SYM
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuantInt8Sym(__gm__ int8_t __out__ *out_s8, __gm__ float __in__ *src,
                                        __gm__ float __in__ *scale)
{
    constexpr int paddedCols_b32 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(float));
    constexpr int paddedCols_b8 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(int8_t));
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstGlobal = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using ParaGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols_b32, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, int8_t, validRows, paddedCols_b8, BLayout::RowMajor, -1, -1>;
    using ParaTile = Tile<TileType::Vec, float, validRows, 1, BLayout::ColMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstS8Tile(validRows, validCols);
    ParaTile scaleTile(validRows, 1);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out_s8);
    ParaGlobal scaleGlobal(scale);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstS8Tile, 0x0);
    TASSIGN(scaleTile, 0x20100);

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TQUANT<pto::QuantType::INT8_SYM, DstTile, SrcTile, ParaTile>(dstS8Tile, srcTile, scaleTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstS8Tile);
}

// FP32 --> INT8 ASYM
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuantInt8Asym(__gm__ uint8_t __out__ *out_u8, __gm__ float __in__ *src,
                                         __gm__ float __in__ *scale, __gm__ float __in__ *offset)
{
    // pad each row to multiple of 32 elements
    constexpr int paddedCols_b32 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(float));
    constexpr int paddedCols_b8 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(uint8_t));
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstGlobal = GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using ParaGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols_b32, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, uint8_t, validRows, paddedCols_b8, BLayout::RowMajor, -1, -1>;
    using ParaTile = Tile<TileType::Vec, float, validRows, 1, BLayout::ColMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstU8Tile(validRows, validCols);
    ParaTile scaleTile(validRows, 1);
    ParaTile offsetTile(validRows, 1);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out_u8);
    ParaGlobal scaleGlobal(scale);
    ParaGlobal offsetGlobal(offset);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstU8Tile, 0x20100);
    TASSIGN(scaleTile, 0x30100);
    TASSIGN(offsetTile, 0x32500);

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);
    TLOAD(offsetTile, offsetGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TQUANT<pto::QuantType::INT8_ASYM, DstTile, SrcTile, ParaTile>(dstU8Tile, srcTile, scaleTile, &offsetTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstU8Tile);
}

template <int validRows, int validCols, int mode>
void LaunchTQuantMXFP8(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream)
{
    runTQuant<validRows, validCols, mode><<<1, nullptr, stream>>>(dst_exp, dst, src);
}

template <int validRows, int validCols, int mode, pto::QuantType quantType>
void LaunchTQuantInt8(std::conditional_t<quantType == pto::QuantType::INT8_SYM, int8_t, uint8_t> *dst, float *src,
                      float *scale, void *stream, float *offset = nullptr)
{
    if constexpr (quantType == pto::QuantType::INT8_SYM) {
        runTQuantInt8Sym<validRows, validCols, mode><<<1, nullptr, stream>>>(dst, src, scale);
    } else {
        runTQuantInt8Asym<validRows, validCols, mode><<<1, nullptr, stream>>>(dst, src, scale, offset);
    }
}

} // namespace TQuantTest

// MXFP8 cases
template void TQuantTest::LaunchTQuantMXFP8<32, 32, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<32, 64, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<64, 128, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
template void TQuantTest::LaunchTQuantMXFP8<128, 128, 0>(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);
// INT8 SYM cases
template void TQuantTest::LaunchTQuantInt8<64, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                 void *stream, float *offset);
template void TQuantTest::LaunchTQuantInt8<128, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                  void *stream, float *offset);
template void TQuantTest::LaunchTQuantInt8<256, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                  void *stream, float *offset);
// INT8 ASYM cases
template void TQuantTest::LaunchTQuantInt8<64, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                  float *scale, void *stream,
                                                                                  float *offset);
template void TQuantTest::LaunchTQuantInt8<128, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                   float *scale, void *stream,
                                                                                   float *offset);
template void TQuantTest::LaunchTQuantInt8<256, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                   float *scale, void *stream,
                                                                                   float *offset);