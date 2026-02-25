/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP
#include "common.hpp"

namespace pto {

constexpr const int KHALF = 2;             // for b4 data
constexpr const int M_STEP_MIN_VAL_B8 = 2; // m_step per loop for fp8
constexpr const int SHIFT_M_STEP_B8 = 1;   // 2^1 = 2
constexpr const int M_STEP_MIN_VAL_B4 = 4; // m_step per loop for fp4
constexpr const int SHIFT_M_STEP_B4 = 2;   // 2^2 = 4

template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToA(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                               uint16_t indexRow, uint16_t indexCol)
{
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> SHIFT_BLOCK_LEN;

        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0,
                      "srcRow must be aligned"); // fp16, fp32 should be aligned to 16
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> SHIFT_BLOCK_LEN;

        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToAVector(typename DstTileData::TileDType __out__ dst,
                                     typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                     uint16_t dstValidCol)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int32_t fractalSize = CUBE_BLOCK_SIZE / typeSize;
    int32_t kAlign = (dstValidCol + fractalSize - 1) & ~(fractalSize - 1);

    static_assert((SrcTileData::Cols % fractalSize) == 0, "srcCol * sizeof(DataType) must be aligned to 512B");
    static_assert((DstTileData::Cols % fractalSize) == 0, "dstCol * sizeof(DataType) must be aligned to 512B");
    PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol * sizeof(DataType) must be aligned to 512B");

    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_FRACTAL_BYTE;
    uint8_t kStep = kAlign / fractalSize;
    load_cbuf_to_ca(dstAddr, srcAddr, 0, kStartPosition, 1, kStep, 1, 1, 0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToACompact(typename DstTileData::TileDType __out__ dst,
                                      typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                      uint16_t madM, uint16_t madK)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint16_t madMAlign = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madMAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madMAlign >> SHIFT_BLOCK_LEN;
    load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToATransCompact(typename DstTileData::TileDType __out__ dst,
                                           typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
                                           uint16_t indexCol, uint16_t madM, uint16_t madK)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);

    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madMAlign = CeilDivision(madM, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madMAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madMAlign >> SHIFT_BLOCK_LEN;

    if constexpr (typeSize == 1) { // b8
        uint16_t dstAddrStride = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t mLoop = mStep >> SHIFT_M_STEP_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for (uint16_t idx = 0; idx < mLoop; ++idx) {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    } else { // b16/b32
        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToB(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                               uint16_t indexRow, uint16_t indexCol)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> SHIFT_BLOCK_LEN;
        load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0,
                      "srcRow must be aligned"); // fp16, fp32 should be aligned to 16
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> SHIFT_BLOCK_LEN;
        load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToBCompact(typename DstTileData::TileDType __out__ dst,
                                      typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                      uint16_t madK, uint16_t madN)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;

    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    uint16_t madNAlign = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madNAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madNAlign >> SHIFT_BLOCK_LEN;
    load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToBTransCompact(typename DstTileData::TileDType __out__ dst,
                                           typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
                                           uint16_t indexCol, uint16_t madK, uint16_t madN)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);

    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madNAlign = CeilDivision(madN, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madNAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madNAlign >> SHIFT_BLOCK_LEN;
    if constexpr (typeSize == 1) { // b8
        uint16_t dstAddrStride = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t nLoop = mStep >> SHIFT_M_STEP_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for (uint16_t idx = 0; idx < nLoop; ++idx) {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    } else { // b16/b32
        load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToMat(typename DstTileData::TileDType __out__ dst,
                                    typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                    uint32_t srcValidRow, uint32_t srcValidCol, uint32_t dstValidRow,
                                    uint32_t dstValidCol)
{
    using T = typename SrcTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    uint32_t offset = SrcTileData::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    if constexpr (SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) {
        offset = indexRow * srcValidCol + indexCol;
    }

    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src) + offset;
    __cbuf__ T *dstPtr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    if constexpr (SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) {
        uint16_t blockLen = dstValidRow * dstValidCol * sizeof(T) / BLOCK_BYTE_SIZE;
        // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, 1, blockLen, 0, 0);
    } else if constexpr (!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor)) {
        uint16_t blockCout = CeilDivision(dstValidCol, c0Size);
        uint32_t alignRow = (dstValidRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        uint16_t blockLen = alignRow * c0Size * sizeof(T) / BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = SrcTileData::Rows - DstTileData::Rows;
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, blockCout, blockLen, srcStride, 0);
    }
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TExtractAccToMat(typename DstTileData::TileDType __out__ dst,
                                    typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol,
                                    uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    constexpr bool channelSplitEnable =
        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<dstType, float>)&&(DstTileData::SFractalSize == CUBE_BLOCK_SIZE);
    constexpr int32_t c0Size = (!channelSplitEnable) && (DstTileData::SFractalSize == 2 * CUBE_BLOCK_SIZE) ?
                                   2 * C0_SIZE_BYTE / sizeof(dstType) :
                                   C0_SIZE_BYTE / sizeof(dstType);
    constexpr uint32_t dstStride = DstTileData::Rows * c0Size;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    uint32_t srcOffset = SrcTileData::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) +
                         (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    __cbuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)(src) + srcOffset;

    copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, nSize, validRow, dstStride, SrcTileData::Rows, 0, 0, QuantPre, reluMode,
                           channelSplitEnable, false, 0, 0, false, false, 0, false, false, false, false, false, false);
}

template <typename T>
constexpr bool is_textract_supported_type =
    std::disjunction_v<std::is_same<T, int8_t>, std::is_same<T, uint8_t>, std::is_same<T, half>,
                       std::is_same<T, int16_t>, std::is_same<T, uint16_t>, std::is_same<T, int32_t> >;

template <typename DstType, typename SrcType>
PTO_INTERNAL void CheckTExtractToL0()
{
    static_assert(std::is_same_v<DstType, SrcType>, "Dst data type must be same with Src.");
    static_assert((std::is_same_v<SrcType, half>) || (std::is_same_v<SrcType, int16_t>) ||
                      (std::is_same_v<SrcType, uint16_t>) || (std::is_same_v<SrcType, int8_t>) ||
                      (std::is_same_v<SrcType, uint8_t>),
                  "The data type must be restricted to half/(u)int8/(u)int16.");
}

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToLeft(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor) ||
                      SrcTileData::isRowMajor,
                  "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                  "TExtract: DstTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                  "TExtract: DstTile Invalid Fractal");
    CheckTExtractToL0<typename DstTileData::DType, typename SrcTileData::DType>();
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(),
                                                         dst.GetValidCol());
        } else {
            TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToATransCompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol,
                                                              dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToRight(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
                  "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: DstTile Invalid Fractal");
    CheckTExtractToL0<typename DstTileData::DType, typename SrcTileData::DType>();
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(),
                                                         dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToBTransCompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol,
                                                              dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType>
PTO_INTERNAL void CheckTExtractAccValid()
{
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    static_assert((SrcTileData::Loc == TileType::Acc), "Source TileType only support Acc.");
    static_assert((!SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor),
                  "Src fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    if constexpr (std::is_same_v<SrcType, half>) {
        static_assert((std::is_same_v<SrcType, half>) || (std::is_same_v<SrcType, int8_t>) ||
                          (std::is_same_v<SrcType, uint8_t>) || (std::is_same_v<SrcType, int16_t>),
                      "The out data type must be restricted to half/int8/uint8/int16");
    } else if constexpr (std::is_same_v<SrcType, int32_t>) {
        static_assert((std::is_same_v<SrcType, half>) || (std::is_same_v<SrcType, int8_t>) ||
                          (std::is_same_v<SrcType, uint8_t>) || (std::is_same_v<SrcType, int16_t>) ||
                          (std::is_same_v<SrcType, int32_t>),
                      "The out data type must be restricted to half/int8/uint8/int16");
    } else {
        static_assert(sizeof(SrcType) == 0, "Src data type only support half or int32_t");
    }
    static_assert(((DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                   (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                   (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor)),
                  "Only support nz2nz, nz2nd or nz2dn.");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::Loc == TileType::Acc) ||
                      std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
                  "TExtract: Destination and Source tile data types must be the same");

    if constexpr (DstTileData::Loc == TileType::Left) {
        TExtractToLeft(dst, src, indexRow, indexCol);
    } else if constexpr (DstTileData::Loc == TileType::Right) {
        TExtractToRight(dst, src, indexRow, indexCol);
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Mat) {
        static_assert(is_textract_supported_type<typename DstTileData::DType>,
                      "TExtract: Unsupported data type! Supported types: int8_t, half, int32");
        TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, src.GetValidRow(),
                                                   src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (DstTileData::Loc == TileType::ScaleLeft) {
        static_assert(sizeof(DstTileData::DType) == 0, "TExtract: ScaleLeft tile type is not supported yet.");
    } else if constexpr (DstTileData::Loc == TileType::ScaleRight) {
        static_assert(sizeof(DstTileData::DType) == 0, "TExtract: ScaleRight tile type is not supported yet.");
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        CheckTExtractAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

// vector quant
template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPC(typename FpTileData::TileDType __in__ fp, uint16_t indexCol)
{
    __fbuf__ typename FpTileData::DType *dstAddrFp =
        (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp) + indexCol;
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTExtractAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    CheckTExtractAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    CheckTExtractAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data(), indexCol);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}
} // namespace pto
#endif // TEXTRACT_HPP
