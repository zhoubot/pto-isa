/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMP_HPP
#define TCMP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

constexpr const uint64_t BITS_IN_BYTE = 8;

    template <typename TileDataDst, typename TileDataSrc>
    AICORE void CmpCall(
        __ubuf__ typename TileDataDst::DType *dst,
        __ubuf__ typename TileDataSrc::DType *src0, 
        __ubuf__ typename TileDataSrc::DType *src1, 
        CmpMode cmpMode,
        uint8_t repeat, uint16_t dstblockstride, uint16_t srcblockstride,
        uint16_t dstrepeatstride, uint16_t srcrepeatstride)
{
        if constexpr (std::is_same<typename TileDataSrc::DType, int32_t>::value) {
            vcmpv_eq(dst, src0, src1, repeat, 
                dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
        }
        else {
            switch (static_cast<CmpMode>(cmpMode)) {
                case CmpMode::EQ:
                    vcmpv_eq(dst, src0, src1, repeat, 
                        dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::NE:
                    vcmpv_ne(dst, src0, src1, repeat, 
                        dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::LT:
                    vcmpv_lt(dst, src0, src1, repeat, 
                        dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::GT:
                    vcmpv_gt(dst, src0, src1, repeat, 
                        dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::GE:
                    vcmpv_ge(dst, src0, src1, repeat, 
                        dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::LE:
                    vcmpv_le(dst, src0, src1, repeat, 
                        dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
                    break;
                default:
                    vcmpv_eq(dst, src0, src1, repeat, 
                        dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride, srcrepeatstride);
                    break;
            }
        }
    }


    template <typename TileDataDst, typename TileDataSrc, typename T>
    __tf__ AICORE void TCmp(
        typename TileDataDst::TileDType __out__ dst,
        typename TileDataSrc::TileDType __in__ src0, 
        typename TileDataSrc::TileDType __in__ src1, 
        CmpMode mode, 
        unsigned numRepeatPerLine,
        unsigned validRow,
        unsigned elementsPerRepeat) 
    {
        __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataSrc::DType *src0Ptr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileDataSrc::DType *src1Ptr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src1);
        
        size_t numLoop = numRepeatPerLine / REPEAT_MAX;
        int numRemainPerLine = numRepeatPerLine % REPEAT_MAX;
        constexpr int srcAlignCols = TileDataSrc::Cols;
        constexpr int dstAlignCols = TileDataDst::Cols;
        constexpr int srcOffset = REPEAT_MAX * REPEAT_BYTE / sizeof(T);
        constexpr int dstOffset = REPEAT_MAX * REPEAT_BYTE / sizeof(T) / BITS_IN_BYTE;

        set_mask_norm();
        set_vector_mask(-1, -1);
        for(size_t i = 0; i< validRow; i++) {
            for (size_t j = 0; i < numLoop; j++) {
                CmpCall<TileDataDst, TileDataSrc>(
                    dstPtr + i * dstAlignCols + j * dstOffset,
                    src0Ptr + i * srcAlignCols + j * srcOffset,
                    src1Ptr + i * srcAlignCols + j * srcOffset,
                    mode, 
                    REPEAT_MAX,
                    1,
                    1,
                    8,
                    8
                );
            }
            if(numRemainPerLine) {
                CmpCall<TileDataDst, TileDataSrc>(
                    dstPtr + i * dstAlignCols + numLoop * dstOffset,
                    src0Ptr + i * srcAlignCols + numLoop * srcOffset,
                    src1Ptr + i * srcAlignCols + numLoop * srcOffset,
                    mode, 
                    numRemainPerLine,
                    1,
                    1,
                    8,
                    8
                );
            }
        }
    }

    template <typename TileDataDst, typename TileDataSrc>
    PTO_INTERNAL void TCMP_IMPL(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1, CmpMode cmpMode) {
        static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src tiles must be TileType::Vec.");
        static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tiles must be TileType::Vec.");
        static_assert(TileDataSrc::ValidCol <= TileDataSrc::Cols, "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileDataSrc::ValidRow <= TileDataSrc::Rows, "Number of valid rows must not be greater than number of tile rows.");
        
        PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataSrc::DType);
        unsigned numRepeatPerLine = CeilDivision(src0.GetValidCol(), elementsPerRepeat);
        unsigned validRow = src0.GetValidRow();
        using T = typename TileDataSrc::DType;
        TCmp<TileDataDst, TileDataSrc, T>(dst.data(), src0.data(), src1.data(), cmpMode, numRepeatPerLine, validRow, elementsPerRepeat);
    }
}
#endif
