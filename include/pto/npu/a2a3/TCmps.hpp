/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMPS_HPP
#define TCMPS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

constexpr const uint64_t NUM_BITS_IN_BYTE = 8;

    template <typename TileDataDst, typename TileDataSrc, typename T>
    AICORE void GenCmpCall(__ubuf__ typename TileDataDst::DType *dst,
        __ubuf__ typename TileDataSrc::DType *src0, T src1, CmpMode cmpMode,
        uint8_t repeat, uint16_t dstblockstride, uint16_t srcblockstride,
        uint16_t dstrepeatstride, uint16_t srcrepeatstride)
{
        if constexpr (std::is_same<typename TileDataSrc::DType, int32_t>::value) {
            vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
        }
        else {
            switch (static_cast<CmpMode>(cmpMode)) {
                case CmpMode::EQ:
                    vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::NE:
                    vcmpvs_ne(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::LT:
                    vcmpvs_lt(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::GT:
                    vcmpvs_gt(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::GE:
                    vcmpvs_ge(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::LE:
                    vcmpvs_le(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                default:
                    vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
            }
        }
    }


    template <typename TileDataDst, typename TileDataSrc, typename T>
    __tf__ AICORE void TCmps(
        typename TileDataDst::TileDType __out__ dst,
        typename TileDataSrc::TileDType __in__ src0, 
        T src1, 
        CmpMode mode, 
        unsigned numRepeatPerLine,
        unsigned validRow,
        unsigned elementsPerRepeat) 
    {
        __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataSrc::DType *srcPtr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src0);

        size_t numLoop = numRepeatPerLine / REPEAT_MAX;
        int numRemainPerLine = numRepeatPerLine % REPEAT_MAX;
        constexpr int srcAlignCols = TileDataSrc::Cols;
        constexpr int dstAlignCols = TileDataDst::Cols;
        constexpr int srcOffset = REPEAT_MAX * REPEAT_BYTE / sizeof(T);
        constexpr int dstOffset = REPEAT_MAX * REPEAT_BYTE / sizeof(T) / NUM_BITS_IN_BYTE;
        
        set_mask_norm();
        set_vector_mask(-1, -1);
        for(size_t i = 0; i < validRow; i++) {
            for(size_t j = 0; j < numLoop; j++) {
                GenCmpCall<TileDataDst, TileDataSrc>(
                        dstPtr + i * dstAlignCols + j * dstOffset,
                        srcPtr + i * srcAlignCols + j * srcOffset,
                        src1,
                        mode,
                        REPEAT_MAX,
                        1,
                        1,
                        8,
                        8);
            }
            if(numRemainPerLine) {
                GenCmpCall<TileDataDst, TileDataSrc>(
                        dstPtr + i * dstAlignCols + numLoop * dstOffset,
                        srcPtr + i * srcAlignCols + numLoop * srcOffset,
                        src1,
                        mode,
                        REPEAT_MAX,
                        1,
                        1,
                        8,
                        8);
            }
        }
    }

    template <typename TileDataDst, typename TileDataSrc0, typename T>
    PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc0 &src0, T src1, CmpMode cmpMode) {
        static_assert(std::is_same<typename TileDataSrc0::DType, int32_t>::value ||
                std::is_same<typename TileDataSrc0::DType, float>::value ||
                std::is_same<typename TileDataSrc0::DType, half>::value,
                "TCMPS: Invalid data type.");
        static_assert(TileDataDst::isRowMajor, "TCMPS: not supported Layout type");

        static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tile must be TileType::Vec.");
        static_assert(TileDataDst::ValidCol <= TileDataDst::Cols, "Number of valid columns for dst must not be greater than number of tile columns.");
        static_assert(TileDataDst::ValidRow <= TileDataDst::Rows, "Number of valid rows for dst must not be greater than number of tile rows.");

        static_assert(TileDataSrc0::Loc == TileType::Vec, "TileType of src tile must be TileType::Vec.");
        static_assert(TileDataSrc0::ValidCol <= TileDataSrc0::Cols, "Number of valid columns for scr must not be greater than number of tile columns.");
        static_assert(TileDataSrc0::ValidRow <= TileDataSrc0::Rows, "Number of valid rows for src must not be greater than number of tile rows.");
        PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
        
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataSrc0::DType);
        unsigned numRepeatPerLine = CeilDivision(src0.GetValidCol(), elementsPerRepeat);
        unsigned validRow = src0.GetValidRow();
        
        TCmps<TileDataDst, TileDataSrc0, T>(dst.data(), src0.data(), src1, cmpMode, numRepeatPerLine, validRow, elementsPerRepeat);
    }
}
#endif
