/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDEQUANT_HPP
#define TDEQUANT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, unsigned dstStride, unsigned srcStride,
          unsigned paraStride>
__tf__ PTO_INTERNAL void TDeQuant(typename TileDataDst::TileDType __out__ dst,
                                  typename TileDataSrc::TileDType __in__ src,
                                  typename TileDataPara::TileDType __in__ scale,
                                  typename TileDataPara::TileDType __in__ offset, unsigned validRows,
                                  unsigned validCols)
{
    using dstType = typename TileDataDst::DType;
    using srcType = typename TileDataSrc::DType;
    __ubuf__ dstType *dstPtr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst);
    __ubuf__ srcType *srcPtr = (__ubuf__ srcType *)__cce_get_tile_ptr(src);
    __ubuf__ dstType *scalePtr = (__ubuf__ dstType *)__cce_get_tile_ptr(scale);
    __ubuf__ dstType *offsetPtr = (__ubuf__ dstType *)__cce_get_tile_ptr(offset);

    constexpr unsigned srcElementsPerRepeat = REPEAT_BYTE / sizeof(srcType);
    constexpr unsigned dstElementsPerRepeat = REPEAT_BYTE / sizeof(dstType);
    uint16_t repeatTimes = CeilDivision(validCols, dstElementsPerRepeat);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<dstType, DistVST::DIST_NORM>())>();

    __VEC_SCOPE__
    {
        RegTensor<srcType> reg_src;
        RegTensor<dstType> reg_dst, reg_scale, reg_offset;
        RegTensor<int32_t> reg_int;
        MaskReg preg = CreatePredicate<dstType>(validCols);
        uint32_t lenSrc = srcElementsPerRepeat;
        MaskReg pregSrc = CreatePredicate<srcType>(lenSrc);

        for (uint16_t i = 0; i < (uint16_t)validRows; i++) {
            vlds(reg_scale, scalePtr, i * paraStride, BRC_B32);
            vlds(reg_offset, offsetPtr, i * paraStride, BRC_B32);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                if constexpr (sizeof(srcType) == 1) {
                    vlds(reg_src, srcPtr, i * srcStride + j * dstElementsPerRepeat, UNPK4_B8);
                    vcvt(reg_int, reg_src, pregSrc, PART_P0);
                    vcvt(reg_dst, reg_int, preg, ROUND_Z);
                } else {
                    vlds(reg_src, srcPtr, i * srcStride + j * dstElementsPerRepeat, UNPK_B16);
                    vcvt(reg_dst, reg_src, pregSrc, PART_EVEN);
                }
                vsub(reg_dst, reg_dst, reg_offset, preg, MODE_ZEROING);
                vmul(reg_dst, reg_dst, reg_scale, preg, MODE_ZEROING);
                vsts(reg_dst, dstPtr, i * dstStride + j * dstElementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDeQuantCheck(const TileDataDst &dst, const TileDataSrc &src, unsigned scaleRow, unsigned offsetRow)
{
    using dstType = typename TileDataDst::DType;
    using srcType = typename TileDataSrc::DType;
    using paraType = typename TileDataPara::DType;
    static_assert(
        std::is_same_v<dstType, float> && (std::is_same_v<srcType, int16_t> || std::is_same_v<srcType, int8_t>),
        "Fix: TDEQUANT has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "Fix: TDEQUANT only support row major layout.");
    static_assert(std::is_same_v<dstType, paraType>, "Fix: TDEQUANT tile dst, para tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src.GetValidRow() == validRows && src.GetValidCol() == validCols,
               "Fix: TDEQUANT input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(scaleRow == validRows && offsetRow == validRows,
               "Fix: TDEQUANT input tile para valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDEQUANT_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset)
{
    TDeQuantCheck<TileDataDst, TileDataSrc, TileDataPara>(dst, src, scale.GetValidRow(), offset.GetValidRow());
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned paraStride = TileDataPara::RowStride;

    TDeQuant<TileDataDst, TileDataSrc, TileDataPara, dstStride, srcStride, paraStride>(
        dst.data(), src.data(), scale.data(), offset.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
