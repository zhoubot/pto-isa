/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDBIN_HPP
#define TCOLEXPANDBIN_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename Op, typename T, unsigned BlockSizeElem, unsigned RowStride>
PTO_INTERNAL void TColExpandBinaryCountMode(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                            unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * RowStride;
        Op::ColExpandBinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned RowStride>
PTO_INTERNAL void TColExpandBinaryNormMode(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                           unsigned validRow, unsigned validCol)
{
    constexpr uint8_t repeatStride = (uint8_t)(RowStride / BlockSizeElem);
    if constexpr (RowStride <= ElementsPerRepeat) {
        SetContMaskByDType<T>(validCol);
        Op::ColExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride, repeatStride);
        SetFullVecMaskByDType<T>();
    } else {
        unsigned numLoop = validCol / ElementsPerRepeat;
        unsigned numRemainAfterLoop = validCol % ElementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            Op::ColExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride, repeatStride);
            dstPtr += ElementsPerRepeat;
            src0Ptr += ElementsPerRepeat;
            src1Ptr += ElementsPerRepeat;
        }
        if (numRemainAfterLoop) {
            SetContMaskByDType<T>(numRemainAfterLoop);
            Op::ColExpandBinInstr(dstPtr, src0Ptr, src1Ptr + numLoop * ElementsPerRepeat, validRow, repeatStride,
                                  repeatStride, repeatStride);
            SetFullVecMaskByDType<T>();
        }
    }
}

template <typename Op, typename TileData, typename TileDataSrc, unsigned ElementsPerRepeat, unsigned BlockSizeElem,
          unsigned RowStride>
__tf__ PTO_INTERNAL void ColExpandBinaryInstr(typename TileData::TileDType __out__ dst,
                                              typename TileData::TileDType __in__ src0,
                                              typename TileDataSrc::TileDType __in__ src1, unsigned validRow,
                                              unsigned validCol)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        TColExpandBinaryNormMode<Op, T, ElementsPerRepeat, BlockSizeElem, RowStride>(dstPtr, src0Ptr, src1Ptr, validRow,
                                                                                     validCol);
    } else {
        TColExpandBinaryCountMode<Op, T, BlockSizeElem, RowStride>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }
}

} // namespace pto
#endif