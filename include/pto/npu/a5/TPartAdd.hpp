/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPATIALADD_HPP
#define TPATIALADD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename T>
struct PartAddOp {
    PTO_INTERNAL static void PartInstr(RegTensor<T> &dst, RegTensor<T> &src0, RegTensor<T> &src1, MaskReg preg)
    {
        vadd(dst, src0, src1, preg, MODE_ZEROING);
    }
};

template <typename T, typename TileDataDst, typename TileDataSrc, unsigned blockSizeElem, unsigned dstStride,
          unsigned srcStride>
PTO_INTERNAL void TPartCopyInstr(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, uint64_t validRow, uint64_t validCol,
                                 uint64_t startRow)
{
    validRow -= startRow;
    srcPtr += startRow * TileDataSrc::RowStride;
    dstPtr += startRow * TileDataDst::RowStride;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        MaskReg preg;
        uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, srcPtr, j * elementsPerRepeat + i * srcStride, NORM);
                vsts(vreg0, dstPtr, j * elementsPerRepeat + i * dstStride, distValue, preg);
            }
        }
    }
} // end of tf

template <typename Op, typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
          unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstStride, unsigned src0Stride,
          unsigned src1Stride>
PTO_INTERNAL void TPartOpInstr(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow,
                               unsigned validCol)
{
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<T> vreg0, vreg1, vreg2;
        uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr, j * elementsPerRepeat + i * src0Stride, NORM);
                vlds(vreg1, src1Ptr, j * elementsPerRepeat + i * src1Stride, NORM);
                Op::PartInstr(vreg2, vreg0, vreg1, preg);
                vsts(vreg2, dstPtr, j * elementsPerRepeat + i * dstStride, distValue, preg);
            }
        }
    } // end VF
}

template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
__tf__ PTO_INTERNAL void TPartOp(typename TileDataDst::TileDType __out__ dst,
                                 typename TileDataSrc0::TileDType __in__ src0,
                                 typename TileDataSrc1::TileDType __in__ src1, unsigned src0ValidRow,
                                 unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol,
                                 unsigned dstValidRow, unsigned dstValidCol)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    bool condSrc1EqDst = (src1ValidRow == dstValidRow && src1ValidCol == dstValidCol);
    bool condSrc1RowLtDst = (src1ValidRow < dstValidRow && src1ValidCol == dstValidCol);
    bool condSrc1ColLtDst = (src1ValidRow <= dstValidRow && src1ValidCol < dstValidCol);

    if (condSrc1EqDst) { // src0 == src1 == dst
        TPartOpInstr<Op, T, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride,
                     src0RowStride, src1RowStride>(dstPtr, src0Ptr, src1Ptr, dstValidRow, dstValidCol);
    } else if (condSrc1ColLtDst) { // src1Col < dstCol
        TPartCopyInstr<T, TileDataDst, TileDataSrc0, blockSizeElem, dstRowStride, src0RowStride>(
            dstPtr, src0Ptr, src0ValidRow, dstValidCol, 0);
        if (src1ValidCol != 0) {
            TPartOpInstr<Op, T, TileDataDst, TileDataSrc1, TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride,
                         src1RowStride, dstRowStride>(dstPtr, src1Ptr, dstPtr, src1ValidRow, src1ValidCol);
        }
    } else if (condSrc1RowLtDst) { // src1Row < dstRow
        if (src1ValidRow != 0) {
            TPartOpInstr<Op, T, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride,
                         src0RowStride, src1RowStride>(dstPtr, src0Ptr, src1Ptr, src1ValidRow, src1ValidCol);
        }
        TPartCopyInstr<T, TileDataDst, TileDataSrc0, blockSizeElem, dstRowStride, src0RowStride>(
            dstPtr, src0Ptr, src0ValidRow, dstValidCol, src1ValidRow);
    } // unsupport other conditions
}

template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTOP_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    if (dstValidRow == 0 || dstValidCol == 0) {
        return;
    }
    bool condSrc0EqDst = (src0ValidRow == dstValidRow && src0ValidCol == dstValidCol);
    bool condSrc1EqDst = (src1ValidRow == dstValidRow && src1ValidCol == dstValidCol);
    PTO_ASSERT(
        condSrc0EqDst || condSrc1EqDst,
        "Fix: TPARTADD/MUL At most one entry in the valid-rows and valid-cols of src0 and src1 is smaller than dst.");

    if (condSrc0EqDst) { // Ensure that the src equal to dst comes before the other
        TPartOp<Op, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride,
                src0RowStride, src1RowStride>(dst.data(), src0.data(), src1.data(), src0ValidRow, src0ValidCol,
                                              src1ValidRow, src1ValidCol, dstValidRow, dstValidCol);
    } else if (condSrc1EqDst) {
        TPartOp<Op, TileDataDst, TileDataSrc1, TileDataSrc0, elementsPerRepeat, blockSizeElem, dstRowStride,
                src1RowStride, src0RowStride>(dst.data(), src1.data(), src0.data(), src1ValidRow, src1ValidCol,
                                              src0ValidRow, src0ValidCol, dstValidRow, dstValidCol);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TPARTADD src and dst data type is different!");
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, float>::value ||
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, half>::value ||
            std::is_same<T, bfloat16_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value,
        "Fix: TPARTADD Invalid data type.");
    TPARTOP_IMPL<PartAddOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}
} // namespace pto
#endif
