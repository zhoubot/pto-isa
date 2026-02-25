/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/kirin9030/common.hpp>
#include <pto/npu/kirin9030/utils.hpp>
#include <pto/npu/kirin9030/TBinOp.hpp>
#include <pto/common/debug.h>

namespace pto {

template <typename T>
struct RemOp {
    PTO_INTERNAL static void BinInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                      MaskReg &preg)
    {
        if constexpr (std::is_same<T, float>::value) {
            RegTensor<int32_t> reg_tmp;
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vcvt(reg_tmp, reg_dst, preg, ROUND_Z, RS_ENABLE);
            vcvt(reg_dst, reg_tmp, preg, ROUND_R);
            vmul(reg_dst, reg_dst, reg_src1, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<int32_t> reg_tmp_even, reg_tmp_odd;
            RegTensor<float> reg_tmp_even0, reg_tmp_even1, reg_tmp_even2, reg_tmp_odd0, reg_tmp_odd1, reg_tmp_odd2;
            RegTensor<T> reg_dst_even, reg_dst_odd;
            vcvt(reg_tmp_even0, reg_src0, preg, PART_EVEN);
            vcvt(reg_tmp_even1, reg_src1, preg, PART_EVEN);
            vdiv(reg_tmp_even2, reg_tmp_even0, reg_tmp_even1, preg, MODE_ZEROING);
            vcvt(reg_tmp_even, reg_tmp_even2, preg, ROUND_Z, RS_ENABLE);
            vcvt(reg_tmp_even2, reg_tmp_even, preg, ROUND_R);
            vmul(reg_tmp_even2, reg_tmp_even2, reg_tmp_even1, preg, MODE_ZEROING);
            vsub(reg_tmp_even2, reg_tmp_even0, reg_tmp_even2, preg, MODE_ZEROING);
            vcvt(reg_dst_even, reg_tmp_even2, preg, ROUND_Z, RS_ENABLE, PART_EVEN);

            vcvt(reg_tmp_odd0, reg_src0, preg, PART_ODD);
            vcvt(reg_tmp_odd1, reg_src1, preg, PART_ODD);
            vdiv(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd1, preg, MODE_ZEROING);
            vcvt(reg_tmp_odd, reg_tmp_odd2, preg, ROUND_Z, RS_ENABLE);
            vcvt(reg_tmp_odd2, reg_tmp_odd, preg, ROUND_R);
            vmul(reg_tmp_odd2, reg_tmp_odd2, reg_tmp_odd1, preg, MODE_ZEROING);
            vsub(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd2, preg, MODE_ZEROING);
            vcvt(reg_dst_odd, reg_tmp_odd2, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);
        } else {
            vmod(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
          unsigned BlockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TREM)
    OP_TYPE(element_wise) void TRem(typename TileDataDst::TileDType __out__ dst,
                                    typename TileDataSrc0::TileDType __in__ src0,
                                    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows,
                                    unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    BinaryInstr<RemOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
        dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TRemCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, half>::value || std::is_same<T, float>::value || std::is_same<T, uint16_t>::value ||
                      std::is_same<T, int16_t>::value || std::is_same<T, uint32_t>::value ||
                      std::is_same<T, int32_t>::value,
                  "Fix: TREM has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
                  "Fix: TREM only support row major layout.");
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TREM input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
               "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
               "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TREM_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    TRemCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TRem<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
