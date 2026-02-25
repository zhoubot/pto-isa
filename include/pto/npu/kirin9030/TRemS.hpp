/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMS_HPP
#define TREMS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"

namespace pto {

template <typename T>
struct RemSOp {
    PTO_INTERNAL static void BinSInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, T scalar, MaskReg &preg)
    {
        if constexpr (std::is_same<T, float>::value) {
            RegTensor<T> reg_src1;
            RegTensor<int32_t> reg_tmp;
            vdup(reg_src1, scalar, preg, MODE_ZEROING);
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vcvt(reg_tmp, reg_dst, preg, ROUND_Z, RS_ENABLE);
            vcvt(reg_dst, reg_tmp, preg, ROUND_R);
            vmuls(reg_dst, reg_dst, scalar, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<T> reg_src1, reg_dst_even, reg_dst_odd;
            RegTensor<int32_t> reg_s32_even, reg_s32_odd;
            RegTensor<float> reg_src0_even, reg_src0_odd, reg_src1_even, reg_src1_odd, reg_even, reg_odd;
            vdup(reg_src1, scalar, preg, MODE_ZEROING);
            vcvt(reg_src0_even, reg_src0, preg, PART_EVEN);
            vcvt(reg_src1_even, reg_src1, preg, PART_EVEN);
            vdiv(reg_even, reg_src0_even, reg_src1_even, preg, MODE_ZEROING);
            vcvt(reg_s32_even, reg_even, preg, ROUND_Z, RS_ENABLE);
            vcvt(reg_even, reg_s32_even, preg, ROUND_R);
            vmuls(reg_even, reg_even, (float)scalar, preg, MODE_ZEROING);
            vsub(reg_even, reg_src0_even, reg_even, preg, MODE_ZEROING);
            vcvt(reg_dst_even, reg_even, preg, ROUND_Z, RS_ENABLE, PART_EVEN);

            vcvt(reg_src0_odd, reg_src0, preg, PART_ODD);
            vcvt(reg_src1_odd, reg_src1, preg, PART_ODD);
            vdiv(reg_odd, reg_src0_odd, reg_src1_odd, preg, MODE_ZEROING);
            vcvt(reg_s32_odd, reg_odd, preg, ROUND_Z, RS_ENABLE);
            vcvt(reg_odd, reg_s32_odd, preg, ROUND_R);
            vmuls(reg_odd, reg_odd, (float)scalar, preg, MODE_ZEROING);
            vsub(reg_odd, reg_src0_odd, reg_odd, preg, MODE_ZEROING);
            vcvt(reg_dst_odd, reg_odd, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);
        } else {
            RegTensor<T> reg_src1;
            vdup(reg_src1, scalar, preg, MODE_ZEROING);
            vmod(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL OP_NAME(TREMS)
    OP_TYPE(element_wise) void TRemS(typename TileDataDst::TileDType __out__ dst,
                                     typename TileDataSrc::TileDType __in__ src, typename TileDataSrc::DType scalar,
                                     unsigned kValidRows, unsigned kValidCols,
                                     VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    BinaryInstr<RemSOp<T>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, TileDataDst::RowStride,
                TileDataSrc::RowStride>(dstPtr, srcPtr, scalar, kValidRows, kValidCols, version);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TRemSCheck()
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc::DType>,
                  "Fix: The TREMS data type must be same of src and dst");
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>,
                  "Fix: TREMS Invalid data type");
    static_assert((TileDataDst::Loc == TileType::Vec) && (TileDataSrc::Loc == TileType::Vec),
                  "Fix: TileType of dst and src tiles must be TileType::Vec.");
    static_assert((TileDataDst::ValidCol <= TileDataDst::Cols) && (TileDataDst::ValidRow <= TileDataDst::Rows) &&
                      (TileDataSrc::ValidCol <= TileDataSrc::Cols) && (TileDataSrc::ValidRow <= TileDataSrc::Rows),
                  "Fix: Number of valid columns and rows must not be greater than number of tile columns and rows.");
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TREMS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    TRemSCheck<TileDataDst, TileDataSrc>();
    PTO_ASSERT((src.GetValidCol() == validCol) && (src.GetValidRow() == validRow),
               "Number of validColumns and validRows of src and dst must be the same.");
    TRemS<TileDataDst, TileDataSrc>(dst.data(), src.data(), scalar, validRow, validCol);
}
} // namespace pto
#endif
