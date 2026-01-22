/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSEL_HPP
#define TSEL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
enum class SELMODE : uint8_t {
    VSEL_CMPMASK_SPR = 0,
    VSEL_TENSOR_SCALAR_MODE = 1,
    VSEL_TENSOR_TENSOR_MODE = 2,
};

template <typename TileData, typename MaskTile, unsigned rowStride, unsigned maskRowStride>
__tf__ PTO_INTERNAL void TSel(typename TileData::TileDType __out__ dst, typename MaskTile::TileDType __in__ selMask,
    typename TileData::TileDType __in__ src0, typename TileData::TileDType __in__ src1, unsigned validRow,
    unsigned validCol) {
    using T = typename std::conditional<sizeof(typename TileData::DType) == 4, float, half>::type;
    if constexpr (sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 2) {
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
        uint32_t maskPtr = static_cast<uint32_t>(
            reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t *>(__cce_get_tile_ptr(selMask))));
        __ubuf__ uint32_t *cmpMaskPtr =
            reinterpret_cast<__ubuf__ uint32_t *>(get_imm(TMP_UB_OFFSET)); // 8KB tmpbuf addr

        set_mask_count();
        for (unsigned i = 0; i < validRow; i++) {
            set_vector_mask(0, BLOCK_BYTE_SIZE);
            vector_dup(cmpMaskPtr, (uint32_t)(maskPtr + i * maskRowStride), 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_cmpmask(cmpMaskPtr);
            pipe_barrier(PIPE_V);
            set_vector_mask(0, validCol);
            vsel((__ubuf__ T *)(dstPtr + i * rowStride), (__ubuf__ T *)(src0Ptr + i * rowStride),
                (__ubuf__ T *)(src1Ptr + i * rowStride), 1, 1, 1, 1, 8, 8, 8, SELMODE::VSEL_TENSOR_TENSOR_MODE);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        static_assert(sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 2,
            "Fix: TSEL has invalid data type.");
    }
}

template <typename TileData, typename MaskTile>
PTO_INTERNAL void TSEL_IMPL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1) {
    static_assert(TileData::isRowMajor, "Fix: TSEL has not supported layout type.");
    constexpr unsigned rowStride = TileData::RowStride;
    constexpr unsigned maskRowStride = MaskTile::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TSel<TileData, MaskTile, rowStride, maskRowStride>(
        dst.data(), selMask.data(), src0.data(), src1.data(), validRow, validCol);
}
} // namespace pto
#endif