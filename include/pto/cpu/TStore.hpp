/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP

#include <pto/common/constants.hpp>
#include <pto/common/type.hpp>
#include <cassert>
#include "pto/cpu/parallel.hpp"

namespace pto {

    template <typename GlobalData, typename TileData, std::enable_if_t<TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void StorePlainMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol, size_t idx3) {
        size_t offsetSrcBase =  idx3*gShape3*TileData::Cols;
        cpu::parallel_for_1d(0, static_cast<std::size_t>(gShape3), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t r) {
            const std::size_t srcBase = offsetSrcBase + r * TileData::Cols;
            const std::size_t dstBase = r * static_cast<std::size_t>(gStride3);
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t c = 0; c < static_cast<std::size_t>(gShape4); c++) {
                dst[dstBase + c * static_cast<std::size_t>(gStride4)] = src[srcBase + c];
            }
        });
    }
    template <typename GlobalData, typename TileData, std::enable_if_t<!TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void StorePlainMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol, size_t idx3) {
        size_t offsetSrcBase =  idx3*gShape4*TileData::Rows;
        cpu::parallel_for_1d(0, static_cast<std::size_t>(gShape4), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t c) {
            const std::size_t srcBase = offsetSrcBase + c * TileData::Rows;
            const std::size_t dstStride4 = static_cast<std::size_t>(gStride4);
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                dst[r * static_cast<std::size_t>(gStride3) + c * dstStride4] = src[srcBase + r];
            }
        });
    }

    template <typename GlobalData, typename TileData>
    __tf__  PTO_INLINE void StorePlain(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        int64_t srcStride1 = gShape2;
        int64_t srcStride0 = gShape1 * srcStride1;
        for (uint32_t i = 0; i < gShape0; i++) {
            int64_t srcAddr0 = i * srcStride0;
            int64_t dstAddr0 = i * gStride0;
            for (uint32_t j = 0; j < gShape1; j++) {
                int64_t srcAddr1 = j * srcStride1;
                int64_t dstAddr1 = j * gStride1;
                for (uint32_t k = 0; k < gShape2; k++) {
                    size_t offsetDstBase = dstAddr0 + dstAddr1 + k * gStride2;
                    StorePlainMatrix<GlobalData, TileData>(dst+offsetDstBase, src, gShape3, gShape4, gStride3, gStride4, validRow, validCol, srcAddr0 + srcAddr1 + k);
                }
            }
        }
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void StoreSubfractalMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol) {
        // Zn layout
        cpu::parallel_for_1d(0, static_cast<std::size_t>(gShape4), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t c) {
            size_t subTileC = c / TileData::InnerCols;
            size_t innerC = c % TileData::InnerCols;
            for (size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                size_t subTileR = r / TileData::InnerRows;
                size_t innerR = r % TileData::InnerRows;

                size_t tile_idx = subTileR * TileData::Cols * TileData::InnerRows +
                    subTileC * TileData::InnerNumel + innerC * TileData::InnerRows + innerR;

                size_t gd_idx = r * static_cast<std::size_t>(gStride3) + c * static_cast<std::size_t>(gStride4);
                dst[gd_idx] = src[tile_idx];
            }
        });
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<!TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void StoreSubfractalMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol) {
        // Nz layout
        cpu::parallel_for_1d(0, static_cast<std::size_t>(gShape4), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t c) {
            size_t subTileC = c / TileData::InnerCols;
            size_t innerC = c % TileData::InnerCols;
            for (size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                size_t subTileR = r / TileData::InnerRows;
                size_t innerR = r % TileData::InnerRows;

                size_t tile_idx = subTileC * TileData::Rows * TileData::InnerCols +
                    subTileR * TileData::InnerNumel + innerR * TileData::InnerCols + innerC;
                size_t gd_idx = r * static_cast<std::size_t>(gStride3) + c * static_cast<std::size_t>(gStride4);

                dst[gd_idx] = src[tile_idx];
            }
        });
    }

    template <typename GlobalData, typename TileData>
    __tf__  PTO_INLINE void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        if constexpr (GlobalData::layout == pto::Layout::NZ) {
            // NZ: validRow = shape2*shape3 (shape3 typically 16), validCol = shape0*shape1*shape4 (shape4=C0).
            PTO_CPU_ASSERT(validRow == gShape2 * gShape3 && validCol == gShape0 * gShape1 * gShape4,
                           "Fix: TSTORE(NZ) requires validRow==shape2*shape3 and validCol==shape0*shape1*shape4");
            PTO_CPU_ASSERT((!TileData::isRowMajor) && (TileData::SFractal == SLayout::RowMajor),
                           "Fix: TSTORE(NZ->NZ) requires TileData in Nz layout");
        } else {
            PTO_CPU_ASSERT((gShape0*gShape1*gShape2*gShape3 == validRow && gShape4==validCol && TileData::isRowMajor) ||
                (gShape0*gShape1*gShape2*gShape4 == validCol && gShape3==validRow && !TileData::isRowMajor),
                "Fix: TSTORE ND/DN shape/valid mismatch");
        }
        if constexpr (GlobalData::layout == pto::Layout::NZ) {
            // Tile NZ -> Global NZ: map 2D (r,c) to 5D NZ indices using (shape0..4, stride0..4).
            // Prefer static enforcement of NZ base shape when available (matches NPU CheckNzFormat).
            static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW &&
                              GlobalData::staticShape[4] == C0_SIZE_BYTE / sizeof(typename TileData::DType),
                          "Fix: GlobalTensor NZ requires last dims [16, 32/sizeof(T)]");
            for (std::size_t r = 0; r < static_cast<std::size_t>(validRow); ++r) {
                const std::size_t idx2 = r / static_cast<std::size_t>(gShape3);
                const std::size_t idx3 = r % static_cast<std::size_t>(gShape3);
                for (std::size_t c = 0; c < static_cast<std::size_t>(validCol); ++c) {
                    const std::size_t denom = static_cast<std::size_t>(gShape1) * static_cast<std::size_t>(gShape4);
                    const std::size_t idx0 = c / denom;
                    const std::size_t rem = c % denom;
                    const std::size_t idx1 = rem / static_cast<std::size_t>(gShape4);
                    const std::size_t idx4 = rem % static_cast<std::size_t>(gShape4);
                    const std::size_t gd_idx =
                        idx0 * static_cast<std::size_t>(gStride0) +
                        idx1 * static_cast<std::size_t>(gStride1) +
                        idx2 * static_cast<std::size_t>(gStride2) +
                        idx3 * static_cast<std::size_t>(gStride3) +
                        idx4 * static_cast<std::size_t>(gStride4);
                    dst[gd_idx] = src[GetTileElementOffset<TileData>(r, c)];
                }
            }
        } else if(TileData::SFractal == SLayout::NoneBox) {
            StorePlain<GlobalData, TileData>(dst, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2,
                gStride3, gStride4, validRow, validCol);
        } else {
            assert(gShape0==1 && gShape1==1 && gShape2==1 && "Nz,Zn -> ND,DN convertion does support only 2D GMs");
            StoreSubfractalMatrix<GlobalData, TileData>(dst, src, gShape3, gShape4, gStride3, gStride4, validRow, validCol);
        }
    }

    template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
     void TSTORE_IMPL(GlobalData &dst, TileData &src)
    {
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype!");
        static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN || GlobalData::layout == pto::Layout::NZ,
            "Only ND/DN/NZ GLobal Tensors are currently supported");
        TStore<GlobalData, TileData>(dst.data(),
            src.data(),
            dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1),
            dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3),
            dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0),
            dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2),
            dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4),
            src.GetValidRow(),
            src.GetValidCol());
    }

    template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
    __aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
    {
        (void)preQuantScalar;
        TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
    }

    template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone>
    __aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &fp)
    {
        (void)fp;
        TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
    }
}
#endif
