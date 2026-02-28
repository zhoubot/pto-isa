/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP
#define TLOAD_HPP

#include <unistd.h>
#include <cassert>
#include <pto/common/type.hpp>
#include <pto/common/constants.hpp>
#include "pto/cpu/parallel.hpp"

namespace pto {
    template <typename TileData>
    AICORE constexpr typename TileData::DType getPadValue()
    {    
        switch (TileData::PadVal)
        {
            case PadValue::Null:
            case PadValue::Zero: return typename TileData::DType(0);
            case PadValue::Min:
                if constexpr(std::numeric_limits<typename TileData::DType>::has_infinity) {
                    return -std::numeric_limits<typename TileData::DType>::infinity();
                } else {
                    return std::numeric_limits<typename TileData::DType>::min();
                }
            case PadValue::Max:
                if constexpr(std::numeric_limits<typename TileData::DType>::has_infinity) {
                    return std::numeric_limits<typename TileData::DType>::infinity();
                } else {
                    return std::numeric_limits<typename TileData::DType>::max();
                }
        }
        return 0;
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void LoadPlainMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol, size_t idx3) {
        size_t offsetDstBase =  idx3*gShape3*TileData::Cols;
        cpu::parallel_for_1d(0, static_cast<std::size_t>(gShape3), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t r) {
            const std::size_t dstBase = offsetDstBase + r * TileData::Cols;
            const std::size_t srcBase = r * static_cast<std::size_t>(gStride3);
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t c = 0; c < static_cast<std::size_t>(gShape4); c++) {
                dst[dstBase + c] = src[srcBase + c * static_cast<std::size_t>(gStride4)];
            }
        });
    }
    template <typename GlobalData, typename TileData, std::enable_if_t<!TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void LoadPlainMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol, size_t idx3) {
        size_t offsetDstBase =  idx3*gShape4*TileData::Rows;
        cpu::parallel_for_1d(0, static_cast<std::size_t>(gShape4), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t c) {
            const std::size_t dstBase = offsetDstBase + c * TileData::Rows;
            const std::size_t srcStride4 = static_cast<std::size_t>(gStride4);
            for (std::size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                dst[dstBase + r] = src[r * static_cast<std::size_t>(gStride3) + c * srcStride4];
            }
        });
    }

    template <typename GlobalData, typename TileData>
    __tf__  PTO_INLINE void LoadPlain(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol) {
        int64_t dstStride1 = gShape2;
        int64_t dstStride0 = gShape1 * dstStride1;

        for (uint32_t i = 0; i < gShape0; i++) {
            int64_t dstAddr0 = i * dstStride0;
            int64_t srcAddr0 = i * gStride0;
            for (uint32_t j = 0; j < gShape1; j++) {
                int64_t dstAddr1 = j * dstStride1;
                int64_t srcAddr1 = j * gStride1;
                for (uint32_t k = 0; k < gShape2; k++) {
                    size_t offsetSrcBase = srcAddr0 + srcAddr1 + k * gStride2;
                    LoadPlainMatrix<GlobalData, TileData>(dst, src+offsetSrcBase, gShape3, gShape4, gStride3, gStride4, validRow, validCol, dstAddr0 + dstAddr1 + k);
                }
            }
        }
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void LoadSubfractalMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
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
                dst[tile_idx] = src[gd_idx];
            }
        });
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<!TileData::isRowMajor, int> = 0>
    __tf__  PTO_INLINE void LoadSubfractalMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
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

                dst[tile_idx] = src[gd_idx];
            }
        });
    }

    template <typename TileData, typename GlobalData>
    __tf__ AICORE void TLoad(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        if constexpr (GlobalData::layout == pto::Layout::NZ) {
            // NZ: validRow = shape2*shape3 (shape3 typically 16), validCol = shape0*shape1*shape4 (shape4=C0).
            PTO_CPU_ASSERT(validRow == gShape2 * gShape3 && validCol == gShape0 * gShape1 * gShape4,
                           "Fix: TLOAD(NZ) requires validRow==shape2*shape3 and validCol==shape0*shape1*shape4");
            PTO_CPU_ASSERT((!TileData::isRowMajor) && (TileData::SFractal == SLayout::RowMajor),
                           "Fix: TLOAD(NZ->NZ) requires TileData in Nz layout");
        } else {
            PTO_CPU_ASSERT((gShape0*gShape1*gShape2*gShape3 == validRow && gShape4==validCol && TileData::isRowMajor) ||
                (gShape0*gShape1*gShape2*gShape4 == validCol && gShape3==validRow && !TileData::isRowMajor),
                "Fix: TLOAD ND/DN shape/valid mismatch");
        }

        // Filling padding
        std::fill(dst,dst+(TileData::Cols*TileData::Rows),getPadValue<TileData>());

        //Filling data
        if constexpr (GlobalData::layout == pto::Layout::NZ) {
            // Global NZ -> Tile NZ: map 2D (r,c) to 5D NZ indices using (shape0..4, stride0..4).
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
                    dst[GetTileElementOffset<TileData>(r, c)] = src[gd_idx];
                }
            }
        } else if (TileData::SFractal == SLayout::NoneBox) {
            LoadPlain<GlobalData, TileData>(dst, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2,
                gStride3, gStride4, validRow, validCol);
        } else {
            assert(gShape0==1 && gShape1==1 && gShape2==1 && "ND,DN -> Nz,Zn convertion does support only 2D GMs");
            LoadSubfractalMatrix<GlobalData, TileData>(dst, src, gShape3, gShape4, gStride3, gStride4, validRow, validCol);
        }
    }

    template <typename TileData, typename GlobalData>
    AICORE void TLOAD_IMPL(TileData &dst, GlobalData &src)
    {
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype");
        static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN ||
                      GlobalData::layout == pto::Layout::NZ,
                      "Only ND/DN/NZ Global Tensors are supported in CPU_SIM TLOAD");

        // Strict contract: empty valid region is NOT allowed.
        PTO_CPU_ASSERT(dst.GetValidRow() > 0 && dst.GetValidCol() > 0,
                       "Fix: TLOAD requires dst validRow/validCol > 0");
        PTO_CPU_ASSERT(src.GetShape(pto::GlobalTensorDim::DIM_0) > 0 &&
                       src.GetShape(pto::GlobalTensorDim::DIM_1) > 0 &&
                       src.GetShape(pto::GlobalTensorDim::DIM_2) > 0 &&
                       src.GetShape(pto::GlobalTensorDim::DIM_3) > 0 &&
                       src.GetShape(pto::GlobalTensorDim::DIM_4) > 0,
                       "Fix: TLOAD requires all src shape dims > 0");

        TLoad<TileData, GlobalData>(dst.data(),
            src.data(),
            src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1),
            src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3),
            src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0),
            src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2),
            src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4),
            dst.GetValidRow(),
            dst.GetValidCol());

    }
}
#endif
