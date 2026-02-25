/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGATHER_HPP
#define PTO_COMM_TGATHER_HPP

#include <type_traits>

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/pto_instr.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TGATHER_IMPL: Gather operation - root collects data from all ranks
//
// The calling NPU is the root and gathers data from all ranks, concatenating
// the results along DIM_3 (row dimension) into a local output buffer.
//
// Each rank r contributes data of shape (D0, D1, D2, H, W). The destination
// tensor has shape (D0, D1, D2, N*H, W), where rank r's data is placed at
// rows [r*H, (r+1)*H).
//
// When the per-rank GlobalTensor exceeds the UB tile capacity in rows and/or
// columns, the transfer is automatically chunked via 2D sliding:
//   - Outer dimensions (DIM_0, DIM_1, DIM_2) are iterated explicitly.
//   - DIM_3 (rows) is split into tileValidRow-sized chunks.
//   - DIM_4 (cols) is split into tileValidCol-sized chunks.
//
// Constraints for chunked mode:
//   - If TileData has static ValidRow, per-rank DIM_3 must be divisible by ValidRow.
//   - If TileData has static ValidCol, DIM_4 must be divisible by ValidCol.
//   - All source tensors in the ParallelGroup are assumed to have the same shape/strides.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TGATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                               TileData &stagingTileData)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGATHER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGATHER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    // Get per-rank dimensions (from first rank, all assumed same)
    const int gShape0 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_3); // H (per-rank rows)
    const int gShape4 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_4); // W

    const int perRankRows = gShape3;
    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;
    const int tileValidRow = stagingTileData.GetValidRow();
    const int tileValidCol = stagingTileData.GetValidCol();

    PTO_ASSERT(tileValidRow > 0, "TGATHER: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TGATHER: tileValidCol must be greater than 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    // ---- Simple path: per-rank data fits in UB tile ----
    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        if (nranks == 1) {
            // Single rank: direct copy, no offset needed
            TLOAD(stagingTileData, parallelGroup[0]);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobalData, stagingTileData);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            return;
        }

        // Multiple ranks: need destination views with per-rank offset
        const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);

        using DynShape5D = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
        using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
        using DstViewT = GlobalTensor<T, DynShape5D, DynStride, GlobalDstData::layout>;

        DynShape5D perRankShape(gShape0, gShape1, gShape2, gShape3, gShape4);
        DynStride dstViewStride(dstGlobalData.GetStride(GlobalTensorDim::DIM_0),
                                dstGlobalData.GetStride(GlobalTensorDim::DIM_1),
                                dstGlobalData.GetStride(GlobalTensorDim::DIM_2), dstStride3,
                                dstGlobalData.GetStride(GlobalTensorDim::DIM_4));

        for (int r = 0; r < nranks; ++r) {
            TLOAD(stagingTileData, parallelGroup[r]);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

            int64_t dstOffset = static_cast<int64_t>(r) * perRankRows * dstStride3;
            DstViewT dstView(dstGlobalData.data() + dstOffset, perRankShape, dstViewStride);
            TSTORE(dstView, stagingTileData);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        return;
    }

    // ---- 2D sliding chunked path ----
    //
    // For each rank r, iterate outer dims and chunk rows/cols.
    // Source: parallelGroup[r] at chunk offset
    // Destination: dstGlobalData at (rank base + chunk offset)

    PTO_ASSERT(tileValidRow > 0, "TGATHER: tile ValidRow must be greater than 0 for chunked transfer");
    PTO_ASSERT(tileValidCol > 0, "TGATHER: tile ValidCol must be greater than 0 for chunked transfer");

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0,
                   "TGATHER chunked: per-rank DIM_3 must be divisible by tile ValidRow when static. "
                   "Use a Tile with DYNAMIC ValidRow for partial row chunk support.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0,
                   "TGATHER chunked: DIM_4 must be divisible by tile ValidCol when static. "
                   "Use a Tile with DYNAMIC ValidCol for partial column chunk support.");
    }

    // Source strides (from first rank, all assumed same)
    const int srcStride0 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_0);
    const int srcStride1 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_1);
    const int srcStride2 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_2);
    const int srcStride3 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_3);
    const int srcStride4 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_4);

    // Destination strides
    const int dstStride0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int dstStride1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int dstStride2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int dstStride4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride srcChunkStride(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    DynStride dstChunkStride(dstStride0, dstStride1, dstStride2, dstStride3, dstStride4);

    for (int r = 0; r < nranks; ++r) {
        int64_t rankDstBase = static_cast<int64_t>(r) * perRankRows * dstStride3;

        for (int i0 = 0; i0 < gShape0; ++i0) {
            for (int i1 = 0; i1 < gShape1; ++i1) {
                for (int i2 = 0; i2 < gShape2; ++i2) {
                    int64_t srcBase = static_cast<int64_t>(i0) * srcStride0 + static_cast<int64_t>(i1) * srcStride1 +
                                      static_cast<int64_t>(i2) * srcStride2;
                    int64_t dstBase = rankDstBase + static_cast<int64_t>(i0) * dstStride0 +
                                      static_cast<int64_t>(i1) * dstStride1 + static_cast<int64_t>(i2) * dstStride2;

                    for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                        int currentRows = (rowOff + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowOff);

                        if constexpr (isDynamicRow) {
                            stagingTileData.RowMaskInternal = currentRows;
                        }

                        for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                            int currentCols = (colOff + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colOff);

                            if constexpr (isDynamicCol) {
                                stagingTileData.ColMaskInternal = currentCols;
                            }

                            int64_t srcOffset = srcBase + static_cast<int64_t>(rowOff) * srcStride3 +
                                                static_cast<int64_t>(colOff) * srcStride4;
                            int64_t dstOffset = dstBase + static_cast<int64_t>(rowOff) * dstStride3 +
                                                static_cast<int64_t>(colOff) * dstStride4;

                            DynShape chunkShape(1, 1, 1, currentRows, currentCols);

                            // TLOAD from rank r's source at chunk position
                            SrcViewT srcView(parallelGroup[r].data() + srcOffset, chunkShape, srcChunkStride);
                            TLOAD(stagingTileData, srcView);
                            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

                            // TSTORE to local destination at rank + chunk position
                            DstViewT dstView(dstGlobalData.data() + dstOffset, chunkShape, dstChunkStride);
                            TSTORE(dstView, stagingTileData);

                            // Sync before next chunk's TLOAD
                            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// TGATHER_IMPL (ping-pong): Gather with double buffering
//
// Uses two staging tiles (pingTile, pongTile) to overlap TLOAD of the next
// chunk (MTE2) with TSTORE of the current chunk (MTE3).
//
// Timeline without ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0] -> [TLOAD chunk1] -> [TSTORE chunk1] -> ...
//
// Timeline with ping-pong:
//   [TLOAD chunk0] -> [TSTORE chunk0 | TLOAD chunk1] -> [TSTORE chunk1 | TLOAD chunk2] -> ...
//
// Constraints: same as TGATHER_IMPL for chunked mode.
// ============================================================================

template <typename ParallelGroupType, typename GlobalDstData, typename TileData>
PTO_INTERNAL void TGATHER_IMPL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData, TileData &pingTile,
                               TileData &pongTile)
{
    using GlobalSrcData = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrcData::RawDType;

    static_assert(std::is_same_v<T, typename GlobalDstData::RawDType>, "TGATHER: GlobalData type mismatch!");
    static_assert(std::is_same_v<T, typename TileData::DType>,
                  "TGATHER: TileData element type must match GlobalData element type");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGATHER: src/dst layout mismatch");

    const int nranks = parallelGroup.GetSize();
    const int rootIdx = parallelGroup.GetRootIdx();

    PTO_ASSERT(nranks > 0, "ParallelGroup size must be greater than 0!");
    PTO_ASSERT(rootIdx >= 0 && rootIdx < nranks, "rootIdx must be in range [0, nranks)!");

    // Get per-rank dimensions
    const int gShape0 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_0);
    const int gShape1 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_1);
    const int gShape2 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_2);
    const int gShape3 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_3);
    const int gShape4 = parallelGroup[0].GetShape(GlobalTensorDim::DIM_4);

    const int perRankRows = gShape3;
    const int64_t totalRows = static_cast<int64_t>(gShape0) * gShape1 * gShape2 * gShape3;
    const int tileValidRow = pingTile.GetValidRow();
    const int tileValidCol = pingTile.GetValidCol();

    PTO_ASSERT(tileValidRow > 0, "TGATHER: tileValidRow must be greater than 0");
    PTO_ASSERT(tileValidCol > 0, "TGATHER: tileValidCol must be greater than 0");

    if (totalRows == 0 || gShape4 == 0) {
        return;
    }

    // ---- Simple path: per-rank data fits in UB tile, no ping-pong benefit ----
    if (totalRows <= tileValidRow && gShape4 <= tileValidCol) {
        if (nranks == 1) {
            TLOAD(pingTile, parallelGroup[0]);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobalData, pingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            return;
        }

        const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);

        using DynShape5D = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
        using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
        using DstViewT = GlobalTensor<T, DynShape5D, DynStride, GlobalDstData::layout>;

        DynShape5D perRankShape(gShape0, gShape1, gShape2, gShape3, gShape4);
        DynStride dstViewStride(dstGlobalData.GetStride(GlobalTensorDim::DIM_0),
                                dstGlobalData.GetStride(GlobalTensorDim::DIM_1),
                                dstGlobalData.GetStride(GlobalTensorDim::DIM_2), dstStride3,
                                dstGlobalData.GetStride(GlobalTensorDim::DIM_4));

        for (int r = 0; r < nranks; ++r) {
            TLOAD(pingTile, parallelGroup[r]);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

            int64_t dstOffset = static_cast<int64_t>(r) * perRankRows * dstStride3;
            DstViewT dstView(dstGlobalData.data() + dstOffset, perRankShape, dstViewStride);
            TSTORE(dstView, pingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        return;
    }

    // ---- 2D sliding chunked path with ping-pong double buffering ----

    constexpr bool isDynamicRow = (TileData::ValidRow == DYNAMIC);
    constexpr bool isDynamicCol = (TileData::ValidCol == DYNAMIC);

    if constexpr (!isDynamicRow) {
        PTO_ASSERT(gShape3 % tileValidRow == 0,
                   "TGATHER chunked: per-rank DIM_3 must be divisible by tile ValidRow when static.");
    }
    if constexpr (!isDynamicCol) {
        PTO_ASSERT(gShape4 % tileValidCol == 0,
                   "TGATHER chunked: DIM_4 must be divisible by tile ValidCol when static.");
    }

    const int srcStride0 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_0);
    const int srcStride1 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_1);
    const int srcStride2 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_2);
    const int srcStride3 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_3);
    const int srcStride4 = parallelGroup[0].GetStride(GlobalTensorDim::DIM_4);

    const int dstStride0 = dstGlobalData.GetStride(GlobalTensorDim::DIM_0);
    const int dstStride1 = dstGlobalData.GetStride(GlobalTensorDim::DIM_1);
    const int dstStride2 = dstGlobalData.GetStride(GlobalTensorDim::DIM_2);
    const int dstStride3 = dstGlobalData.GetStride(GlobalTensorDim::DIM_3);
    const int dstStride4 = dstGlobalData.GetStride(GlobalTensorDim::DIM_4);

    using DynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using DynStride = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcViewT = GlobalTensor<T, DynShape, DynStride, GlobalSrcData::layout>;
    using DstViewT = GlobalTensor<T, DynShape, DynStride, GlobalDstData::layout>;
    DynStride srcChunkStride(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4);
    DynStride dstChunkStride(dstStride0, dstStride1, dstStride2, dstStride3, dstStride4);

    // Ping-pong state
    bool usePing = true;
    bool hasPending = false;
    int64_t pendingDstOffset = 0;
    int pendingRows = 0;
    int pendingCols = 0;

    for (int r = 0; r < nranks; ++r) {
        int64_t rankDstBase = static_cast<int64_t>(r) * perRankRows * dstStride3;

        for (int i0 = 0; i0 < gShape0; ++i0) {
            for (int i1 = 0; i1 < gShape1; ++i1) {
                for (int i2 = 0; i2 < gShape2; ++i2) {
                    int64_t srcBase = static_cast<int64_t>(i0) * srcStride0 + static_cast<int64_t>(i1) * srcStride1 +
                                      static_cast<int64_t>(i2) * srcStride2;
                    int64_t dstBase = rankDstBase + static_cast<int64_t>(i0) * dstStride0 +
                                      static_cast<int64_t>(i1) * dstStride1 + static_cast<int64_t>(i2) * dstStride2;

                    for (int rowOff = 0; rowOff < gShape3; rowOff += tileValidRow) {
                        int currentRows = (rowOff + tileValidRow <= gShape3) ? tileValidRow : (gShape3 - rowOff);

                        for (int colOff = 0; colOff < gShape4; colOff += tileValidCol) {
                            int currentCols = (colOff + tileValidCol <= gShape4) ? tileValidCol : (gShape4 - colOff);

                            int64_t srcOffset = srcBase + static_cast<int64_t>(rowOff) * srcStride3 +
                                                static_cast<int64_t>(colOff) * srcStride4;
                            int64_t dstOffset = dstBase + static_cast<int64_t>(rowOff) * dstStride3 +
                                                static_cast<int64_t>(colOff) * dstStride4;

                            // Select load tile
                            TileData &loadTile = usePing ? pingTile : pongTile;
                            event_t curEvent = usePing ? EVENT_ID0 : EVENT_ID1;

                            if constexpr (isDynamicRow)
                                loadTile.RowMaskInternal = currentRows;
                            if constexpr (isDynamicCol)
                                loadTile.ColMaskInternal = currentCols;

                            DynShape chunkShape(1, 1, 1, currentRows, currentCols);
                            SrcViewT srcView(parallelGroup[r].data() + srcOffset, chunkShape, srcChunkStride);

                            if (hasPending) {
                                TileData &storeTile = usePing ? pongTile : pingTile;
                                event_t prevEvent = usePing ? EVENT_ID1 : EVENT_ID0;

                                // Wait for previous TLOAD to finish
                                wait_flag(PIPE_MTE2, PIPE_MTE3, prevEvent);

                                DynShape pendShape(1, 1, 1, pendingRows, pendingCols);
                                DstViewT dstView(dstGlobalData.data() + pendingDstOffset, pendShape, dstChunkStride);

                                // Issue TSTORE + TLOAD concurrently (MTE3 and MTE2 in parallel)
                                TSTORE(dstView, storeTile);
                                TLOAD(loadTile, srcView);

                                set_flag(PIPE_MTE3, PIPE_MTE2, prevEvent); // store done
                                set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);  // load done

                                // Ensure storeTile UB is safe before overwrite
                                wait_flag(PIPE_MTE3, PIPE_MTE2, prevEvent);
                            } else {
                                // First chunk: just issue TLOAD
                                TLOAD(loadTile, srcView);
                                set_flag(PIPE_MTE2, PIPE_MTE3, curEvent);
                            }

                            pendingDstOffset = dstOffset;
                            pendingRows = currentRows;
                            pendingCols = currentCols;
                            hasPending = true;
                            usePing = !usePing;
                        }
                    }
                }
            }
        }
    }

    // Epilogue: drain the last pending chunk
    if (hasPending) {
        TileData &lastTile = usePing ? pongTile : pingTile;
        event_t lastEvent = usePing ? EVENT_ID1 : EVENT_ID0;

        wait_flag(PIPE_MTE2, PIPE_MTE3, lastEvent);

        DynShape lastShape(1, 1, 1, pendingRows, pendingCols);
        DstViewT dstView(dstGlobalData.data() + pendingDstOffset, lastShape, dstChunkStride);
        TSTORE(dstView, lastTile);

        set_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
        wait_flag(PIPE_MTE3, PIPE_MTE2, lastEvent);
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGATHER_HPP
