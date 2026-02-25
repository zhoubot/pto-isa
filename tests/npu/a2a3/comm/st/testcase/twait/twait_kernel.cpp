/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#include <pto/pto-inst.hpp>

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// Kernel 1: TWAIT Basic Test
// Rank 0 sends signal to rank 1, rank 1 waits for the signal
// Tests blocking wait functionality
// ============================================================================
__global__ AICORE void TWaitBasicKernel(__gm__ int32_t *shmem_signal)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Send signal to rank 1
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        // Set signal value to 42
        pto::comm::TNOTIFY(targetSignal, 42, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();
    } else if (my_rank == 1) {
        ShmemDeviceBarrierAll();
        // Rank 1: Wait for signal to equal 42
        pto::comm::Signal localSignal(shmem_signal);

        // Blocking wait until signal == 42
        pto::comm::TWAIT(localSignal, 42, pto::comm::WaitCmp::EQ);
    }
    // Global synchronization
    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 2: TWAIT with different comparison operators
// Tests EQ, NE, GT, GE, LT, LE comparisons
// ============================================================================
__global__ AICORE void TWaitCompareKernel(__gm__ int32_t *shmem_signal, int32_t notifyValue)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Send signal with specified value to rank 1
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, notifyValue, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    } else if (my_rank == 1) {
        // Rank 1: Wait for signal >= 100
        pto::comm::Signal localSignal(shmem_signal);

        // Blocking wait until signal >= 100
        pto::comm::TWAIT(localSignal, 100, pto::comm::WaitCmp::GE);
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 3: TWAIT with multi-rank atomic add
// Multiple ranks atomically add to rank 0's counter, rank 0 waits for threshold
// ============================================================================
__global__ AICORE void TWaitAtomicKernel(__gm__ int32_t *shmem_counter, int threshold, int iters)
{
    int my_rank = shmem_my_pe();

    __gm__ int32_t *remote_counter = ShmemPtr(shmem_counter, 0);
    pto::comm::Signal counterSignal(remote_counter);

    ShmemDeviceBarrierAll();

    if (my_rank != 0) {
        // Non-rank-0: atomically add multiple times
        for (int i = 0; i < iters; ++i) {
            pto::comm::TNOTIFY(counterSignal, 1, pto::comm::NotifyOp::AtomicAdd);
        }
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();
    } else {
        // Rank 0: Wait until counter >= threshold
        pto::comm::Signal localCounter(shmem_counter);
        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localCounter, threshold, pto::comm::WaitCmp::GE);
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 4: TWAIT 2D Signal Matrix
// Rank 0 sets a 2D signal matrix for rank 1, rank 1 waits on the matrix
// ============================================================================
template <int Rows, int Cols>
__global__ AICORE void TWaitMatrixKernel(__gm__ int32_t *shmem_matrix)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        __gm__ int32_t *remote_matrix = ShmemPtr(shmem_matrix, 1);
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Cols; ++c) {
                __gm__ int32_t *remote_elem = remote_matrix + r * Cols + c;
                pto::comm::Signal targetElem(remote_elem);
                pto::comm::TNOTIFY(targetElem, 1, pto::comm::NotifyOp::Set);
            }
        }
    } else if (my_rank == 1) {
        pto::comm::Signal2D<Rows, Cols> localMatrix(shmem_matrix);
        pto::comm::TWAIT(localMatrix, 1, pto::comm::WaitCmp::EQ);
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 4b: TWAIT Sub-Region
// Rank 0 sets signals in a sub-region of rank 1's larger grid
// Rank 1 uses Signal2D with stride to wait on just that sub-region
// ============================================================================
template <int FullCols, int SubRows, int SubCols>
__global__ AICORE void TWaitSubRegionKernel(__gm__ int32_t *shmem_matrix)
{
    int my_rank = shmem_my_pe();

    // Sub-region starts at row=2, col=4 within the FullCols-wide grid
    constexpr int startRow = 2;
    constexpr int startCol = 4;

    if (my_rank == 0) {
        __gm__ int32_t *remote_matrix = ShmemPtr(shmem_matrix, 1);
        // Set only the sub-region elements
        for (int r = 0; r < SubRows; ++r) {
            for (int c = 0; c < SubCols; ++c) {
                __gm__ int32_t *elem = remote_matrix + (startRow + r) * FullCols + (startCol + c);
                pto::comm::Signal sig(elem);
                pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::Set);
            }
        }
    } else if (my_rank == 1) {
        // Wait on sub-region: ptr offset to (startRow, startCol), stride = FullCols
        __gm__ int32_t *subPtr = shmem_matrix + startRow * FullCols + startCol;
        pto::comm::Signal2D<SubRows, SubCols> subRegion(subPtr, FullCols);
        pto::comm::TWAIT(subRegion, 1, pto::comm::WaitCmp::EQ);
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 5: TWAIT Multi-Phase
// Rank 0 updates signal in phases, rank 1 waits in phases
// ============================================================================
__global__ AICORE void TWaitMultiPhaseKernel(__gm__ int32_t *shmem_signal)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, 1, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();

        pto::comm::TNOTIFY(targetSignal, 3, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();

        pto::comm::TNOTIFY(targetSignal, 5, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
        ShmemDeviceBarrierAll();
    } else if (my_rank == 1) {
        pto::comm::Signal localSignal(shmem_signal);

        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localSignal, 1, pto::comm::WaitCmp::EQ);

        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localSignal, 3, pto::comm::WaitCmp::GE);

        ShmemDeviceBarrierAll();
        pto::comm::TWAIT(localSignal, 5, pto::comm::WaitCmp::EQ);

        ShmemDeviceBarrierAll();
    }

    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunTWaitBasicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8780", 8ULL * 1024 * 1024))
        return false;

    // Allocate symmetric memory for signal
    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_signal == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    // Initialize signal to 0
    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    ShmemBarrierAll();

    // Execute kernel
    TWaitBasicKernel<<<1, nullptr, ctx.stream>>>(shmem_signal);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    // Rank 1 verifies the signal value
    if (rank_id == 1) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (result != 42) {
            std::cerr << "TWait Basic test failed! Expected: 42, Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TWait received signal = " << result << " (expected 42)" << std::endl;
        }
    }

    ShmemFree(shmem_signal);

    return ctx.Finalize() && is_ok;
}

bool RunTWaitCompareKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int32_t notifyValue)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8781", 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_signal == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TWaitCompareKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, notifyValue);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (result != notifyValue) {
            std::cerr << "TWait Compare test failed! Expected: " << notifyValue << ", Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TWait (GE) received signal = " << result << " (expected >= 100)" << std::endl;
        }
    }

    ShmemFree(shmem_signal);

    return ctx.Finalize() && is_ok;
}

bool RunTWaitAtomicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8782", 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_counter = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_counter == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_counter, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    constexpr int kAtomicIters = 50;
    const int threshold = (n_ranks - 1) * kAtomicIters;
    TWaitAtomicKernel<<<1, nullptr, ctx.stream>>>(shmem_counter, threshold, kAtomicIters);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 0) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_counter, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (result != threshold) {
            std::cerr << "TWait Atomic test failed! Expected: " << threshold << ", Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 0: TWait (GE) atomic counter = " << result << " (expected >= " << threshold << ")"
                      << std::endl;
        }
    }

    ShmemFree(shmem_counter);

    return ctx.Finalize() && is_ok;
}

template <int Rows, int Cols>
bool RunTWaitMatrixKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8783", 8ULL * 1024 * 1024))
        return false;

    constexpr size_t total = Rows * Cols;
    int32_t *shmem_matrix = (int32_t *)ShmemMalloc(total * sizeof(int32_t));
    if (shmem_matrix == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    std::vector<int32_t> zeros(total, 0);
    aclrtMemcpy(shmem_matrix, total * sizeof(int32_t), zeros.data(), total * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TWaitMatrixKernel<Rows, Cols><<<1, nullptr, ctx.stream>>>(shmem_matrix);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;
    if (rank_id == 1) {
        std::vector<int32_t> result(total, 0);
        aclrtMemcpy(result.data(), total * sizeof(int32_t), shmem_matrix, total * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < total; ++i) {
            if (result[i] != 1) {
                std::cerr << "TWait Matrix test failed at " << i << " got " << result[i] << std::endl;
                is_ok = false;
                break;
            }
        }
    }

    ShmemFree(shmem_matrix);

    return ctx.Finalize() && is_ok;
}

bool RunTWaitMultiPhaseKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8786", 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    if (shmem_signal == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TWaitMultiPhaseKernel<<<1, nullptr, ctx.stream>>>(shmem_signal);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;
    if (rank_id == 1) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (result != 5) {
            std::cerr << "TWait MultiPhase failed! Expected: 5, Got: " << result << std::endl;
            is_ok = false;
        }
    }

    ShmemFree(shmem_signal);

    return ctx.Finalize() && is_ok;
}

template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegionKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8787", 8ULL * 1024 * 1024))
        return false;

    constexpr size_t totalRows = 8;
    constexpr size_t total = totalRows * FullCols;
    int32_t *shmem_matrix = (int32_t *)ShmemMalloc(total * sizeof(int32_t));
    if (shmem_matrix == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    std::vector<int32_t> zeros(total, 0);
    aclrtMemcpy(shmem_matrix, total * sizeof(int32_t), zeros.data(), total * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TWaitSubRegionKernel<FullCols, SubRows, SubCols><<<1, nullptr, ctx.stream>>>(shmem_matrix);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;
    if (rank_id == 1) {
        std::vector<int32_t> result(total, 0);
        aclrtMemcpy(result.data(), total * sizeof(int32_t), shmem_matrix, total * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        constexpr int startRow = 2;
        constexpr int startCol = 4;
        for (int r = 0; r < SubRows; ++r) {
            for (int c = 0; c < SubCols; ++c) {
                int idx = (startRow + r) * FullCols + (startCol + c);
                if (result[idx] != 1) {
                    std::cerr << "TWait SubRegion test failed at (" << (startRow + r) << "," << (startCol + c)
                              << ") got " << result[idx] << std::endl;
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
                break;
        }
    }

    ShmemFree(shmem_matrix);

    return ctx.Finalize() && is_ok;
}

// ============================================================================
// Multi-process Launcher Functions
// ============================================================================

bool RunTWaitBasic(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id,
                      [&](int rankId) { return RunTWaitBasicKernel(rankId, n_ranks, n_devices, first_device_id); });
}

bool RunTWaitCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t notifyValue)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTWaitCompareKernel(rankId, n_ranks, n_devices, first_device_id, notifyValue);
    });
}

bool RunTWaitAtomic(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id,
                      [&](int rankId) { return RunTWaitAtomicKernel(rankId, n_ranks, n_devices, first_device_id); });
}

template <int Rows, int Cols>
bool RunTWaitMatrix(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTWaitMatrixKernel<Rows, Cols>(rankId, n_ranks, n_devices, first_device_id);
    });
}

bool RunTWaitMultiPhase(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTWaitMultiPhaseKernel(rankId, n_ranks, n_devices, first_device_id);
    });
}

template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegion(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTWaitSubRegionKernel<FullCols, SubRows, SubCols>(rankId, n_ranks, n_devices, first_device_id);
    });
}

template bool RunTWaitMatrix<4, 8>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunTWaitMatrix<7, 13>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunTWaitSubRegion<16, 4, 8>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
