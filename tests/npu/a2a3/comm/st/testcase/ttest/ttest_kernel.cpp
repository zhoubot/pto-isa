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
// Kernel 1: TTEST True Condition Test
// Tests that TTEST returns true when signal matches expected value
// ============================================================================
__global__ AICORE void TTestTrueKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal value to 42 using ShmemPtr for remote address
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, 42, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test if local signal == 42 (should be true)
        pto::comm::Signal localSignal(shmem_signal);

        bool testResult = pto::comm::TTEST(localSignal, 42, pto::comm::WaitCmp::EQ);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 2: TTEST False Condition Test
// Tests that TTEST returns false when signal does not match expected value
// ============================================================================
__global__ AICORE void TTestFalseKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal value to 42 using ShmemPtr
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, 42, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test if local signal == 100 (should be false, signal is 42)
        pto::comm::Signal localSignal(shmem_signal);

        bool testResult = pto::comm::TTEST(localSignal, 100, pto::comm::WaitCmp::EQ);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 3: TTEST with different comparison operators
// Tests GE (>=), GT (>), LE (<=), LT (<), NE (!=) operators
// ============================================================================
template <pto::comm::WaitCmp cmp>
__global__ AICORE void TTestCompareKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result, int32_t signalValue,
                                          int32_t cmpValue)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal to specified value using ShmemPtr
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, signalValue, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test with specified comparison on local signal
        pto::comm::Signal localSignal(shmem_signal);

        bool testResult = pto::comm::TTEST(localSignal, cmpValue, cmp);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 4: TTEST Polling with Timeout
// Demonstrates polling pattern: check, do work, check again
// ============================================================================
__global__ AICORE void TTestPollingTimeoutKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *poll_count,
                                                 __gm__ int32_t *final_result, int32_t delay_iters, int32_t max_polls,
                                                 bool send_signal)
{
    int my_rank = shmem_my_pe();

    // Sync start for polling
    ShmemDeviceBarrierAll();

    if (my_rank == 0 && send_signal) {
        // Rank 0: delay before sending signal
        for (int32_t i = 0; i < delay_iters; ++i) {
            __asm__ __volatile__("");
        }
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, 999, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    if (my_rank == 1) {
        // Rank 1: Poll with TTEST until signal or timeout
        pto::comm::Signal localSignal(shmem_signal);

        int32_t count = 0;
        bool found = false;
        while (count < max_polls) {
            if (pto::comm::TTEST(localSignal, 999, pto::comm::WaitCmp::EQ)) {
                found = true;
                break;
            }
            ++count;
        }

        *poll_count = count;
        *final_result = found ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 5: TTEST Not Equal (NE) Test
// Tests that TTEST with NE returns true when values differ
// ============================================================================
__global__ AICORE void TTestNEKernel(__gm__ int32_t *shmem_signal, __gm__ int32_t *result)
{
    int my_rank = shmem_my_pe();

    if (my_rank == 0) {
        // Rank 0: Set rank 1's signal to 50 using ShmemPtr
        __gm__ int32_t *remote_signal = ShmemPtr(shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, 50, pto::comm::NotifyOp::Set);
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Rank 1: Test if local signal != 0 (should be true, signal is 50)
        pto::comm::Signal localSignal(shmem_signal);

        bool testResult = pto::comm::TTEST(localSignal, 0, pto::comm::WaitCmp::NE);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Kernel 6: TTEST Sub-Region Test
// Tests TTEST on a sub-region of a larger signal grid
// Rank 0 sets some signals in rank 1's 8x16 grid
// Rank 1 uses Signal2D<4, 8> with stride to test a sub-region
// ============================================================================
template <int FullCols, int SubRows, int SubCols>
__global__ AICORE void TTestSubRegionKernel(__gm__ int32_t *shmem_matrix, __gm__ int32_t *result)
{
    int my_rank = shmem_my_pe();

    constexpr int startRow = 2;
    constexpr int startCol = 4;

    if (my_rank == 0) {
        // Set all sub-region elements of rank 1's grid to 1
        __gm__ int32_t *remote_matrix = ShmemPtr(shmem_matrix, 1);
        for (int r = 0; r < SubRows; ++r) {
            for (int c = 0; c < SubCols; ++c) {
                __gm__ int32_t *elem = remote_matrix + (startRow + r) * FullCols + (startCol + c);
                pto::comm::Signal sig(elem);
                pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::Set);
            }
        }
        ShmemDeviceQuiet();
    }

    ShmemDeviceBarrierAll();

    if (my_rank == 1) {
        // Test sub-region: all elements should be 1
        __gm__ int32_t *subPtr = shmem_matrix + startRow * FullCols + startCol;
        pto::comm::Signal2D<SubRows, SubCols> subRegion(subPtr, FullCols);
        bool testResult = pto::comm::TTEST(subRegion, 1, pto::comm::WaitCmp::EQ);
        *result = testResult ? 1 : 0;
    }

    ShmemDeviceBarrierAll();
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunTTestTrueKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8790", 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestTrueKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, result);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (testResult != 1) {
            std::cerr << "TTest True test failed! TTEST(EQ, 42) should return true (1), Got: " << testResult
                      << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST(EQ, 42) returned " << testResult << " (expected 1/true)" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);

    return ctx.Finalize() && is_ok;
}

bool RunTTestFalseKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8791", 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    int32_t one = 1; // Initialize to 1 so we can detect if TTEST correctly returns 0
    aclrtMemcpy(result, sizeof(int32_t), &one, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestFalseKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, result);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (testResult != 0) {
            std::cerr << "TTest False test failed! TTEST(EQ, 100) when signal=42 should return false (0), Got: "
                      << testResult << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST(EQ, 100) when signal=42 returned " << testResult << " (expected 0/false)"
                      << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);

    return ctx.Finalize() && is_ok;
}

template <pto::comm::WaitCmp cmp>
bool RunTTestCompareKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int32_t signalValue,
                           int32_t cmpValue, bool expectedResult)
{
    char ipPort[64];
    snprintf(ipPort, sizeof(ipPort), "tcp://127.0.0.1:%d", 8792 + static_cast<int>(cmp));

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, ipPort, 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestCompareKernel<cmp><<<1, nullptr, ctx.stream>>>(shmem_signal, result, signalValue, cmpValue);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        int32_t expectedInt = expectedResult ? 1 : 0;
        if (testResult != expectedInt) {
            std::cerr << "TTest Compare test failed! signal=" << signalValue << ", cmpValue=" << cmpValue
                      << ", expected " << expectedInt << ", Got: " << testResult << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST compare (signal=" << signalValue << ", cmpValue=" << cmpValue << ") returned "
                      << testResult << " (expected " << expectedInt << ")" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);

    return ctx.Finalize() && is_ok;
}

bool RunTTestPollingTimeoutKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int32_t delay_iters,
                                  int32_t max_polls, bool expected_found, bool send_signal)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8800", 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *poll_count = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *final_result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || poll_count == nullptr || final_result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(poll_count, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(final_result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestPollingTimeoutKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, poll_count, final_result, delay_iters,
                                                          max_polls, send_signal);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t count = 0;
        int32_t found = 0;
        aclrtMemcpy(&count, sizeof(int32_t), poll_count, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&found, sizeof(int32_t), final_result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        const int32_t expected = expected_found ? 1 : 0;
        if (found != expected) {
            std::cerr << "TTest Polling Timeout test failed! expected=" << expected << ", found=" << found << std::endl;
            is_ok = false;
        } else if (count < 0 || count > max_polls) {
            std::cerr << "TTest Polling Timeout test failed! Poll count out of range: " << count << std::endl;
            is_ok = false;
        } else if (expected_found) {
            std::cout << "Rank 1: TTEST polling found signal after " << count << " iterations" << std::endl;
        } else {
            std::cout << "Rank 1: TTEST polling timed out after " << count << " iterations" << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(poll_count);
    ShmemFree(final_result);

    return ctx.Finalize() && is_ok;
}

bool RunTTestNEKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8801", 8ULL * 1024 * 1024))
        return false;

    int32_t *shmem_signal = (int32_t *)ShmemMalloc(sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_signal == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestNEKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, result);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (testResult != 1) {
            std::cerr << "TTest NE test failed! TTEST(NE, 0) when signal=50 should return true (1), Got: " << testResult
                      << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST(NE, 0) when signal=50 returned " << testResult << " (expected 1/true)"
                      << std::endl;
        }
    }

    ShmemFree(shmem_signal);
    ShmemFree(result);

    return ctx.Finalize() && is_ok;
}

template <int FullCols, int SubRows, int SubCols>
bool RunTTestSubRegionKernel(int rank_id, int n_ranks, int n_devices, int first_device_id)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, "tcp://127.0.0.1:8802", 8ULL * 1024 * 1024))
        return false;

    constexpr int FullRows = 8;
    int32_t *shmem_matrix = (int32_t *)ShmemMalloc(FullRows * FullCols * sizeof(int32_t));
    int32_t *result = (int32_t *)ShmemMalloc(sizeof(int32_t));

    if (shmem_matrix == nullptr || result == nullptr) {
        std::cerr << "[ERROR] ShmemMalloc failed!" << std::endl;
        return false;
    }

    // Zero-initialize matrix and result
    std::vector<int32_t> zeros(FullRows * FullCols, 0);
    aclrtMemcpy(shmem_matrix, FullRows * FullCols * sizeof(int32_t), zeros.data(),
                FullRows * FullCols * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    int32_t zero = 0;
    aclrtMemcpy(result, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ShmemBarrierAll();

    TTestSubRegionKernel<FullCols, SubRows, SubCols><<<1, nullptr, ctx.stream>>>(shmem_matrix, result);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    ShmemBarrierAll();

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t testResult = 0;
        aclrtMemcpy(&testResult, sizeof(int32_t), result, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

        if (testResult != 1) {
            std::cerr << "TTest SubRegion test failed! TTEST on sub-region should return true (1), Got: " << testResult
                      << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TTEST sub-region returned " << testResult << " (expected 1/true)" << std::endl;
        }
    }

    ShmemFree(shmem_matrix);
    ShmemFree(result);

    return ctx.Finalize() && is_ok;
}

// ============================================================================
// Multi-process Launcher Functions
// ============================================================================

bool RunTTestTrue(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id,
                      [&](int rankId) { return RunTTestTrueKernel(rankId, n_ranks, n_devices, first_device_id); });
}

bool RunTTestFalse(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id,
                      [&](int rankId) { return RunTTestFalseKernel(rankId, n_ranks, n_devices, first_device_id); });
}

template <pto::comm::WaitCmp cmp>
bool RunTTestCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue,
                     int32_t cmpValue, bool expectedResult)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTTestCompareKernel<cmp>(rankId, n_ranks, n_devices, first_device_id, signalValue, cmpValue,
                                          expectedResult);
    });
}

bool RunTTestPollingTimeout(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTTestPollingTimeoutKernel(rankId, n_ranks, n_devices, first_device_id, 50000, 200000, true, true);
    });
}

bool RunTTestPollingTimeoutMiss(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTTestPollingTimeoutKernel(rankId, n_ranks, n_devices, first_device_id, 0, 50000, false, false);
    });
}

bool RunTTestNE(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id,
                      [&](int rankId) { return RunTTestNEKernel(rankId, n_ranks, n_devices, first_device_id); });
}

// Non-template wrapper functions for host-side linkage (avoid including comm_types.hpp in main.cpp)
bool RunTTestCompare_GE(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue,
                        int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::GE>(n_ranks, n_devices, first_rank_id, first_device_id, signalValue,
                                                   cmpValue, expectedResult);
}

bool RunTTestCompare_GT(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue,
                        int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::GT>(n_ranks, n_devices, first_rank_id, first_device_id, signalValue,
                                                   cmpValue, expectedResult);
}

bool RunTTestCompare_LE(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue,
                        int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::LE>(n_ranks, n_devices, first_rank_id, first_device_id, signalValue,
                                                   cmpValue, expectedResult);
}

bool RunTTestCompare_LT(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue,
                        int32_t cmpValue, bool expectedResult)
{
    return RunTTestCompare<pto::comm::WaitCmp::LT>(n_ranks, n_devices, first_rank_id, first_device_id, signalValue,
                                                   cmpValue, expectedResult);
}

template <int FullCols, int SubRows, int SubCols>
bool RunTTestSubRegion(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRun(n_ranks, first_rank_id, [&](int rankId) {
        return RunTTestSubRegionKernel<FullCols, SubRows, SubCols>(rankId, n_ranks, n_devices, first_device_id);
    });
}

template bool RunTTestSubRegion<16, 4, 8>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
