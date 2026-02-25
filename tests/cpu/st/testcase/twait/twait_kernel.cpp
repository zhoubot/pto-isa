/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "twait_kernel.h"
#include <pto/pto-inst.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <cassert>
#include <functional>

using namespace pto;

// ============================================================================
// Test 1: Basic single signal wait (EQ comparison)
// ============================================================================
bool RunTWaitBasic()
{
    alignas(64) std::atomic<int32_t> signal{0};
    comm::Signal sig(reinterpret_cast<int32_t *>(&signal));

    bool success = true;

    // Launch a thread to set the signal after a delay
    std::thread notifier([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        signal.store(42, std::memory_order_release);
    });

    // Wait for signal == 42
    cpu::TWAIT(sig, 42, comm::WaitCmp::EQ);

    // Verify
    if (signal.load() != 42) {
        success = false;
    }

    notifier.join();
    return success;
}

// ============================================================================
// Test 2: All comparison operators
// ============================================================================
bool RunTWaitCompare()
{
    // Returns false if the post-wait condition is violated.
    auto runCase = [](int32_t initial, int32_t stored, int32_t threshold, comm::WaitCmp cmp,
                      std::function<bool(int32_t)> postCheck) -> bool {
        alignas(64) std::atomic<int32_t> signal{initial};
        comm::Signal sig(reinterpret_cast<int32_t *>(&signal));

        std::thread notifier([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            signal.store(stored, std::memory_order_release);
        });

        cpu::TWAIT(sig, threshold, cmp);
        bool ok = postCheck(signal.load());
        notifier.join();
        return ok;
    };

    return runCase(0, 150, 100, comm::WaitCmp::GE, [](int32_t v) { return v >= 100; })   // GE
           && runCase(0, 1, 0, comm::WaitCmp::NE, [](int32_t v) { return v != 0; })      // NE
           && runCase(0, 51, 50, comm::WaitCmp::GT, [](int32_t v) { return v > 50; })    // GT
           && runCase(100, 10, 20, comm::WaitCmp::LE, [](int32_t v) { return v <= 20; }) // LE
           && runCase(100, 5, 10, comm::WaitCmp::LT, [](int32_t v) { return v < 10; });  // LT
}

// ============================================================================
// Test 3: Multi-threaded atomic counter
// ============================================================================
bool RunTWaitAtomic(int numThreads)
{
    alignas(64) std::atomic<int32_t> counter{0};
    comm::Signal sig(reinterpret_cast<int32_t *>(&counter));

    constexpr int kIncrementsPerThread = 25;
    const int kExpectedTotal = numThreads * kIncrementsPerThread;

    std::vector<std::thread> workers;

    // Launch worker threads that atomically increment the counter
    for (int i = 0; i < numThreads; ++i) {
        workers.emplace_back([&]() {
            for (int j = 0; j < kIncrementsPerThread; ++j) {
                counter.fetch_add(1, std::memory_order_release);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    // Wait for counter >= expected total
    cpu::TWAIT(sig, kExpectedTotal, comm::WaitCmp::GE);

    bool success = (counter.load() >= kExpectedTotal);

    for (auto &worker : workers) {
        worker.join();
    }

    return success;
}

// ============================================================================
// Test 4: 2D Signal Matrix
// ============================================================================
template <int Rows, int Cols>
bool RunTWaitMatrix()
{
    constexpr int kTotal = Rows * Cols;

    alignas(64) std::vector<std::atomic<int32_t>> matrix(kTotal);
    for (auto &elem : matrix) {
        elem.store(0, std::memory_order_release);
    }

    comm::Signal2D<Rows, Cols> sig(reinterpret_cast<int32_t *>(matrix.data()));

    // Launch a thread to set all matrix elements to 1
    std::thread notifier([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Cols; ++c) {
                matrix[r * Cols + c].store(1, std::memory_order_release);
            }
        }
    });

    // Wait for all matrix elements == 1
    cpu::TWAIT(sig, 1, comm::WaitCmp::EQ);

    // Verify all elements are 1
    bool success = true;
    for (int i = 0; i < kTotal; ++i) {
        if (matrix[i].load() != 1) {
            success = false;
            break;
        }
    }

    notifier.join();
    return success;
}

// ============================================================================
// Test 5: Multi-phase waiting
// ============================================================================
bool RunTWaitMultiPhase()
{
    alignas(64) std::atomic<int32_t> signal{0};
    comm::Signal sig(reinterpret_cast<int32_t *>(&signal));

    std::thread notifier([&]() {
        // Phase 1
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        signal.store(1, std::memory_order_release);

        // Phase 2
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        signal.store(3, std::memory_order_release);

        // Phase 3
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        signal.store(5, std::memory_order_release);
    });

    // Phase 1: Wait for signal == 1
    cpu::TWAIT(sig, 1, comm::WaitCmp::EQ);
    if (signal.load() != 1) {
        notifier.join();
        return false;
    }

    // Phase 2: Wait for signal >= 3
    cpu::TWAIT(sig, 3, comm::WaitCmp::GE);
    if (signal.load() < 3) {
        notifier.join();
        return false;
    }

    // Phase 3: Wait for signal == 5
    cpu::TWAIT(sig, 5, comm::WaitCmp::EQ);
    if (signal.load() != 5) {
        notifier.join();
        return false;
    }

    notifier.join();
    return true;
}

// ============================================================================
// Test 6: 2D Sub-region with stride
// ============================================================================
template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegion()
{
    constexpr int kFullRows = 8;
    constexpr int kStartRow = 2;
    constexpr int kStartCol = 5;

    alignas(64) std::vector<std::atomic<int32_t>> matrix(kFullRows * FullCols);
    for (auto &elem : matrix) {
        elem.store(0, std::memory_order_release);
    }

    // Create signal pointing to the sub-region with proper stride
    int32_t *subPtr = reinterpret_cast<int32_t *>(matrix.data()) + kStartRow * FullCols + kStartCol;
    comm::Signal2D<SubRows, SubCols> sig(subPtr, FullCols);

    // Launch a thread to set only the sub-region elements to 1
    std::thread notifier([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        for (int r = 0; r < SubRows; ++r) {
            for (int c = 0; c < SubCols; ++c) {
                int idx = (kStartRow + r) * FullCols + (kStartCol + c);
                matrix[idx].store(1, std::memory_order_release);
            }
        }
    });

    // Wait for all sub-region elements == 1
    cpu::TWAIT(sig, 1, comm::WaitCmp::EQ);

    // Verify sub-region elements are 1
    bool success = true;
    for (int r = 0; r < SubRows; ++r) {
        for (int c = 0; c < SubCols; ++c) {
            int idx = (kStartRow + r) * FullCols + (kStartCol + c);
            if (matrix[idx].load() != 1) {
                success = false;
                break;
            }
        }
        if (!success)
            break;
    }

    notifier.join();
    return success;
}

// Template instantiations
template bool RunTWaitMatrix<4, 8>();
template bool RunTWaitMatrix<7, 13>();
template bool RunTWaitSubRegion<16, 4, 8>();
