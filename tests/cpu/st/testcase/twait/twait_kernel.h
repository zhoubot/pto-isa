/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

// TWAIT CPU Tests - Thread-based synchronization tests
// Unlike NPU tests which use multi-process/Shmem, CPU tests use std::thread

// Test 1: Basic single signal wait (EQ comparison)
bool RunTWaitBasic();

// Test 2: All comparison operators (EQ, NE, GT, GE, LT, LE)
bool RunTWaitCompare();

// Test 3: Multi-threaded atomic counter
bool RunTWaitAtomic(int numThreads);

// Test 4: 2D Signal Matrix
template <int Rows, int Cols>
bool RunTWaitMatrix();

// Test 5: Multi-phase waiting
bool RunTWaitMultiPhase();

// Test 6: 2D Sub-region with stride
template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegion();
