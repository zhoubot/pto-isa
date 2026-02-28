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

// ============================================================================
// HcclDeviceContext
//
// Binary layout must match the struct returned by HcclAllocComResourceByTiling().
// Tests only read rankId / winSize / windowsIn[]; the remaining fields exist
// solely to preserve the correct ABI-compatible memory layout.
// ============================================================================

static constexpr uint32_t HCCL_MAX_RANK_NUM = 32;

struct HcclDeviceContext {
    // mc2WorkSpace
    uint64_t workSpace;
    uint64_t workSpaceSize;

    // Rank metadata (used by tests)
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];

    // ---------- ABI tail (not accessed by tests) ----------

    char hcomId[128];

    struct {
        int32_t streamIds;
        uint32_t sqIds, cqIds, logicCqids;
    } streamInfo[HCCL_MAX_RANK_NUM];

    struct {
        struct {
            uint64_t resId, addr;
            uint32_t devId, tsId, rankId, flag;
        } noIpcNotifys[HCCL_MAX_RANK_NUM * 2], ipcNotifys[HCCL_MAX_RANK_NUM * 4], noIpcEvents[HCCL_MAX_RANK_NUM],
            aicpuNotify, aicpuOpNotify[2];
    } signalInfo;

    struct {
        uint8_t deterministic;
        uint8_t retryEnable;
        uint8_t highPerfEnable;
        uint8_t _pad0[5];
        uint8_t linkTimeOut[8];
        uint64_t notifyWaitTime;
        uint32_t retryHoldTime;
        uint32_t retryIntervalTime;
        bool interXLinkDisable;
        // rtFloatOverflowMode_t (enum class, underlying = int32_t)
        int32_t floatOverflowMode;
        uint32_t multiQpThreshold;
    } config;

    uint64_t overFlowAddr;
    uint8_t onlyRead;

    struct {
        uint64_t hostAddr, deviceAddr, readCacheAddr;
        uint32_t devMemSize, buffLen, flag;
    } kfcControlTransferH2DParams, kfcStatusTransferD2HParams;

    uint8_t _pad1[16];
    uint64_t winExpSize;
    uint64_t windowsExp[HCCL_MAX_RANK_NUM];

    uint8_t multiServerFlag;
    uint64_t ibverbsData;
    uint64_t ibverbsDataSize;
};
