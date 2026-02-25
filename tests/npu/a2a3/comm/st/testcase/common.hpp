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
#include <cstdlib>
#include <iostream>
#include <cstring>

// Shmem API headers for symmetric heap memory initialization
#include "shmem.h"

#if defined(CANN_SHMEM)
using ShmemUniqueId = aclshmemx_uniqueid_t;
#else
struct ShmemUniqueId {
    uint8_t _unused[1];
};
#endif

// ============================================================================
// ShmemEnv: Environment configuration for shmem initialization
// ============================================================================
struct ShmemEnv {
    int rank{0};
    int size{1};
    const char *ipPort{nullptr};
    uint64_t heapBytes{512ULL * 1024 * 1024}; // Default 512MB symmetric heap
    char ipPortBuf_[256]{};                   // G.STD.18: Internal buffer for copied getenv() result
};

// ============================================================================
// LoadEnv: Load shmem environment from environment variables
// ============================================================================
inline bool LoadEnv(ShmemEnv &env)
{
    const char *rankEnv = std::getenv("SHMEM_RANK");
    const char *sizeEnv = std::getenv("SHMEM_SIZE");
    const char *ipEnv = std::getenv("SHMEM_IP_PORT");
    const char *heapEnv = std::getenv("SHMEM_HEAP_BYTES");

    if (rankEnv == nullptr || sizeEnv == nullptr || ipEnv == nullptr) {
        std::cerr << "[ERROR] SHMEM_RANK/SHMEM_SIZE/SHMEM_IP_PORT must be set.\n";
        return false;
    }
    env.rank = static_cast<int>(std::strtol(rankEnv, nullptr, 10));
    env.size = static_cast<int>(std::strtol(sizeEnv, nullptr, 10));
    // G.STD.18: Copy getenv() result to internal buffer to avoid dangling pointer from non-reentrant function
    size_t ipLen = std::strlen(ipEnv);
    if (ipLen >= sizeof(env.ipPortBuf_)) {
        std::cerr << "[ERROR] SHMEM_IP_PORT string too long.\n";
        return false;
    }
    std::memcpy(env.ipPortBuf_, ipEnv, ipLen + 1);
    env.ipPort = env.ipPortBuf_;
    if (heapEnv != nullptr) {
        env.heapBytes = std::strtoull(heapEnv, nullptr, 10);
    }
    return true;
}

// ============================================================================
// FillShmemAttr: Fill common fields of aclshmemx_init_attr_t from ShmemEnv.
// ============================================================================
inline void FillShmemAttr(aclshmemx_init_attr_t &attributes, const ShmemEnv &env)
{
    attributes.my_pe = env.rank;
    attributes.n_pes = env.size;
    attributes.local_mem_size = env.heapBytes;

    // Copy IP:port string
    size_t ipLen = 0;
    if (env.ipPort != nullptr) {
        for (; ipLen < ACLSHMEM_MAX_IP_PORT_LEN - 1 && env.ipPort[ipLen] != '\0'; ++ipLen) {
            attributes.ip_port[ipLen] = env.ipPort[ipLen];
        }
    }
    attributes.ip_port[ipLen] = '\0';

    // Set option attributes
    constexpr int attrVersion = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    constexpr int DEFAULT_TIMEOUT = 120; // seconds
    attributes.option_attr = {attrVersion, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, -1};
}

// ============================================================================
// ShmemInit: Initialize shmem symmetric heap with given options
// Adapted from ShmemBackend::Init in shmem_backend.hpp
// ============================================================================
inline int ShmemInit(const ShmemEnv &env)
{
    aclshmemx_init_attr_t attributes;
    FillShmemAttr(attributes, env);

    // Use default unique ID
    aclshmemx_uniqueid_t defaultUid = ACLSHMEM_UNIQUEID_INITIALIZER;
    attributes.comm_args = reinterpret_cast<void *>(&defaultUid);

    int initRet = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (initRet != 0) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed with code: " << initRet << std::endl;
    }
    return initRet;
}

// ============================================================================
// ShmemInitFromEnv: Initialize shmem from ShmemEnv structure
// ============================================================================
inline bool ShmemInitFromEnv(ShmemEnv &env)
{
    const int ret = ShmemInit(env);
    return (ret == 0);
}

// ============================================================================
// ShmemInitFromEnvWithUniqueId: Initialize shmem with a provided unique id
// ============================================================================
inline bool ShmemInitFromEnvWithUniqueId(ShmemEnv &env, const ShmemUniqueId *uid)
{
#if defined(CANN_SHMEM)
    if (uid == nullptr) {
        return ShmemInitFromEnv(env);
    }

    aclshmemx_init_attr_t attributes;
    FillShmemAttr(attributes, env);

    attributes.comm_args = reinterpret_cast<void *>(const_cast<ShmemUniqueId *>(uid));
    int initRet = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (initRet != 0) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed with code: " << initRet << std::endl;
    }
    return (initRet == 0);
#else
    (void)uid;
    return ShmemInitFromEnv(env);
#endif
}

// ============================================================================
// ShmemFinalize: Finalize shmem
// ============================================================================
inline void ShmemFinalize()
{
    shmem_finalize();
}

// ============================================================================
// ShmemMalloc: Allocate symmetric heap memory
// ============================================================================
inline void *ShmemMalloc(size_t bytes)
{
    return shmem_malloc(bytes);
}

// ============================================================================
// ShmemFree: Free symmetric heap memory
// ============================================================================
inline void ShmemFree(void *ptr)
{
    shmem_free(ptr);
}

// ============================================================================
// ShmemBarrierAll: Global barrier synchronization
// ============================================================================
inline void ShmemBarrierAll()
{
    aclshmem_barrier_all();
}

// ============================================================================
// ShmemQuiet: Ensure all pending operations complete
// Note: aclshmem_quiet() is device-only in CANN_SHMEM, so we use barrier instead
// ============================================================================
inline void ShmemQuiet()
{
    // aclshmem_quiet() is device-only, use barrier for host synchronization
    aclshmem_barrier_all();
}

// ============================================================================
// ShmemMyPe: Get current rank ID
// ============================================================================
inline int ShmemMyPe()
{
    return shmem_my_pe();
}

// ============================================================================
// ShmemNPes: Get total number of ranks
// ============================================================================
inline int ShmemNPes()
{
    return shmem_n_pes();
}

// ============================================================================
// ShmemSetConfStoreTls: Configure TLS for shmem operations
// Note: CANN_SHMEM API changed - now takes (bool, const char*, uint32_t)
// ============================================================================
inline int ShmemSetConfStoreTls(bool enable, const char *tlsInfo, uint32_t tlsInfoLen)
{
    return shmem_set_conf_store_tls(enable, tlsInfo, tlsInfoLen);
}

// ============================================================================
// ShmemPtr: Get remote PE's address mapping for symmetric memory (Device)
//
// Converts a local symmetric heap address to the actual remote address that
// can be used to access memory on the specified PE. This is essential for
// remote memory operations (PUT/GET).
//
// Parameters:
//   - localPtr: Local symmetric heap address
//   - pe: Target PE (rank) number
//
// Returns: Mapped address for accessing memory on remote PE
// ============================================================================
template <typename T>
AICORE inline __gm__ T *ShmemPtr(__gm__ T *localPtr, int pe)
{
    return (__gm__ T *)aclshmem_ptr(localPtr, pe);
}

// ============================================================================
// ShmemDeviceBarrierAll: Global barrier synchronization (Device)
//
// Synchronizes all PEs in the device kernel. All PEs must call this function,
// and no PE will proceed until all have arrived.
// ============================================================================
AICORE inline void ShmemDeviceBarrierAll()
{
    aclshmem_barrier_all();
}

// ============================================================================
// ShmemDeviceQuiet: Ensure all pending remote operations complete (Device)
//
// Ensures that all previously issued remote memory operations (PUT, GET, etc.)
// from this PE have completed before proceeding.
// ============================================================================
AICORE inline void ShmemDeviceQuiet()
{
    aclshmem_quiet();
}

// ============================================================================
// TestContext: RAII-style ACL + Shmem initialization / teardown helper.
// Call Init() to set up ACL runtime and shmem, then Finalize() to tear down.
// ============================================================================
struct TestContext {
    int32_t deviceId{-1};
    aclrtStream stream{nullptr};
    int aclStatus{0};

    /// Initialize ACL runtime + shmem symmetric heap.
    /// @param uid  Optional ShmemUniqueId pointer (nullptr = use default).
    bool Init(int rankId, int nRanks, int nDevices, int firstDeviceId, const char *ipPort,
              uint64_t heapBytes = 1024ULL * 1024 * 1024, const ShmemUniqueId *uid = nullptr)
    {
        if (nDevices <= 0 || nRanks <= 0) {
            std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
            return false;
        }
        int32_t ret = ShmemSetConfStoreTls(false, nullptr, 0);
        if (ret != 0) {
            std::cerr << "[ERROR] Failed to init shmem tls\n";
            return false;
        }
        if (nDevices < 1) {
            std::cerr << "[ERROR] n_devices must be >= 1\n";
            return false;
        }

        deviceId = rankId % nDevices + firstDeviceId;

        aclStatus |= aclInit(nullptr);
        aclStatus |= aclrtSetDevice(deviceId);
        aclStatus |= aclrtCreateStream(&stream);

        ShmemEnv env;
        env.rank = rankId;
        env.size = nRanks;
        env.ipPort = ipPort;
        env.heapBytes = heapBytes;

        bool shmemOk = (uid != nullptr) ? ShmemInitFromEnvWithUniqueId(env, uid) : ShmemInitFromEnv(env);
        if (!shmemOk) {
            std::cerr << "[ERROR] ShmemInit failed!" << std::endl;
            return false;
        }
        return true;
    }

    /// Teardown shmem + ACL runtime.  Returns true when all ACL calls succeeded.
    bool Finalize()
    {
        ShmemFinalize();
        aclStatus |= aclrtDestroyStream(stream);
        aclStatus |= aclrtResetDevice(deviceId);
        aclStatus |= aclFinalize();
        return (aclStatus == 0);
    }
};

// ============================================================================
// ForkAndRun: Fork one child process per rank, run perRankFn, collect results.
//
// perRankFn signature: bool(int rankId)
// ============================================================================
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

template <typename Func>
inline bool ForkAndRun(int nRanks, int firstRankId, Func &&perRankFn)
{
    std::vector<pid_t> pids;
    for (int r = 0; r < nRanks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = perRankFn(firstRankId + r);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "[ERROR] fork() failed for rank " << r << std::endl;
            return false;
        }
    }
    bool success = true;
    for (pid_t p : pids) {
        int status = 0;
        waitpid(p, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            success = false;
        }
    }
    return success;
}

// ============================================================================
// ForkAndRunWithUniqueId: Same as ForkAndRun but generates a ShmemUniqueId
// before forking and passes it to each child.
//
// perRankFn signature: bool(int rankId, const ShmemUniqueId *uid)
// ============================================================================
template <typename Func>
inline bool ForkAndRunWithUniqueId(int nRanks, int firstRankId, Func &&perRankFn)
{
#if defined(CANN_SHMEM)
    ShmemUniqueId uid;
    ShmemUniqueId *uidPtr = nullptr;
    if (aclshmemx_get_uniqueid(&uid) == 0) {
        uidPtr = &uid;
    }
#else
    ShmemUniqueId *uidPtr = nullptr;
#endif

    std::vector<pid_t> pids;
    for (int r = 0; r < nRanks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            const bool ok = perRankFn(firstRankId + r, uidPtr);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "[ERROR] fork() failed for rank " << r << std::endl;
            return false;
        }
    }
    bool success = true;
    for (pid_t p : pids) {
        int status = 0;
        waitpid(p, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            success = false;
        }
    }
    return success;
}
