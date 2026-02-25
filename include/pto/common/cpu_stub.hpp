/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPUSTUB_HPP
#define PTO_CPUSTUB_HPP

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <thread>
#include <vector>

// CPU simulator assertion helper (always enabled).
#define PTO_CPU_STUB_ASSERT(cond)                                                                                  \
    do {                                                                                                           \
        if (!(cond)) {                                                                                             \
            std::fprintf(stderr, "[PTO][CA] Constraint violated. Condition: %s. Hint: see docs/coding/debug.md\n", \
                         #cond);                                                                                   \
            std::abort();                                                                                          \
        }                                                                                                          \
    } while (0)

#define __global__
#define AICORE
#define __aicore__
#define __gm__
#define __out__
#define __in__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define __fbuf__
#define __tf__

typedef void *aclrtStream;
typedef int pipe_t;
const pipe_t PIPE_S = 0;
const pipe_t PIPE_V = 1;
const pipe_t PIPE_MTE1 = 2;
const pipe_t PIPE_MTE2 = 3;
const pipe_t PIPE_MTE3 = 4;
const pipe_t PIPE_M = 5;
const pipe_t PIPE_ALL = 6;
inline void pipe_barrier(pipe_t pipe)
{
    (void)pipe;
}

constexpr pipe_t opPipeList[] = {};

#define aclFloat16ToFloat(x) ((float)(x)
#define aclInit(x)
#define aclrtSetDevice(x)

#define aclrtCreateStream(x)

static inline void aclrtMallocHost(void **p, size_t sz)
{
    PTO_CPU_STUB_ASSERT(sz != 0);
    *p = malloc(sz);
}

#define aclrtMalloc(a, b, c) aclrtMallocHost(a, b)

#define aclrtMemcpy(dst, sz_dst, src, sz_src, type)                              \
    {                                                                            \
        for (size_t i = 0; i < sz_src && i < sz_dst; i++)                        \
            reinterpret_cast<char *>(dst)[i] = reinterpret_cast<char *>(src)[i]; \
    }

#define aclrtSynchronizeStream(x)
#define aclrtFree(x) free(x)
#define aclrtFreeHost(x) free(x)
#define aclrtDestroyStream(x)
#define aclrtResetDevice(x)
#define aclFinalize(x)
#define set_flag(a, b, c)
#define wait_flag(a, b, c)
#define __cce_get_tile_ptr(x) x

typedef int event_t;
#define EVENT_ID0 0

// -----------------------------------------------------------------------------
// CPU simulator stubs for control register helpers used by some tile ops.
// -----------------------------------------------------------------------------
namespace pto {
inline uint64_t get_ctrl() { return 0; }
inline void set_ctrl(uint64_t) {}
inline uint64_t sbitset1(uint64_t v, int bit) { return v | (1ULL << static_cast<uint64_t>(bit)); }
inline uint64_t sbitset0(uint64_t v, int bit) { return v & ~(1ULL << static_cast<uint64_t>(bit)); }
} // namespace pto

// -----------------------------------------------------------------------------
// CPU simulator launch context
// -----------------------------------------------------------------------------
// PTOAS-emitted kernels may reference these builtins via `using namespace pto;`.
// On NPU they are provided by the device toolchain; on CPU we emulate them.

namespace pto {
namespace cpu_sim {
// Thread-local so a host-side launcher can iterate blocks/subblocks deterministically.
inline thread_local int64_t g_block_idx = 0;
inline thread_local int64_t g_block_num = 1;
inline thread_local int64_t g_subblock_id = 0;
inline thread_local int64_t g_subblock_dim = 1;

inline void set_launch_context(int64_t block_idx, int64_t subblock_id, int64_t subblock_dim)
{
    g_block_idx = block_idx;
    g_subblock_id = subblock_id;
    g_subblock_dim = subblock_dim;
}

inline void set_grid_dim(int64_t block_num, int64_t subblock_dim)
{
    // Treat "block num" as the total number of execution lanes.
    // When subblocks are used, total lanes = blocks * subblocks.
    g_block_num = block_num * (subblock_dim > 0 ? subblock_dim : 1);
    g_subblock_dim = subblock_dim;
}

// Deterministic CPU launch (single-threaded).
template <typename KernelFn, typename... Args>
inline void launch_sequential(int64_t block_dim, int64_t subblock_dim, KernelFn fn, Args... args)
{
    set_grid_dim(block_dim, subblock_dim);
    for (int64_t b = 0; b < block_dim; ++b) {
        for (int64_t sb = 0; sb < subblock_dim; ++sb) {
            set_launch_context(b, sb, subblock_dim);
            fn(args...);
        }
    }
}

// Best-effort parallel CPU launch.
//
// Notes:
// - Uses TLS launch context, so each worker thread must set it before calling the kernel.
// - Ordering is not deterministic. Only use if the kernel has no cross-core dependencies.
// - `max_threads <= 0` means use hardware_concurrency().
template <typename KernelFn, typename... Args>
inline void launch_parallel(int64_t block_dim, int64_t subblock_dim, KernelFn fn, int max_threads, Args... args)
{
    const int64_t tasks = block_dim * subblock_dim;
    if (tasks <= 0) {
        return;
    }

    unsigned hc = std::thread::hardware_concurrency();
    int threads = max_threads > 0 ? max_threads : (hc ? static_cast<int>(hc) : 1);
    if (threads < 1) {
        threads = 1;
    }
    if (threads > tasks) {
        threads = static_cast<int>(tasks);
    }

    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));

    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&]() {
            set_grid_dim(block_dim, subblock_dim);
            while (true) {
                const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= tasks) {
                    break;
                }
                const int64_t b = i / subblock_dim;
                const int64_t sb = i % subblock_dim;
                set_launch_context(b, sb, subblock_dim);
                fn(args...);
            }
        });
    }

    for (auto &th : pool) {
        th.join();
    }
}

} // namespace cpu_sim

// Match PTOAS-emitted naming.
inline int64_t get_block_idx() { return cpu_sim::g_block_idx; }
inline int64_t get_block_num() { return cpu_sim::g_block_num; }
// Alias often used by toolchains.
inline int64_t get_blockdim() { return cpu_sim::g_block_num; }

inline int64_t get_subblockid() { return cpu_sim::g_subblock_id; }
inline int64_t get_subblockdim() { return cpu_sim::g_subblock_dim; }
// Alias to match MLIR naming.
inline int64_t get_subblock_num() { return cpu_sim::g_subblock_dim; }
} // namespace pto

#endif
