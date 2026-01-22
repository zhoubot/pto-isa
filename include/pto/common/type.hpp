/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef _PTO_INCLUDE_NPU_TYPE_H_
#define _PTO_INCLUDE_NPU_TYPE_H_
#if defined(MEMORY_BASE) || defined(REGISTER_BASE)
#define AICORE [aicore]
#else
#define AICORE
#endif
#define PTO_INLINE inline __attribute__((always_inline))

// for pto instruction declaration
#define PTO_INST AICORE PTO_INLINE __attribute__((visibility("default")))
// for pto internal implementation
#define PTO_INTERNAL AICORE PTO_INLINE

#define OP_NAME(Name) __attribute__((vf_name(#Name)))
#define OP_TYPE(TypeName) __attribute__((vf_kind(#TypeName)))

// -----------------------------------------------------------------------------
// PTO assertion helpers
//
// Goals:
// - Provide a consistent diagnostic prefix across compile-time and runtime checks.
// - Always print/encode the violated condition when possible.
// - Provide a stable “next step” for users: see docs/coding/debug.md.
//
// Note:
// - `static_assert` diagnostics are compile-time only; we use a macro so we can
//   include the condition string and a docs hint consistently.
// - CPU simulator uses `PTO_CPU_ASSERT(...)` for runtime checks; it prints and
//   aborts (always enabled).
// -----------------------------------------------------------------------------

#define PTO_DETAIL_STR_(x) #x
#define PTO_DETAIL_STR(x) PTO_DETAIL_STR_(x)

#define PTO_STATIC_ASSERT_1(cond)                                                                        \
    static_assert((cond),                                                                                \
        "[PTO][SA] Constraint violated. "                                                                \
        "Condition: " #cond ". "                                                                         \
        "Hint: see docs/coding/debug.md and search for " __FILE__ ":" PTO_DETAIL_STR(__LINE__))

#define PTO_STATIC_ASSERT_2(cond, msg)                                                                   \
    static_assert((cond),                                                                                \
        "[PTO][SA] " msg " "                                                                             \
        "Condition: " #cond ". "                                                                         \
        "Hint: see docs/coding/debug.md and search for " __FILE__ ":" PTO_DETAIL_STR(__LINE__))

#define PTO_DETAIL_GET_MACRO(_1, _2, NAME, ...) NAME
#define PTO_STATIC_ASSERT(...)                                                                           \
    PTO_DETAIL_GET_MACRO(__VA_ARGS__, PTO_STATIC_ASSERT_2, PTO_STATIC_ASSERT_1)(__VA_ARGS__)

#if defined(__CPU_SIM)
#include <cstdio>
#include <cstdlib>

#define PTO_CPU_ASSERT_1(cond)                                                                           \
    do {                                                                                                 \
        if (!(cond)) {                                                                                   \
            std::fprintf(stderr,                                                                         \
                "[PTO][CA] Constraint violated. Condition: %s. Hint: see docs/coding/debug.md and "      \
                "search for %s:%d\n",                                                                    \
                #cond, __FILE__, __LINE__);                                                              \
            std::abort();                                                                                \
        }                                                                                                \
    } while (0)

#define PTO_CPU_ASSERT_2(cond, msg)                                                                      \
    do {                                                                                                 \
        if (!(cond)) {                                                                                   \
            std::fprintf(stderr,                                                                         \
                "[PTO][CA] %s Condition: %s. Hint: see docs/coding/debug.md and search for %s:%d\n",     \
                (msg), #cond, __FILE__, __LINE__);                                                       \
            std::abort();                                                                                \
        }                                                                                                \
    } while (0)

#define PTO_CPU_ASSERT(...)                                                                              \
    PTO_DETAIL_GET_MACRO(__VA_ARGS__, PTO_CPU_ASSERT_2, PTO_CPU_ASSERT_1)(__VA_ARGS__)
#else
// Non-CPU builds should not depend on CPU-only assertion behavior.
#define PTO_CPU_ASSERT(...) ((void)0)
#endif

namespace pto {
    // 01-bits patterns are read from right to left.
    // Right bits are low bits, corresponding to low index positions of data.
    enum class MaskPattern : uint8_t
    {
        // 以下1~7与指令VREDUCEv2的pattern mode保持一致
        P0101 = 1,  // 1: 01010101...0101 # 每个repeat内每两个元素取第一个元素
        P1010 = 2,  // 2: 10101010...1010 # 每个repeat内每两个元素取第二个元素
        P0001 = 3,  // 3: 00010001...0001 # 每个repeat内每四个元素取第一个元素
        P0010 = 4,  // 4: 00100010...0010 # 每个repeat内每四个元素取第二个元素
        P0100 = 5,  // 5: 01000100...0100 # 每个repeat内每四个元素取第三个元素
        P1000 = 6,  // 6: 10001000...1000 # 每个repeat内每四个元素取第四个元素
        P1111 = 7,  // 7: 11111111...1111 # 每个repeat内取全部元素
    };

    enum class CmpMode : uint8_t {
        EQ = 0,
        NE = 1,
        LT = 2,
        LE = 3,
        GT = 4,
        GE = 5,
    };
}

#if defined(__CPU_SIM)
    // Note: clang version should be >=15 and gcc version should be >=14
    #if defined(__has_include) && __has_include(<stdfloat>) && !(defined(__clang__))
        #include <stdfloat>
        typedef std::float16_t half;
        typedef std::float16_t bfloat16_t;
        typedef std::float16_t aclFloat16;
    #else
        // macOS libc++ (and some other toolchains) may not ship <stdfloat> yet.
        // For CPU simulation, a best-effort 16-bit float type is sufficient.
        typedef _Float16 half;
        typedef _Float16 bfloat16_t;
        typedef _Float16 aclFloat16;
    #endif
#endif

#endif
