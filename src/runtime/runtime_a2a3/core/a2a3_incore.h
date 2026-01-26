/**
 * PTO Runtime - Ascend A2/A3 InCore Function Interface
 * 
 * This header defines the interface for InCore functions that run on AI cores
 * (Cube and Vector units). It provides:
 * - Platform intrinsics for tensor operations
 * - Memory management within InCore SRAM (L1/UB buffer)
 * - Synchronization primitives for core pipelines
 * 
 * Usage:
 * InCore functions should include this header to access Ascend-specific
 * intrinsics. The actual implementation differs between:
 * - ascend_a2a3: Real hardware with Ascend SDK
 * - ascend_a2a3_sim: Cycle-accurate simulator with heuristic models
 * 
 * Both platforms share the same InCore function source code by including
 * this header, which abstracts the platform differences.
 */

#ifndef A2A3_INCORE_H
#define A2A3_INCORE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Platform Detection (early - for macros only)
// =============================================================================

#if defined(A2A3_TARGET_HARDWARE)
    #define A2A3_PLATFORM_NAME "Ascend A2/A3 Hardware"
#elif defined(A2A3_TARGET_SIMULATOR)
    #define A2A3_PLATFORM_NAME "Ascend A2/A3 Simulator"
#else
    #define A2A3_TARGET_SIMULATOR
    #define A2A3_PLATFORM_NAME "Ascend A2/A3 Simulator (default)"
#endif

// =============================================================================
// Core Type Definitions
// =============================================================================

typedef enum {
    CORE_TYPE_VECTOR = 0,    // Vector core for element-wise ops
    CORE_TYPE_CUBE = 1,      // Cube core for matrix multiply
} A2A3CoreType;

// =============================================================================
// Memory Space Definitions
// =============================================================================

typedef enum {
    A2A3_MEM_GM = 0,         // Global Memory (DDR)
    A2A3_MEM_L2 = 1,         // L2 Cache (shared across cores)
    A2A3_MEM_L1 = 2,         // L1/UB Buffer (per-core scratchpad, 192KB)
    A2A3_MEM_REG = 3,        // Register file
} A2A3MemorySpace;

// InCore SRAM (L1/UB) configuration
#define A2A3_L1_SIZE_KB        192
#define A2A3_L1_SIZE_BYTES     (A2A3_L1_SIZE_KB * 1024)

// L2 Cache configuration
#define A2A3_L2_SIZE_MB        200
#define A2A3_L2_SIZE_BYTES     (A2A3_L2_SIZE_MB * 1024 * 1024)

// =============================================================================
// Tile Configuration
// =============================================================================

// Default tile sizes optimized for Ascend A2/A3
#define A2A3_TILE_ROWS_DEFAULT     32
#define A2A3_TILE_COLS_DEFAULT     128
#define A2A3_TILE_K_DEFAULT        64

// Maximum tile dimensions for Cube unit
#define A2A3_CUBE_MAX_M            256
#define A2A3_CUBE_MAX_N            256
#define A2A3_CUBE_MAX_K            256

// =============================================================================
// InCore Function Prototype
// =============================================================================

/**
 * InCore function type signature.
 * 
 * @param args      Array of argument pointers (tensor data pointers)
 * @param num_args  Number of arguments
 */
typedef void (*A2A3IncoreFunc)(void** args, int num_args);

/**
 * InCore function with cycle cost estimation.
 * Returns the number of cycles required for execution.
 */
typedef int64_t (*A2A3IncoreCycleFunc)(void** args, int num_args);

// =============================================================================
// Core Execution Context
// =============================================================================

/**
 * Context structure passed to InCore functions.
 * Contains information about the execution environment.
 */
typedef struct {
    A2A3CoreType core_type;      // Which core type this is executing on
    int32_t core_id;             // Core ID within the type
    int32_t worker_id;           // Global worker ID
    
    // L1 buffer management
    void* l1_buffer;             // Pointer to L1/UB scratchpad
    int32_t l1_size;             // Available L1 size in bytes
    int32_t l1_used;             // Currently used L1 bytes
    
    // Cycle tracking (for simulation)
    int64_t start_cycle;
    int64_t current_cycle;
    
    // Platform-specific data
    void* platform_data;
} A2A3CoreContext;

// =============================================================================
// L1 Buffer Management API
// =============================================================================

/**
 * Allocate memory from L1/UB buffer.
 * 
 * @param ctx   Core execution context
 * @param size  Size in bytes to allocate
 * @return Pointer to allocated memory, or NULL if insufficient space
 */
static inline void* a2a3_l1_alloc(A2A3CoreContext* ctx, int32_t size) {
    if (!ctx || ctx->l1_used + size > ctx->l1_size) {
        return NULL;
    }
    void* ptr = (char*)ctx->l1_buffer + ctx->l1_used;
    ctx->l1_used += size;
    return ptr;
}

/**
 * Reset L1 buffer allocation (free all).
 */
static inline void a2a3_l1_reset(A2A3CoreContext* ctx) {
    if (ctx) {
        ctx->l1_used = 0;
    }
}

/**
 * Get remaining L1 buffer space.
 */
static inline int32_t a2a3_l1_available(A2A3CoreContext* ctx) {
    return ctx ? (ctx->l1_size - ctx->l1_used) : 0;
}

// =============================================================================
// Data Transfer API (GM <-> L1)
// =============================================================================

/**
 * Load tile from Global Memory to L1 buffer.
 * 
 * @param ctx       Core context
 * @param l1_dst    Destination in L1 buffer
 * @param gm_src    Source in Global Memory
 * @param rows      Number of rows
 * @param cols      Number of columns
 * @param gm_stride Stride in GM (elements per row)
 */
void a2a3_load_tile(A2A3CoreContext* ctx, void* l1_dst, const void* gm_src,
                    int32_t rows, int32_t cols, int32_t gm_stride);

/**
 * Store tile from L1 buffer to Global Memory.
 */
void a2a3_store_tile(A2A3CoreContext* ctx, void* gm_dst, const void* l1_src,
                     int32_t rows, int32_t cols, int32_t gm_stride);

// =============================================================================
// Vector Operations (element-wise)
// =============================================================================

// Binary operations
void a2a3_vec_add(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n);
void a2a3_vec_sub(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n);
void a2a3_vec_mul(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n);
void a2a3_vec_div(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n);
void a2a3_vec_max(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n);
void a2a3_vec_min(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n);

// Scalar-vector operations
void a2a3_vec_adds(A2A3CoreContext* ctx, void* dst, const void* a, float s, int32_t n);
void a2a3_vec_muls(A2A3CoreContext* ctx, void* dst, const void* a, float s, int32_t n);

// Unary operations
void a2a3_vec_neg(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_abs(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_sqrt(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_rsqrt(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_exp(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_log(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_recip(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);

// Activation functions
void a2a3_vec_relu(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_silu(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_gelu(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_tanh(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);
void a2a3_vec_sigmoid(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n);

// =============================================================================
// Reduction Operations
// =============================================================================

void a2a3_row_sum(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols);
void a2a3_row_max(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols);
void a2a3_row_min(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols);
void a2a3_col_sum(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols);

// =============================================================================
// Broadcast Operations
// =============================================================================

void a2a3_row_expand_add(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols);
void a2a3_row_expand_sub(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols);
void a2a3_row_expand_mul(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols);
void a2a3_row_expand_div(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols);

// =============================================================================
// Matrix Operations (Cube Core)
// =============================================================================

/**
 * Matrix multiply: C = A @ B
 * 
 * @param ctx   Core context (must be CORE_TYPE_CUBE)
 * @param C     Output matrix (M x N)
 * @param A     Left input matrix (M x K)
 * @param B     Right input matrix (K x N)
 * @param M, K, N   Matrix dimensions
 */
void a2a3_matmul(A2A3CoreContext* ctx, void* C, const void* A, const void* B,
                 int32_t M, int32_t K, int32_t N);

/**
 * Matrix multiply accumulate: C += A @ B
 */
void a2a3_matmul_acc(A2A3CoreContext* ctx, void* C, const void* A, const void* B,
                     int32_t M, int32_t K, int32_t N);

// =============================================================================
// Synchronization
// =============================================================================

/**
 * Memory fence - ensure all previous memory operations complete.
 */
void a2a3_memory_fence(A2A3CoreContext* ctx);

/**
 * Pipeline barrier - synchronize vector and cube pipelines.
 */
void a2a3_pipeline_barrier(A2A3CoreContext* ctx);

// =============================================================================
// Platform-Specific Intrinsic Implementations
// =============================================================================
// Include after all type definitions

#if defined(A2A3_TARGET_HARDWARE)
    #include "a2a3_intrinsics_hw.h"
#else
    #include "a2a3_intrinsics_sim.h"
#endif

#ifdef __cplusplus
}
#endif

#endif // A2A3_INCORE_H
