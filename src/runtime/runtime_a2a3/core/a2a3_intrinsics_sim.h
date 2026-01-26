/**
 * PTO Runtime - Ascend A2/A3 Simulator Intrinsics
 * 
 * This header provides intrinsic implementations for the SIMULATOR.
 * It implements tensor operations using standard C code and provides
 * cycle cost estimation for accurate timing simulation.
 * 
 * This file is used when compiling for:
 * - ascend_a2a3_sim platform (cycle-accurate simulator)
 * - Development/testing without hardware
 * 
 * The same InCore function source code can be used on both real hardware
 * and simulator by including the appropriate intrinsics header.
 */

#ifndef A2A3_INTRINSICS_SIM_H
#define A2A3_INTRINSICS_SIM_H

#include <math.h>
#include <string.h>

// Note: This header is included within a2a3_incore.h which provides:
// - A2A3CoreContext typedef
// - extern "C" wrapping

// =============================================================================
// Cycle Cost Model (Simulator)
// =============================================================================

// Approximate cycle costs for Ascend 910B @ 1.8GHz
#define A2A3_SIM_CYCLES_LOAD       100   // GM to L1
#define A2A3_SIM_CYCLES_STORE      100   // L1 to GM
#define A2A3_SIM_CYCLES_VEC_BASIC  8     // add, sub, mul, etc.
#define A2A3_SIM_CYCLES_VEC_DIV    16    // division
#define A2A3_SIM_CYCLES_VEC_TRANS  24    // exp, log, sqrt
#define A2A3_SIM_CYCLES_VEC_ACT    20    // silu, gelu
#define A2A3_SIM_CYCLES_REDUCTION  20    // rowsum, rowmax
#define A2A3_SIM_CYCLES_BROADCAST  12    // rowexpand
#define A2A3_SIM_CYCLES_MATMUL     50    // per 32x32 output tile

#define A2A3_INTRINSIC_IMPL_TYPE "Simulator (C Reference)"

// =============================================================================
// Simulator-Specific Context Extensions
// =============================================================================

typedef struct {
    // Cycle accounting
    int64_t total_cycles;
    int64_t memory_cycles;
    int64_t compute_cycles;
    
    // Statistics
    int64_t num_loads;
    int64_t num_stores;
    int64_t num_vec_ops;
    int64_t num_cube_ops;
    
    // Trace output
    void* trace_file;
    int trace_enabled;
} A2A3SimulatorData;

// =============================================================================
// Cycle Accounting Macros
// =============================================================================

#define A2A3_SIM_ADD_CYCLES(ctx, cycles) do { \
    if ((ctx) && (ctx)->platform_data) { \
        ((A2A3SimulatorData*)(ctx)->platform_data)->total_cycles += (cycles); \
        ((A2A3SimulatorData*)(ctx)->platform_data)->compute_cycles += (cycles); \
        (ctx)->current_cycle += (cycles); \
    } \
} while(0)

#define A2A3_SIM_ADD_MEMORY_CYCLES(ctx, cycles) do { \
    if ((ctx) && (ctx)->platform_data) { \
        ((A2A3SimulatorData*)(ctx)->platform_data)->total_cycles += (cycles); \
        ((A2A3SimulatorData*)(ctx)->platform_data)->memory_cycles += (cycles); \
        (ctx)->current_cycle += (cycles); \
    } \
} while(0)

// =============================================================================
// Inline Implementations for Simulator
// =============================================================================

// Scale cycle cost by data size
static inline int64_t a2a3_sim_scale_cycles(int64_t base_cycles, int32_t elements) {
    // Scale based on number of elements relative to 32x128 base tile
    int64_t base_elements = 32 * 128;
    int64_t scale = (elements + base_elements - 1) / base_elements;
    return base_cycles * (scale > 0 ? scale : 1);
}

// =============================================================================
// Data Transfer Implementation (Simulator)
// =============================================================================

static inline void a2a3_load_tile_sim(A2A3CoreContext* ctx, void* l1_dst, const void* gm_src,
                                       int32_t rows, int32_t cols, int32_t gm_stride) {
    // Simulate memory copy
    float* dst = (float*)l1_dst;
    const float* src = (const float*)gm_src;
    
    for (int r = 0; r < rows; r++) {
        memcpy(dst + r * cols, src + r * gm_stride, cols * sizeof(float));
    }
    
    // Account for cycles
    int64_t cycles = a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_LOAD, rows * cols);
    A2A3_SIM_ADD_MEMORY_CYCLES(ctx, cycles);
}

static inline void a2a3_store_tile_sim(A2A3CoreContext* ctx, void* gm_dst, const void* l1_src,
                                        int32_t rows, int32_t cols, int32_t gm_stride) {
    float* dst = (float*)gm_dst;
    const float* src = (const float*)l1_src;
    
    for (int r = 0; r < rows; r++) {
        memcpy(dst + r * gm_stride, src + r * cols, cols * sizeof(float));
    }
    
    int64_t cycles = a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_STORE, rows * cols);
    A2A3_SIM_ADD_MEMORY_CYCLES(ctx, cycles);
}

// =============================================================================
// Vector Operations Implementation (Simulator)
// =============================================================================

static inline void a2a3_vec_add_sim(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (int i = 0; i < n; i++) d[i] = va[i] + vb[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

static inline void a2a3_vec_sub_sim(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (int i = 0; i < n; i++) d[i] = va[i] - vb[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

static inline void a2a3_vec_mul_sim(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (int i = 0; i < n; i++) d[i] = va[i] * vb[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

static inline void a2a3_vec_div_sim(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (int i = 0; i < n; i++) d[i] = va[i] / vb[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_DIV, n));
}

static inline void a2a3_vec_max_sim(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (int i = 0; i < n; i++) d[i] = va[i] > vb[i] ? va[i] : vb[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

static inline void a2a3_vec_min_sim(A2A3CoreContext* ctx, void* dst, const void* a, const void* b, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (int i = 0; i < n; i++) d[i] = va[i] < vb[i] ? va[i] : vb[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

// Scalar-vector operations
static inline void a2a3_vec_adds_sim(A2A3CoreContext* ctx, void* dst, const void* a, float s, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = va[i] + s;
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

static inline void a2a3_vec_muls_sim(A2A3CoreContext* ctx, void* dst, const void* a, float s, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = va[i] * s;
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

// Unary operations
static inline void a2a3_vec_neg_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = -va[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(4, n));  // Simple negation
}

static inline void a2a3_vec_abs_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = fabsf(va[i]);
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(4, n));
}

static inline void a2a3_vec_sqrt_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = sqrtf(va[i]);
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(16, n));
}

static inline void a2a3_vec_rsqrt_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = 1.0f / sqrtf(va[i]);
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(16, n));
}

static inline void a2a3_vec_exp_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = expf(va[i]);
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_TRANS, n));
}

static inline void a2a3_vec_log_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = logf(va[i]);
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_TRANS, n));
}

static inline void a2a3_vec_recip_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = 1.0f / va[i];
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_DIV, n));
}

// Activation functions
static inline void a2a3_vec_relu_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = va[i] > 0 ? va[i] : 0;
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_BASIC, n));
}

static inline void a2a3_vec_silu_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) {
        float x = va[i];
        d[i] = x / (1.0f + expf(-x));  // SiLU = x * sigmoid(x)
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_ACT, n));
}

static inline void a2a3_vec_gelu_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    const float sqrt2_inv = 0.7071067811865476f;
    for (int i = 0; i < n; i++) {
        float x = va[i];
        d[i] = 0.5f * x * (1.0f + erff(x * sqrt2_inv));
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(28, n));  // GELU is expensive
}

static inline void a2a3_vec_tanh_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = tanhf(va[i]);
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_TRANS, n));
}

static inline void a2a3_vec_sigmoid_sim(A2A3CoreContext* ctx, void* dst, const void* a, int32_t n) {
    float* d = (float*)dst;
    const float* va = (const float*)a;
    for (int i = 0; i < n; i++) d[i] = 1.0f / (1.0f + expf(-va[i]));
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_VEC_TRANS, n));
}

// =============================================================================
// Reduction Operations Implementation (Simulator)
// =============================================================================

static inline void a2a3_row_sum_sim(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* s = (const float*)src;
    for (int r = 0; r < rows; r++) {
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            sum += s[r * cols + c];
        }
        d[r] = sum;
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_REDUCTION, rows * cols));
}

static inline void a2a3_row_max_sim(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* s = (const float*)src;
    for (int r = 0; r < rows; r++) {
        float max_val = s[r * cols];
        for (int c = 1; c < cols; c++) {
            if (s[r * cols + c] > max_val) max_val = s[r * cols + c];
        }
        d[r] = max_val;
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_REDUCTION, rows * cols));
}

static inline void a2a3_row_min_sim(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* s = (const float*)src;
    for (int r = 0; r < rows; r++) {
        float min_val = s[r * cols];
        for (int c = 1; c < cols; c++) {
            if (s[r * cols + c] < min_val) min_val = s[r * cols + c];
        }
        d[r] = min_val;
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_REDUCTION, rows * cols));
}

static inline void a2a3_col_sum_sim(A2A3CoreContext* ctx, void* dst, const void* src, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* s = (const float*)src;
    memset(d, 0, cols * sizeof(float));
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            d[c] += s[r * cols + c];
        }
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(A2A3_SIM_CYCLES_REDUCTION, rows * cols));
}

// =============================================================================
// Broadcast Operations Implementation (Simulator)
// =============================================================================

static inline void a2a3_row_expand_add_sim(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* m = (const float*)mat;
    const float* v = (const float*)vec;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            d[r * cols + c] = m[r * cols + c] + v[r];
        }
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(16, rows * cols));
}

static inline void a2a3_row_expand_sub_sim(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* m = (const float*)mat;
    const float* v = (const float*)vec;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            d[r * cols + c] = m[r * cols + c] - v[r];
        }
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(16, rows * cols));
}

static inline void a2a3_row_expand_mul_sim(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* m = (const float*)mat;
    const float* v = (const float*)vec;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            d[r * cols + c] = m[r * cols + c] * v[r];
        }
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(16, rows * cols));
}

static inline void a2a3_row_expand_div_sim(A2A3CoreContext* ctx, void* dst, const void* mat, const void* vec, int32_t rows, int32_t cols) {
    float* d = (float*)dst;
    const float* m = (const float*)mat;
    const float* v = (const float*)vec;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            d[r * cols + c] = m[r * cols + c] / v[r];
        }
    }
    A2A3_SIM_ADD_CYCLES(ctx, a2a3_sim_scale_cycles(20, rows * cols));
}

// =============================================================================
// Matrix Operations Implementation (Simulator)
// =============================================================================

static inline void a2a3_matmul_sim(A2A3CoreContext* ctx, void* C, const void* A, const void* B,
                                    int32_t M, int32_t K, int32_t N) {
    float* c = (float*)C;
    const float* a = (const float*)A;
    const float* b = (const float*)B;
    
    // Simple reference implementation
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = sum;
        }
    }
    
    // Cycle cost based on output tile count
    int64_t output_tiles = ((M + 31) / 32) * ((N + 31) / 32);
    int64_t k_iterations = (K + 63) / 64;
    A2A3_SIM_ADD_CYCLES(ctx, A2A3_SIM_CYCLES_MATMUL * output_tiles * k_iterations);
}

static inline void a2a3_matmul_acc_sim(A2A3CoreContext* ctx, void* C, const void* A, const void* B,
                                        int32_t M, int32_t K, int32_t N) {
    float* c = (float*)C;
    const float* a = (const float*)A;
    const float* b = (const float*)B;
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = c[m * N + n];  // Accumulate
            for (int k = 0; k < K; k++) {
                sum += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = sum;
        }
    }
    
    int64_t output_tiles = ((M + 31) / 32) * ((N + 31) / 32);
    int64_t k_iterations = (K + 63) / 64;
    A2A3_SIM_ADD_CYCLES(ctx, A2A3_SIM_CYCLES_MATMUL * output_tiles * k_iterations);
}

// =============================================================================
// Synchronization Implementation (Simulator - No-ops)
// =============================================================================

static inline void a2a3_memory_fence_sim(A2A3CoreContext* ctx) {
    (void)ctx;  // No-op in simulation
}

static inline void a2a3_pipeline_barrier_sim(A2A3CoreContext* ctx) {
    (void)ctx;  // No-op in simulation
}

// =============================================================================
// Map generic names to simulator implementations
// =============================================================================

#define a2a3_load_tile    a2a3_load_tile_sim
#define a2a3_store_tile   a2a3_store_tile_sim

#define a2a3_vec_add      a2a3_vec_add_sim
#define a2a3_vec_sub      a2a3_vec_sub_sim
#define a2a3_vec_mul      a2a3_vec_mul_sim
#define a2a3_vec_div      a2a3_vec_div_sim
#define a2a3_vec_max      a2a3_vec_max_sim
#define a2a3_vec_min      a2a3_vec_min_sim
#define a2a3_vec_adds     a2a3_vec_adds_sim
#define a2a3_vec_muls     a2a3_vec_muls_sim
#define a2a3_vec_neg      a2a3_vec_neg_sim
#define a2a3_vec_abs      a2a3_vec_abs_sim
#define a2a3_vec_sqrt     a2a3_vec_sqrt_sim
#define a2a3_vec_rsqrt    a2a3_vec_rsqrt_sim
#define a2a3_vec_exp      a2a3_vec_exp_sim
#define a2a3_vec_log      a2a3_vec_log_sim
#define a2a3_vec_recip    a2a3_vec_recip_sim
#define a2a3_vec_relu     a2a3_vec_relu_sim
#define a2a3_vec_silu     a2a3_vec_silu_sim
#define a2a3_vec_gelu     a2a3_vec_gelu_sim
#define a2a3_vec_tanh     a2a3_vec_tanh_sim
#define a2a3_vec_sigmoid  a2a3_vec_sigmoid_sim

#define a2a3_row_sum      a2a3_row_sum_sim
#define a2a3_row_max      a2a3_row_max_sim
#define a2a3_row_min      a2a3_row_min_sim
#define a2a3_col_sum      a2a3_col_sum_sim

#define a2a3_row_expand_add  a2a3_row_expand_add_sim
#define a2a3_row_expand_sub  a2a3_row_expand_sub_sim
#define a2a3_row_expand_mul  a2a3_row_expand_mul_sim
#define a2a3_row_expand_div  a2a3_row_expand_div_sim

#define a2a3_matmul       a2a3_matmul_sim
#define a2a3_matmul_acc   a2a3_matmul_acc_sim

#define a2a3_memory_fence    a2a3_memory_fence_sim
#define a2a3_pipeline_barrier a2a3_pipeline_barrier_sim

#endif // A2A3_INTRINSICS_SIM_H
