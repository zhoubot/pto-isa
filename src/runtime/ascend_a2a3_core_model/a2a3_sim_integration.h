/**
 * Ascend A2/A3 Core Simulator - Runtime Integration
 * 
 * This header provides the interface for integrating the A2A3 Core Simulator
 * with the PTO runtime for cycle-accurate simulation.
 * 
 * Usage:
 * 1. Define A2A3_CORE_SIM_AVAILABLE before including this header
 * 2. Link with liba2a3_core.a
 * 3. Call a2a3_sim_init() before running simulation
 * 4. Use a2a3_sim_get_task_cycles() to get cycle costs for tasks
 * 5. Call a2a3_sim_cleanup() when done
 */

#ifndef A2A3_SIM_INTEGRATION_H
#define A2A3_SIM_INTEGRATION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Integration API
// =============================================================================

/**
 * Initialize the A2A3 simulation subsystem
 * Call this before running any simulation
 */
void a2a3_sim_init(void);

/**
 * Cleanup the A2A3 simulation subsystem
 * Call this when done with simulation
 */
void a2a3_sim_cleanup(void);

/**
 * Get cycle cost for executing a task
 * 
 * @param func_name Name of the InCore function
 * @param is_cube Whether this is a cube (matmul) operation
 * @param tile_size Number of elements in the tile
 * @return Estimated cycle count
 */
int64_t a2a3_sim_get_task_cycles(const char* func_name, bool is_cube, int64_t tile_size);

/**
 * Register an InCore function for simulation
 * 
 * @param func_name Name of the InCore function
 * @param is_cube Whether this is a cube (matmul) operation
 * @param instruction_code The Ascend instruction code (newline-separated)
 * @param tile_rows Tile row dimension
 * @param tile_cols Tile column dimension
 * @return Function ID (>= 0) or -1 on error
 */
int a2a3_sim_register_function(const char* func_name, bool is_cube,
                                const char* instruction_code,
                                int tile_rows, int tile_cols);

/**
 * Enable/disable tracing for the simulation
 */
void a2a3_sim_enable_trace(bool enable);

/**
 * Print simulation statistics
 */
void a2a3_sim_print_stats(void);

// =============================================================================
// Implementation (when A2A3_CORE_SIM_AVAILABLE is defined)
// =============================================================================

#ifdef A2A3_CORE_SIM_AVAILABLE

#include "a2a3_core_model.h"
#include "a2a3_incore_sim.h"

// Global simulator instance
static IncoreSimulator* g_a2a3_sim = NULL;
static bool g_a2a3_sim_trace = false;

void a2a3_sim_init(void) {
    if (!g_a2a3_sim) {
        g_a2a3_sim = a2a3_incore_sim_create();
    }
}

void a2a3_sim_cleanup(void) {
    if (g_a2a3_sim) {
        a2a3_incore_sim_destroy(g_a2a3_sim);
        g_a2a3_sim = NULL;
    }
}

int64_t a2a3_sim_get_task_cycles(const char* func_name, bool is_cube, int64_t tile_size) {
    if (g_a2a3_sim) {
        // Try to find registered function
        int func_id = a2a3_incore_sim_find(g_a2a3_sim, func_name);
        if (func_id >= 0) {
            return a2a3_incore_sim_execute(g_a2a3_sim, func_id);
        }
    }
    // Fallback to heuristic
    return a2a3_get_incore_cycle_cost(func_name, tile_size);
}

int a2a3_sim_register_function(const char* func_name, bool is_cube,
                                const char* instruction_code,
                                int tile_rows, int tile_cols) {
    if (!g_a2a3_sim) {
        a2a3_sim_init();
    }
    
    CoreType core_type = is_cube ? CORE_TYPE_CUBE : CORE_TYPE_VECTOR;
    return a2a3_incore_sim_register_code(g_a2a3_sim, func_name, core_type,
                                          instruction_code, tile_rows, tile_cols);
}

void a2a3_sim_enable_trace(bool enable) {
    g_a2a3_sim_trace = enable;
    if (g_a2a3_sim) {
        if (enable) {
            a2a3_incore_sim_enable_trace(g_a2a3_sim, stdout);
        } else {
            a2a3_incore_sim_disable_trace(g_a2a3_sim);
        }
    }
}

void a2a3_sim_print_stats(void) {
    if (g_a2a3_sim) {
        a2a3_incore_sim_print_stats(g_a2a3_sim);
    }
}

#else  // A2A3_CORE_SIM_AVAILABLE not defined

// Stub implementations when core simulator not available

void a2a3_sim_init(void) {
    // No-op
}

void a2a3_sim_cleanup(void) {
    // No-op
}

int64_t a2a3_sim_get_task_cycles(const char* func_name, bool is_cube, int64_t tile_size) {
    // Heuristic-based estimation
    (void)is_cube;
    
    if (!func_name) return 10;
    
    // Matrix multiply (Cube)
    if (strstr(func_name, "matmul") || strstr(func_name, "gemm") || 
        strstr(func_name, "linear")) {
        return 50 + (tile_size / 64);
    }
    
    // RMSNorm (multiple vector ops + reduction)
    if (strstr(func_name, "rmsnorm") || strstr(func_name, "layernorm")) {
        return 70;
    }
    
    // Softmax (reduction + element-wise)
    if (strstr(func_name, "softmax")) {
        return 70;
    }
    
    // RoPE (element-wise)
    if (strstr(func_name, "rope") || strstr(func_name, "rotary")) {
        return 60;
    }
    
    // SwiGLU (activation + multiply)
    if (strstr(func_name, "swiglu") || strstr(func_name, "silu")) {
        return 25;
    }
    
    // Attention score/output
    if (strstr(func_name, "attention") || strstr(func_name, "score")) {
        return 60;
    }
    
    // Reduction ops
    if (strstr(func_name, "rowsum") || strstr(func_name, "rowmax") ||
        strstr(func_name, "colsum") || strstr(func_name, "reduce")) {
        return 20;
    }
    
    // Simple element-wise ops
    if (strstr(func_name, "add") || strstr(func_name, "mul") ||
        strstr(func_name, "sub") || strstr(func_name, "div")) {
        return 10;
    }
    
    // Activation functions
    if (strstr(func_name, "relu") || strstr(func_name, "gelu") ||
        strstr(func_name, "sigmoid") || strstr(func_name, "tanh")) {
        return 15;
    }
    
    // Math functions
    if (strstr(func_name, "exp") || strstr(func_name, "log") ||
        strstr(func_name, "sqrt") || strstr(func_name, "rsqrt")) {
        return 10;
    }
    
    // Default
    return 10;
}

int a2a3_sim_register_function(const char* func_name, bool is_cube,
                                const char* instruction_code,
                                int tile_rows, int tile_cols) {
    (void)func_name;
    (void)is_cube;
    (void)instruction_code;
    (void)tile_rows;
    (void)tile_cols;
    // No-op when simulator not available
    return -1;
}

void a2a3_sim_enable_trace(bool enable) {
    (void)enable;
    // No-op
}

void a2a3_sim_print_stats(void) {
    printf("A2A3 Core Simulator not available (not linked with liba2a3_core.a)\n");
}

#endif  // A2A3_CORE_SIM_AVAILABLE

#ifdef __cplusplus
}
#endif

#endif // A2A3_SIM_INTEGRATION_H
