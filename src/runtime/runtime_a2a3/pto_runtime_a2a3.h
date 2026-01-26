/**
 * PTO Runtime System - A2A3 (Ascend) Platform Header
 * 
 * This header provides the unified interface for the Ascend A2/A3 platform.
 * The implementation is split into three layers:
 * 
 * 1. Host Layer (host/)
 *    - Runtime initialization and shutdown
 *    - Memory management and data transfer
 *    - Worker thread management
 * 
 * 2. Orchestration Layer (orchestration/)
 *    - Dual ready queues (Vector and Cube)
 *    - Task dependency management
 *    - Work distribution
 * 
 * 3. Core Layer (core/)
 *    - InCore function interface
 *    - Platform intrinsics
 *    - L1 buffer management
 * 
 * A2A3 Architecture Notes:
 * - Vector cores (48): element-wise ops, reductions, memory ops
 * - Cube cores (24): matrix multiplication
 * - Memory: 32GB GM, 200MB L2, 192KB L1/UB per core
 */

#ifndef PTO_RUNTIME_A2A3_H
#define PTO_RUNTIME_A2A3_H

#include "../pto_runtime_common.h"

// Include layer headers (host and orchestration only)
// Note: core/a2a3_incore.h is for InCore function code, not runtime
#include "host/a2a3_host.h"
#include "orchestration/a2a3_orchestration.h"

// Core intrinsics are only included when compiling InCore functions
// Use #include "runtime_a2a3/core/a2a3_incore.h" in InCore code

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// A2A3 Platform API (for backward compatibility)
// =============================================================================

// Re-export commonly used functions from host layer
// (pto_runtime_enable_a2a3_sim and runtime_entry_a2a3 are in a2a3_host.h)

// =============================================================================
// A2A3 Dual Queue API (from orchestration layer)
// =============================================================================

/**
 * Get next ready task for a vector worker (is_cube=0 tasks only)
 * @return task_id or -1 if no tasks ready
 */
static inline int32_t pto_get_ready_task_vector(PTORuntime* rt) {
    return a2a3_orch_get_vector_task(rt);
}

/**
 * Get next ready task for a cube worker (is_cube=1 tasks only)
 * @return task_id or -1 if no tasks ready
 */
static inline int32_t pto_get_ready_task_cube(PTORuntime* rt) {
    return a2a3_orch_get_cube_task(rt);
}

/**
 * Thread-safe get ready task for vector worker (blocking)
 */
static inline int32_t pto_get_ready_task_vector_blocking(PTORuntime* rt) {
    return a2a3_orch_get_vector_task_blocking(rt);
}

/**
 * Thread-safe get ready task for cube worker (blocking)
 */
static inline int32_t pto_get_ready_task_cube_blocking(PTORuntime* rt) {
    return a2a3_orch_get_cube_task_blocking(rt);
}

// =============================================================================
// A2A3 Task Submit/Complete (from orchestration layer)
// =============================================================================

/**
 * A2A3 task submit - routes to appropriate dual queue
 */
static inline void pto_task_submit_a2a3(PTORuntime* rt, int32_t task_id) {
    a2a3_orch_submit_task(rt, task_id);
}

/**
 * A2A3 task complete with dedicated dependency management
 */
static inline void pto_task_complete_a2a3(PTORuntime* rt, int32_t task_id) {
    a2a3_orch_complete_task(rt, task_id);
}

/**
 * Thread-safe A2A3 task complete
 */
static inline void pto_task_complete_a2a3_threadsafe(PTORuntime* rt, int32_t task_id) {
    a2a3_orch_complete_task_threadsafe(rt, task_id);
}

// =============================================================================
// Legacy Definitions (for compatibility)
// =============================================================================

// Platform configuration macros
#ifndef A2A3_DEFAULT_VECTOR_WORKERS
#define A2A3_DEFAULT_VECTOR_WORKERS  48
#endif

#ifndef A2A3_DEFAULT_CUBE_WORKERS
#define A2A3_DEFAULT_CUBE_WORKERS    24
#endif

#ifdef __cplusplus
}
#endif

#endif // PTO_RUNTIME_A2A3_H
