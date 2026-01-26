/**
 * PTO Runtime - Ascend A2/A3 Orchestration Layer
 * 
 * This module handles task orchestration for the A2A3 platform:
 * - Dual ready queues (Vector and Cube)
 * - Task dependency management
 * - Work distribution across heterogeneous cores
 * 
 * The orchestration code conceptually runs on the "Orchestration Processor"
 * which manages task dispatch to Vector (48) and Cube (24) cores.
 * 
 * In simulation mode, this runs on the host CPU simulating the behavior
 * of the orchestration processor.
 */

#ifndef A2A3_ORCHESTRATION_H
#define A2A3_ORCHESTRATION_H

#include "../../pto_runtime_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Orchestration Configuration
// =============================================================================

// Default worker configuration for Ascend 910B (A2/A3)
#define A2A3_NUM_VECTOR_CORES    48
#define A2A3_NUM_CUBE_CORES      24
#define A2A3_TOTAL_CORES         (A2A3_NUM_VECTOR_CORES + A2A3_NUM_CUBE_CORES)

// Queue sizes
#define A2A3_QUEUE_SIZE          8192

// =============================================================================
// Dual Ready Queue Interface
// =============================================================================

/**
 * Initialize dual ready queues for A2A3 execution.
 * 
 * @param rt  Runtime context
 * @param num_vector_workers  Number of vector worker threads
 * @param num_cube_workers    Number of cube worker threads
 */
void a2a3_orch_init(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers);

/**
 * Submit a task to the appropriate ready queue.
 * Routes based on task's is_cube flag.
 */
void a2a3_orch_submit_task(PTORuntime* rt, int32_t task_id);

/**
 * Mark a task as complete and propagate to dependents.
 * Newly ready tasks are routed to appropriate queues.
 */
void a2a3_orch_complete_task(PTORuntime* rt, int32_t task_id);

/**
 * Thread-safe version of task completion.
 */
void a2a3_orch_complete_task_threadsafe(PTORuntime* rt, int32_t task_id);

// =============================================================================
// Queue Access (for workers)
// =============================================================================

/**
 * Get next ready task for a Vector worker.
 * @return task_id or -1 if queue empty
 */
int32_t a2a3_orch_get_vector_task(PTORuntime* rt);

/**
 * Get next ready task for a Cube worker.
 * @return task_id or -1 if queue empty
 */
int32_t a2a3_orch_get_cube_task(PTORuntime* rt);

/**
 * Blocking versions (wait for tasks to become available).
 */
int32_t a2a3_orch_get_vector_task_blocking(PTORuntime* rt);
int32_t a2a3_orch_get_cube_task_blocking(PTORuntime* rt);

// =============================================================================
// Dependency Management
// =============================================================================

/**
 * Route a task to the appropriate ready queue based on its type.
 * This is the core of the dependency management module.
 */
void a2a3_orch_route_to_queue(PTORuntime* rt, int32_t task_id);

/**
 * Thread-safe version of routing.
 */
void a2a3_orch_route_to_queue_threadsafe(PTORuntime* rt, int32_t task_id);

// =============================================================================
// Statistics and Monitoring
// =============================================================================

typedef struct {
    int64_t vector_tasks_executed;
    int64_t cube_tasks_executed;
    int64_t total_vector_cycles;
    int64_t total_cube_cycles;
    int64_t max_vector_queue_depth;
    int64_t max_cube_queue_depth;
} A2A3OrchStats;

/**
 * Get orchestration statistics.
 */
void a2a3_orch_get_stats(PTORuntime* rt, A2A3OrchStats* stats);

/**
 * Print orchestration statistics.
 */
void a2a3_orch_print_stats(PTORuntime* rt);

#ifdef __cplusplus
}
#endif

#endif // A2A3_ORCHESTRATION_H
