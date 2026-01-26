/**
 * PTO Runtime - Ascend A2/A3 Host Layer
 * 
 * This module provides the host CPU interface for A2A3 execution:
 * - Runtime initialization and shutdown
 * - Memory allocation and data transfer to/from device
 * - Orchestration function invocation
 * - Worker thread management
 * 
 * The host layer runs entirely on the host CPU and communicates
 * with the NPU for task execution.
 */

#ifndef A2A3_HOST_H
#define A2A3_HOST_H

#include "../../pto_runtime_common.h"
#include "../orchestration/a2a3_orchestration.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Host Configuration
// =============================================================================

// Default configuration for A2A3 platform
#define A2A3_DEFAULT_VECTOR_WORKERS  48
#define A2A3_DEFAULT_CUBE_WORKERS    24

// Memory configuration (Ascend 910B)
#define A2A3_GLOBAL_MEMORY_GB        32
#define A2A3_L2_CACHE_MB             200
#define A2A3_L1_SIZE_KB              192
#define A2A3_L1_SIZE_BYTES           (A2A3_L1_SIZE_KB * 1024)

// Core configuration
#define A2A3_NUM_VECTOR_CORES        48
#define A2A3_NUM_CUBE_CORES          24

// =============================================================================
// Host API
// =============================================================================

/**
 * Enable A2A3 simulation mode on the runtime.
 * 
 * @param rt                 Runtime context
 * @param num_vector_workers Number of vector worker threads
 * @param num_cube_workers   Number of cube worker threads
 */
void pto_runtime_enable_a2a3_sim(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers);

/**
 * A2A3 Runtime Entry Point
 * 
 * This is the main entry point for running an orchestration function
 * on the A2A3 platform. It:
 * 1. Initializes runtime and workers
 * 2. Calls the orchestration function to build task graph
 * 3. Executes tasks using dual-queue scheduling
 * 4. Returns when all tasks complete
 * 
 * @param orch_func               Orchestration function
 * @param user_data               User data passed to orchestration
 * @param num_vector_workers      Number of vector workers (0 = default)
 * @param num_cube_workers        Number of cube workers (0 = default)
 * @param execution_task_threshold Pipelining threshold (0 = no pipelining)
 * @return 0 on success, -1 on failure
 */
int runtime_entry_a2a3(PTOOrchFunc orch_func, void* user_data, 
                       int num_vector_workers, int num_cube_workers,
                       int execution_task_threshold);

// =============================================================================
// Memory Management
// =============================================================================

/**
 * Allocate memory on the A2A3 device.
 * 
 * @param size_bytes  Size in bytes to allocate
 * @return Pointer to device memory, or NULL on failure
 */
void* a2a3_host_malloc(size_t size_bytes);

/**
 * Free memory on the A2A3 device.
 */
void a2a3_host_free(void* ptr);

/**
 * Copy data from host to device.
 * 
 * @param dst_device  Destination on device
 * @param src_host    Source on host
 * @param size_bytes  Number of bytes to copy
 * @return 0 on success, -1 on failure
 */
int a2a3_host_copy_to_device(void* dst_device, const void* src_host, size_t size_bytes);

/**
 * Copy data from device to host.
 */
int a2a3_host_copy_from_device(void* dst_host, const void* src_device, size_t size_bytes);

// =============================================================================
// Synchronization
// =============================================================================

/**
 * Wait for all pending operations to complete.
 */
void a2a3_host_synchronize(void);

// =============================================================================
// Device Query
// =============================================================================

typedef struct {
    const char* name;
    int num_vector_cores;
    int num_cube_cores;
    int64_t global_memory_bytes;
    int64_t l2_cache_bytes;
    int64_t l1_buffer_bytes;
    double compute_capability;  // TFLOPS
} A2A3DeviceInfo;

/**
 * Query device information.
 */
void a2a3_host_get_device_info(A2A3DeviceInfo* info);

/**
 * Print device information.
 */
void a2a3_host_print_device_info(void);

#ifdef __cplusplus
}
#endif

#endif // A2A3_HOST_H
