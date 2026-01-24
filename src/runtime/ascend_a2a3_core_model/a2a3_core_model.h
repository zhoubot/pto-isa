/**
 * Ascend A2/A3 Core Model - Cycle-Accurate Simulator
 * 
 * This module implements a cycle-accurate model of the Ascend A2/A3 NPU core
 * architecture for simulation purposes. The model executes scalar instructions
 * and synchronization primitives while tracking cycles for compute operations.
 * 
 * Architecture Overview:
 * 
 * CUBE CORE:
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
 * │  │  Scalar  │   │ MTE_GM2L1│   │ MTE_L12GM│   │ MTE_L0C  │             │
 * │  │   Unit   │   │   Pipe   │   │   Pipe   │   │   Pipe   │             │
 * │  └──────────┘   └──────────┘   └──────────┘   └──────────┘             │
 * │                                                                         │
 * │  ┌──────────────────────────────────────────────────────────────┐      │
 * │  │                      CUBE Unit                                │      │
 * │  │    (Matrix Multiply: supports MATMUL instructions)            │      │
 * │  └──────────────────────────────────────────────────────────────┘      │
 * │                                                                         │
 * │  Memory Hierarchy: GM <-> L1 <-> L0A/L0B/L0C                           │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * VECTOR CORE:
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  ┌──────────┐   ┌──────────┐   ┌──────────┐                             │
 * │  │  Scalar  │   │ MTE_GM2UB│   │ MTE_UB2GM│                             │
 * │  │   Unit   │   │   Pipe   │   │   Pipe   │                             │
 * │  └──────────┘   └──────────┘   └──────────┘                             │
 * │                                                                         │
 * │  ┌──────────────────────────────────────────────────────────────┐      │
 * │  │                     Vector Unit                               │      │
 * │  │    (Element-wise ops, reductions, activations)                │      │
 * │  └──────────────────────────────────────────────────────────────┘      │
 * │                                                                         │
 * │  Memory Hierarchy: GM <-> UB (Unified Buffer)                          │
 * └─────────────────────────────────────────────────────────────────────────┘
 * 
 * Synchronization Primitives:
 * - SET_FLAG: Signal completion on a pipe
 * - WAIT_FLAG: Wait for completion signal from another pipe
 * - PIPE_BARRIER: Synchronize all pipes
 * 
 * Simulation Behavior:
 * - Scalar instructions: Execute immediately, advance scalar cycle counter
 * - MTE instructions: Issue to MTE pipe, track completion cycle
 * - Vector/Cube instructions: Issue to compute pipe, track completion cycle
 * - SET/WAIT/BARRIER: Execute synchronization logic
 * - Actual compute results are NOT calculated (cycle tracking only)
 */

#ifndef A2A3_CORE_MODEL_H
#define A2A3_CORE_MODEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Configuration Constants
// =============================================================================

#define A2A3_MAX_PIPES          8       // Maximum number of parallel pipes
#define A2A3_MAX_FLAGS          16      // Maximum synchronization flags
#define A2A3_MAX_PENDING_OPS    64      // Maximum pending operations per pipe
#define A2A3_MAX_INSTR_NAME     64      // Max instruction name length

// Pipe IDs for Cube Core
typedef enum {
    CUBE_PIPE_SCALAR = 0,       // Scalar unit
    CUBE_PIPE_MTE_GM2L1 = 1,    // GM -> L1 data transfer
    CUBE_PIPE_MTE_L12GM = 2,    // L1 -> GM data transfer
    CUBE_PIPE_MTE_L0C = 3,      // L0C data movement
    CUBE_PIPE_CUBE = 4,         // Cube compute unit
    CUBE_PIPE_COUNT = 5
} CubePipeId;

// Pipe IDs for Vector Core
typedef enum {
    VEC_PIPE_SCALAR = 0,        // Scalar unit
    VEC_PIPE_MTE_GM2UB = 1,     // GM -> UB data transfer
    VEC_PIPE_MTE_UB2GM = 2,     // UB -> GM data transfer
    VEC_PIPE_VECTOR = 3,        // Vector compute unit
    VEC_PIPE_COUNT = 4
} VectorPipeId;

// Instruction categories
typedef enum {
    INSTR_CAT_SCALAR,           // Scalar arithmetic, control flow
    INSTR_CAT_MTE,              // Memory transfer engine
    INSTR_CAT_VECTOR,           // Vector operations
    INSTR_CAT_CUBE,             // Matrix multiply
    INSTR_CAT_SYNC,             // Synchronization (SET/WAIT/BARRIER)
    INSTR_CAT_CONTROL           // Control flow (FOR/IF/etc.)
} InstrCategory;

// Core type
typedef enum {
    CORE_TYPE_CUBE,
    CORE_TYPE_VECTOR
} CoreType;

// =============================================================================
// Cycle Cost Table
// =============================================================================

// MTE latencies (cycles)
#define MTE_GM2L1_LATENCY       100     // GM to L1 transfer base latency
#define MTE_L12GM_LATENCY       100     // L1 to GM transfer base latency
#define MTE_GM2UB_LATENCY       80      // GM to UB transfer base latency
#define MTE_UB2GM_LATENCY       80      // UB to GM transfer base latency
#define MTE_L0C_LATENCY         20      // L0C movement latency

// Compute latencies (cycles per tile)
#define CUBE_MATMUL_LATENCY     50      // Matrix multiply base latency
#define VEC_BINARY_LATENCY      10      // Binary ops (add, mul, etc.)
#define VEC_UNARY_LATENCY       10      // Unary ops (exp, sqrt, etc.)
#define VEC_REDUCE_LATENCY      20      // Reduction ops (rowsum, rowmax)
#define VEC_ACTIVATION_LATENCY  15      // Activations (relu, sigmoid, etc.)

// Scalar instruction latency
#define SCALAR_LATENCY          1       // Scalar arithmetic

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Pending operation in a pipe
 */
typedef struct {
    int64_t issue_cycle;        // Cycle when operation was issued
    int64_t complete_cycle;     // Cycle when operation completes
    char name[A2A3_MAX_INSTR_NAME];
    bool active;
} PendingOp;

/**
 * Execution pipe state
 */
typedef struct {
    int pipe_id;
    int64_t current_cycle;      // Current cycle for this pipe
    int64_t last_issue_cycle;   // Cycle of last issued instruction
    
    // Pending operations queue
    PendingOp pending[A2A3_MAX_PENDING_OPS];
    int pending_count;
    
    // Statistics
    int64_t total_ops;
    int64_t total_stall_cycles;
} Pipe;

/**
 * Synchronization flag
 */
typedef struct {
    bool signaled;              // Flag has been set
    int64_t signal_cycle;       // Cycle when flag was set
    int src_pipe;               // Pipe that sets the flag
    int dst_pipe;               // Pipe that waits on the flag
} SyncFlag;

/**
 * Core model state
 */
typedef struct {
    CoreType type;
    int core_id;
    
    // Pipes
    Pipe pipes[A2A3_MAX_PIPES];
    int num_pipes;
    
    // Synchronization flags
    SyncFlag flags[A2A3_MAX_FLAGS];
    int num_flags;
    
    // Global state
    int64_t global_cycle;       // Maximum cycle across all pipes
    int64_t scalar_cycle;       // Scalar unit cycle counter
    
    // Statistics
    int64_t total_instructions;
    int64_t total_mte_ops;
    int64_t total_compute_ops;
    int64_t total_sync_ops;
    
    // Trace output
    bool trace_enabled;
    FILE* trace_file;
} A2A3Core;

/**
 * Decoded instruction for simulation
 */
typedef struct {
    char name[A2A3_MAX_INSTR_NAME];
    InstrCategory category;
    int target_pipe;            // Target pipe for execution
    int64_t latency;            // Estimated latency in cycles
    
    // For sync instructions
    int flag_id;
    int src_pipe;
    int dst_pipe;
    
    // For MTE instructions
    int64_t transfer_size;      // Bytes to transfer
} A2A3Instruction;

// =============================================================================
// Core Lifecycle API
// =============================================================================

/**
 * Create a new core model
 * @param type CORE_TYPE_CUBE or CORE_TYPE_VECTOR
 * @param core_id Unique identifier for this core
 * @return Pointer to initialized core, or NULL on error
 */
A2A3Core* a2a3_core_create(CoreType type, int core_id);

/**
 * Destroy a core model and free resources
 */
void a2a3_core_destroy(A2A3Core* core);

/**
 * Reset core state for new simulation
 */
void a2a3_core_reset(A2A3Core* core);

// =============================================================================
// Instruction Execution API
// =============================================================================

/**
 * Execute a single instruction on the core
 * @param core Core model
 * @param instr Decoded instruction
 * @return Number of cycles taken (0 for pipelined ops)
 */
int64_t a2a3_core_execute(A2A3Core* core, const A2A3Instruction* instr);

/**
 * Execute a scalar instruction
 */
int64_t a2a3_core_exec_scalar(A2A3Core* core, const char* name);

/**
 * Issue an MTE instruction (non-blocking)
 * @param pipe_id Target MTE pipe
 * @param transfer_size Bytes to transfer
 * @return Issue cycle
 */
int64_t a2a3_core_issue_mte(A2A3Core* core, int pipe_id, 
                            const char* name, int64_t transfer_size);

/**
 * Issue a compute instruction (non-blocking)
 * @param latency Estimated compute latency
 * @return Issue cycle
 */
int64_t a2a3_core_issue_compute(A2A3Core* core, const char* name, int64_t latency);

// =============================================================================
// Synchronization API
// =============================================================================

/**
 * Execute SET_FLAG instruction
 * @param flag_id Flag to set
 * @param src_pipe Pipe that sets the flag
 */
void a2a3_core_set_flag(A2A3Core* core, int flag_id, int src_pipe);

/**
 * Execute WAIT_FLAG instruction
 * @param flag_id Flag to wait on
 * @param dst_pipe Pipe that waits
 * @return Cycles stalled waiting
 */
int64_t a2a3_core_wait_flag(A2A3Core* core, int flag_id, int dst_pipe);

/**
 * Execute PIPE_BARRIER instruction
 * Synchronizes all pipes to the maximum cycle
 * @return Global cycle after barrier
 */
int64_t a2a3_core_pipe_barrier(A2A3Core* core);

// =============================================================================
// Query API
// =============================================================================

/**
 * Get total cycles for core execution
 */
int64_t a2a3_core_get_total_cycles(const A2A3Core* core);

/**
 * Get cycle count for a specific pipe
 */
int64_t a2a3_core_get_pipe_cycles(const A2A3Core* core, int pipe_id);

/**
 * Drain all pending operations and get final cycle count
 */
int64_t a2a3_core_drain(A2A3Core* core);

// =============================================================================
// Tracing API
// =============================================================================

/**
 * Enable instruction tracing
 */
void a2a3_core_enable_trace(A2A3Core* core, FILE* trace_file);

/**
 * Disable instruction tracing
 */
void a2a3_core_disable_trace(A2A3Core* core);

/**
 * Print core statistics
 */
void a2a3_core_print_stats(const A2A3Core* core);

// =============================================================================
// Instruction Decode Helpers
// =============================================================================

/**
 * Decode instruction name to category and target pipe
 */
InstrCategory a2a3_decode_instr_category(const char* instr_name, CoreType core_type);

/**
 * Get target pipe for an instruction
 */
int a2a3_get_target_pipe(const char* instr_name, CoreType core_type);

/**
 * Estimate latency for an instruction
 */
int64_t a2a3_estimate_latency(const char* instr_name, int64_t data_size);

#ifdef __cplusplus
}
#endif

#endif // A2A3_CORE_MODEL_H
