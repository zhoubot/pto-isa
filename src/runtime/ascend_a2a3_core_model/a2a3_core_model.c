/**
 * Ascend A2/A3 Core Model - Implementation
 * 
 * Implements the cycle-accurate simulation of Ascend A2/A3 NPU cores.
 */

#include "a2a3_core_model.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Internal Helpers
// =============================================================================

static void init_pipe(Pipe* pipe, int pipe_id) {
    pipe->pipe_id = pipe_id;
    pipe->current_cycle = 0;
    pipe->last_issue_cycle = 0;
    pipe->pending_count = 0;
    pipe->total_ops = 0;
    pipe->total_stall_cycles = 0;
    memset(pipe->pending, 0, sizeof(pipe->pending));
}

static int64_t max_cycle(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

// Drain completed operations from a pipe and update current_cycle
static void drain_completed_ops(Pipe* pipe) {
    int64_t max_complete = pipe->current_cycle;
    
    for (int i = 0; i < pipe->pending_count; i++) {
        if (pipe->pending[i].active && pipe->pending[i].complete_cycle > max_complete) {
            max_complete = pipe->pending[i].complete_cycle;
        }
    }
    
    // Mark completed ops as inactive
    int new_count = 0;
    for (int i = 0; i < pipe->pending_count; i++) {
        if (pipe->pending[i].active && pipe->pending[i].complete_cycle <= max_complete) {
            pipe->pending[i].active = false;
        } else if (pipe->pending[i].active) {
            if (i != new_count) {
                pipe->pending[new_count] = pipe->pending[i];
            }
            new_count++;
        }
    }
    pipe->pending_count = new_count;
    pipe->current_cycle = max_complete;
}

// Issue an operation to a pipe
static int64_t issue_op_to_pipe(Pipe* pipe, const char* name, int64_t latency) {
    // Find earliest cycle we can issue (after last issue)
    int64_t issue_cycle = max_cycle(pipe->current_cycle, pipe->last_issue_cycle);
    int64_t complete_cycle = issue_cycle + latency;
    
    // Add to pending queue if not full
    if (pipe->pending_count < A2A3_MAX_PENDING_OPS) {
        PendingOp* op = &pipe->pending[pipe->pending_count++];
        op->issue_cycle = issue_cycle;
        op->complete_cycle = complete_cycle;
        strncpy(op->name, name, A2A3_MAX_INSTR_NAME - 1);
        op->name[A2A3_MAX_INSTR_NAME - 1] = '\0';
        op->active = true;
    }
    
    pipe->last_issue_cycle = issue_cycle;
    pipe->total_ops++;
    
    return issue_cycle;
}

// =============================================================================
// Core Lifecycle
// =============================================================================

A2A3Core* a2a3_core_create(CoreType type, int core_id) {
    A2A3Core* core = (A2A3Core*)calloc(1, sizeof(A2A3Core));
    if (!core) return NULL;
    
    core->type = type;
    core->core_id = core_id;
    
    // Initialize pipes based on core type
    if (type == CORE_TYPE_CUBE) {
        core->num_pipes = CUBE_PIPE_COUNT;
        for (int i = 0; i < CUBE_PIPE_COUNT; i++) {
            init_pipe(&core->pipes[i], i);
        }
    } else {
        core->num_pipes = VEC_PIPE_COUNT;
        for (int i = 0; i < VEC_PIPE_COUNT; i++) {
            init_pipe(&core->pipes[i], i);
        }
    }
    
    // Initialize sync flags
    core->num_flags = A2A3_MAX_FLAGS;
    for (int i = 0; i < A2A3_MAX_FLAGS; i++) {
        core->flags[i].signaled = false;
        core->flags[i].signal_cycle = 0;
    }
    
    core->global_cycle = 0;
    core->scalar_cycle = 0;
    core->trace_enabled = false;
    core->trace_file = NULL;
    
    return core;
}

void a2a3_core_destroy(A2A3Core* core) {
    if (core) {
        if (core->trace_file && core->trace_file != stdout) {
            fclose(core->trace_file);
        }
        free(core);
    }
}

void a2a3_core_reset(A2A3Core* core) {
    if (!core) return;
    
    for (int i = 0; i < core->num_pipes; i++) {
        init_pipe(&core->pipes[i], i);
    }
    
    for (int i = 0; i < A2A3_MAX_FLAGS; i++) {
        core->flags[i].signaled = false;
        core->flags[i].signal_cycle = 0;
    }
    
    core->global_cycle = 0;
    core->scalar_cycle = 0;
    core->total_instructions = 0;
    core->total_mte_ops = 0;
    core->total_compute_ops = 0;
    core->total_sync_ops = 0;
}

// =============================================================================
// Instruction Execution
// =============================================================================

int64_t a2a3_core_execute(A2A3Core* core, const A2A3Instruction* instr) {
    if (!core || !instr) return 0;
    
    core->total_instructions++;
    
    if (core->trace_enabled && core->trace_file) {
        fprintf(core->trace_file, "[Core %d] Cycle %lld: %s (cat=%d, pipe=%d, lat=%lld)\n",
                core->core_id, (long long)core->global_cycle, instr->name,
                instr->category, instr->target_pipe, (long long)instr->latency);
    }
    
    switch (instr->category) {
        case INSTR_CAT_SCALAR:
            return a2a3_core_exec_scalar(core, instr->name);
            
        case INSTR_CAT_MTE:
            core->total_mte_ops++;
            return a2a3_core_issue_mte(core, instr->target_pipe, 
                                       instr->name, instr->transfer_size);
            
        case INSTR_CAT_VECTOR:
        case INSTR_CAT_CUBE:
            core->total_compute_ops++;
            return a2a3_core_issue_compute(core, instr->name, instr->latency);
            
        case INSTR_CAT_SYNC:
            core->total_sync_ops++;
            if (strstr(instr->name, "SET_FLAG") || strstr(instr->name, "set_flag")) {
                a2a3_core_set_flag(core, instr->flag_id, instr->src_pipe);
                return 0;
            } else if (strstr(instr->name, "WAIT_FLAG") || strstr(instr->name, "wait_flag")) {
                return a2a3_core_wait_flag(core, instr->flag_id, instr->dst_pipe);
            } else if (strstr(instr->name, "BARRIER") || strstr(instr->name, "barrier")) {
                return a2a3_core_pipe_barrier(core);
            }
            break;
            
        case INSTR_CAT_CONTROL:
            // Control flow instructions (FOR, IF, etc.) don't consume cycles
            // The scalar unit handles the iteration/branching
            return a2a3_core_exec_scalar(core, instr->name);
    }
    
    return 0;
}

int64_t a2a3_core_exec_scalar(A2A3Core* core, const char* name) {
    if (!core) return 0;
    
    // Scalar instructions execute immediately on scalar unit
    core->scalar_cycle += SCALAR_LATENCY;
    
    // Update pipe 0 (scalar pipe)
    core->pipes[0].current_cycle = core->scalar_cycle;
    
    // Update global cycle
    core->global_cycle = max_cycle(core->global_cycle, core->scalar_cycle);
    
    return SCALAR_LATENCY;
}

int64_t a2a3_core_issue_mte(A2A3Core* core, int pipe_id, 
                            const char* name, int64_t transfer_size) {
    if (!core || pipe_id < 0 || pipe_id >= core->num_pipes) return 0;
    
    Pipe* pipe = &core->pipes[pipe_id];
    
    // Calculate latency based on transfer size
    int64_t base_latency;
    if (core->type == CORE_TYPE_CUBE) {
        if (pipe_id == CUBE_PIPE_MTE_GM2L1) {
            base_latency = MTE_GM2L1_LATENCY;
        } else if (pipe_id == CUBE_PIPE_MTE_L12GM) {
            base_latency = MTE_L12GM_LATENCY;
        } else {
            base_latency = MTE_L0C_LATENCY;
        }
    } else {
        if (pipe_id == VEC_PIPE_MTE_GM2UB) {
            base_latency = MTE_GM2UB_LATENCY;
        } else {
            base_latency = MTE_UB2GM_LATENCY;
        }
    }
    
    // Scale latency by transfer size (simplified model)
    int64_t latency = base_latency + (transfer_size / 256);  // 256 bytes per cycle
    
    // Issue to pipe
    int64_t issue_cycle = issue_op_to_pipe(pipe, name, latency);
    
    // MTE doesn't block scalar, but we track for synchronization
    return issue_cycle;
}

int64_t a2a3_core_issue_compute(A2A3Core* core, const char* name, int64_t latency) {
    if (!core) return 0;
    
    // Get compute pipe based on core type
    int pipe_id;
    if (core->type == CORE_TYPE_CUBE) {
        pipe_id = CUBE_PIPE_CUBE;
    } else {
        pipe_id = VEC_PIPE_VECTOR;
    }
    
    Pipe* pipe = &core->pipes[pipe_id];
    
    // Issue to compute pipe
    int64_t issue_cycle = issue_op_to_pipe(pipe, name, latency);
    
    return issue_cycle;
}

// =============================================================================
// Synchronization
// =============================================================================

void a2a3_core_set_flag(A2A3Core* core, int flag_id, int src_pipe) {
    if (!core || flag_id < 0 || flag_id >= A2A3_MAX_FLAGS) return;
    if (src_pipe < 0 || src_pipe >= core->num_pipes) return;
    
    // Drain the source pipe to get its completion cycle
    drain_completed_ops(&core->pipes[src_pipe]);
    
    SyncFlag* flag = &core->flags[flag_id];
    flag->signaled = true;
    flag->signal_cycle = core->pipes[src_pipe].current_cycle;
    flag->src_pipe = src_pipe;
    
    if (core->trace_enabled && core->trace_file) {
        fprintf(core->trace_file, "[Core %d] SET_FLAG %d from pipe %d at cycle %lld\n",
                core->core_id, flag_id, src_pipe, (long long)flag->signal_cycle);
    }
}

int64_t a2a3_core_wait_flag(A2A3Core* core, int flag_id, int dst_pipe) {
    if (!core || flag_id < 0 || flag_id >= A2A3_MAX_FLAGS) return 0;
    if (dst_pipe < 0 || dst_pipe >= core->num_pipes) return 0;
    
    SyncFlag* flag = &core->flags[flag_id];
    Pipe* pipe = &core->pipes[dst_pipe];
    
    int64_t stall_cycles = 0;
    
    if (flag->signaled) {
        // If flag is signaled, advance pipe to at least the signal cycle
        if (flag->signal_cycle > pipe->current_cycle) {
            stall_cycles = flag->signal_cycle - pipe->current_cycle;
            pipe->current_cycle = flag->signal_cycle;
            pipe->total_stall_cycles += stall_cycles;
        }
    } else {
        // Flag not yet signaled - this shouldn't happen in correct code
        // but we handle it by waiting indefinitely (error case)
        fprintf(stderr, "[Core %d] WARNING: WAIT_FLAG %d not signaled!\n",
                core->core_id, flag_id);
    }
    
    // Clear the flag after waiting
    flag->signaled = false;
    
    if (core->trace_enabled && core->trace_file) {
        fprintf(core->trace_file, "[Core %d] WAIT_FLAG %d on pipe %d, stalled %lld cycles\n",
                core->core_id, flag_id, dst_pipe, (long long)stall_cycles);
    }
    
    return stall_cycles;
}

int64_t a2a3_core_pipe_barrier(A2A3Core* core) {
    if (!core) return 0;
    
    // Drain all pipes
    for (int i = 0; i < core->num_pipes; i++) {
        drain_completed_ops(&core->pipes[i]);
    }
    
    // Find maximum cycle across all pipes
    int64_t max_cycle_val = 0;
    for (int i = 0; i < core->num_pipes; i++) {
        if (core->pipes[i].current_cycle > max_cycle_val) {
            max_cycle_val = core->pipes[i].current_cycle;
        }
    }
    
    // Synchronize all pipes to max cycle
    int64_t total_stall = 0;
    for (int i = 0; i < core->num_pipes; i++) {
        int64_t stall = max_cycle_val - core->pipes[i].current_cycle;
        core->pipes[i].current_cycle = max_cycle_val;
        core->pipes[i].total_stall_cycles += stall;
        total_stall += stall;
    }
    
    // Update global cycle
    core->global_cycle = max_cycle_val;
    core->scalar_cycle = max_cycle_val;
    
    if (core->trace_enabled && core->trace_file) {
        fprintf(core->trace_file, "[Core %d] PIPE_BARRIER: all pipes synced to cycle %lld\n",
                core->core_id, (long long)max_cycle_val);
    }
    
    return core->global_cycle;
}

// =============================================================================
// Query API
// =============================================================================

int64_t a2a3_core_get_total_cycles(const A2A3Core* core) {
    if (!core) return 0;
    return core->global_cycle;
}

int64_t a2a3_core_get_pipe_cycles(const A2A3Core* core, int pipe_id) {
    if (!core || pipe_id < 0 || pipe_id >= core->num_pipes) return 0;
    return core->pipes[pipe_id].current_cycle;
}

int64_t a2a3_core_drain(A2A3Core* core) {
    if (!core) return 0;
    
    // Drain all pipes and find maximum cycle
    int64_t max_cycle_val = 0;
    for (int i = 0; i < core->num_pipes; i++) {
        drain_completed_ops(&core->pipes[i]);
        if (core->pipes[i].current_cycle > max_cycle_val) {
            max_cycle_val = core->pipes[i].current_cycle;
        }
    }
    
    core->global_cycle = max_cycle_val;
    return max_cycle_val;
}

// =============================================================================
// Tracing
// =============================================================================

void a2a3_core_enable_trace(A2A3Core* core, FILE* trace_file) {
    if (!core) return;
    core->trace_enabled = true;
    core->trace_file = trace_file ? trace_file : stdout;
}

void a2a3_core_disable_trace(A2A3Core* core) {
    if (!core) return;
    core->trace_enabled = false;
}

void a2a3_core_print_stats(const A2A3Core* core) {
    if (!core) return;
    
    const char* type_str = (core->type == CORE_TYPE_CUBE) ? "CUBE" : "VECTOR";
    
    printf("\n=== A2A3 %s Core %d Statistics ===\n", type_str, core->core_id);
    printf("Total cycles: %lld\n", (long long)core->global_cycle);
    printf("Total instructions: %lld\n", (long long)core->total_instructions);
    printf("  MTE ops: %lld\n", (long long)core->total_mte_ops);
    printf("  Compute ops: %lld\n", (long long)core->total_compute_ops);
    printf("  Sync ops: %lld\n", (long long)core->total_sync_ops);
    
    printf("\nPipe Statistics:\n");
    for (int i = 0; i < core->num_pipes; i++) {
        const Pipe* pipe = &core->pipes[i];
        if (pipe->total_ops > 0) {
            printf("  Pipe %d: %lld ops, %lld cycles, %lld stall cycles\n",
                   i, (long long)pipe->total_ops, (long long)pipe->current_cycle,
                   (long long)pipe->total_stall_cycles);
        }
    }
    printf("\n");
}

// =============================================================================
// Instruction Decode Helpers
// =============================================================================

InstrCategory a2a3_decode_instr_category(const char* instr_name, CoreType core_type) {
    if (!instr_name) return INSTR_CAT_SCALAR;
    
    // Synchronization
    if (strstr(instr_name, "SET_FLAG") || strstr(instr_name, "set_flag") ||
        strstr(instr_name, "WAIT_FLAG") || strstr(instr_name, "wait_flag") ||
        strstr(instr_name, "BARRIER") || strstr(instr_name, "barrier") ||
        strstr(instr_name, "pipe_barrier")) {
        return INSTR_CAT_SYNC;
    }
    
    // Control flow
    if (strstr(instr_name, "for") || strstr(instr_name, "FOR") ||
        strstr(instr_name, "if") || strstr(instr_name, "IF") ||
        strstr(instr_name, "else") || strstr(instr_name, "ELSE") ||
        strstr(instr_name, "endif") || strstr(instr_name, "ENDIF") ||
        strstr(instr_name, "endfor") || strstr(instr_name, "ENDFOR")) {
        return INSTR_CAT_CONTROL;
    }
    
    // Memory transfers
    if (strstr(instr_name, "DataCopy") || strstr(instr_name, "LoadData") ||
        strstr(instr_name, "StoreData") || strstr(instr_name, "GM2L1") ||
        strstr(instr_name, "L12GM") || strstr(instr_name, "GM2UB") ||
        strstr(instr_name, "UB2GM") || strstr(instr_name, "TLOAD") ||
        strstr(instr_name, "TSTORE")) {
        return INSTR_CAT_MTE;
    }
    
    // Matrix multiply (Cube)
    if (strstr(instr_name, "Matmul") || strstr(instr_name, "MATMUL") ||
        strstr(instr_name, "matmul") || strstr(instr_name, "TMATMUL") ||
        strstr(instr_name, "Mmad") || strstr(instr_name, "MMAD")) {
        return INSTR_CAT_CUBE;
    }
    
    // Vector operations
    if (strstr(instr_name, "Add") || strstr(instr_name, "Sub") ||
        strstr(instr_name, "Mul") || strstr(instr_name, "Div") ||
        strstr(instr_name, "Exp") || strstr(instr_name, "Ln") ||
        strstr(instr_name, "Sqrt") || strstr(instr_name, "Rsqrt") ||
        strstr(instr_name, "Relu") || strstr(instr_name, "Sigmoid") ||
        strstr(instr_name, "Tanh") || strstr(instr_name, "Gelu") ||
        strstr(instr_name, "Swish") || strstr(instr_name, "Silu") ||
        strstr(instr_name, "ReduceSum") || strstr(instr_name, "ReduceMax") ||
        strstr(instr_name, "Broadcast") || strstr(instr_name, "Duplicate") ||
        strstr(instr_name, "TADD") || strstr(instr_name, "TSUB") ||
        strstr(instr_name, "TMUL") || strstr(instr_name, "TDIV")) {
        return INSTR_CAT_VECTOR;
    }
    
    // Default to scalar
    return INSTR_CAT_SCALAR;
}

int a2a3_get_target_pipe(const char* instr_name, CoreType core_type) {
    InstrCategory cat = a2a3_decode_instr_category(instr_name, core_type);
    
    if (core_type == CORE_TYPE_CUBE) {
        switch (cat) {
            case INSTR_CAT_SCALAR:
            case INSTR_CAT_CONTROL:
            case INSTR_CAT_SYNC:
                return CUBE_PIPE_SCALAR;
            case INSTR_CAT_MTE:
                // Determine which MTE pipe based on instruction
                if (strstr(instr_name, "GM2L1") || strstr(instr_name, "LoadData") ||
                    strstr(instr_name, "TLOAD")) {
                    return CUBE_PIPE_MTE_GM2L1;
                } else if (strstr(instr_name, "L12GM") || strstr(instr_name, "StoreData") ||
                           strstr(instr_name, "TSTORE")) {
                    return CUBE_PIPE_MTE_L12GM;
                } else {
                    return CUBE_PIPE_MTE_L0C;
                }
            case INSTR_CAT_CUBE:
                return CUBE_PIPE_CUBE;
            case INSTR_CAT_VECTOR:
                // Vector ops on cube core go through scalar (shouldn't happen much)
                return CUBE_PIPE_SCALAR;
        }
    } else {
        // Vector core
        switch (cat) {
            case INSTR_CAT_SCALAR:
            case INSTR_CAT_CONTROL:
            case INSTR_CAT_SYNC:
                return VEC_PIPE_SCALAR;
            case INSTR_CAT_MTE:
                if (strstr(instr_name, "GM2UB") || strstr(instr_name, "LoadData") ||
                    strstr(instr_name, "TLOAD")) {
                    return VEC_PIPE_MTE_GM2UB;
                } else {
                    return VEC_PIPE_MTE_UB2GM;
                }
            case INSTR_CAT_VECTOR:
                return VEC_PIPE_VECTOR;
            case INSTR_CAT_CUBE:
                // Cube ops shouldn't be on vector core
                return VEC_PIPE_SCALAR;
        }
    }
    
    return 0;  // Default to scalar pipe
}

int64_t a2a3_estimate_latency(const char* instr_name, int64_t data_size) {
    InstrCategory cat = a2a3_decode_instr_category(instr_name, CORE_TYPE_VECTOR);
    
    switch (cat) {
        case INSTR_CAT_SCALAR:
        case INSTR_CAT_CONTROL:
            return SCALAR_LATENCY;
            
        case INSTR_CAT_SYNC:
            return 0;  // Sync doesn't have inherent latency
            
        case INSTR_CAT_MTE:
            // Scale by data size
            return MTE_GM2UB_LATENCY + (data_size / 256);
            
        case INSTR_CAT_CUBE:
            return CUBE_MATMUL_LATENCY;
            
        case INSTR_CAT_VECTOR:
            // Check for reduction or activation
            if (strstr(instr_name, "Reduce") || strstr(instr_name, "ROWSUM") ||
                strstr(instr_name, "ROWMAX")) {
                return VEC_REDUCE_LATENCY;
            } else if (strstr(instr_name, "Relu") || strstr(instr_name, "Sigmoid") ||
                       strstr(instr_name, "Tanh") || strstr(instr_name, "Gelu") ||
                       strstr(instr_name, "Swish") || strstr(instr_name, "Silu")) {
                return VEC_ACTIVATION_LATENCY;
            } else if (strstr(instr_name, "Exp") || strstr(instr_name, "Ln") ||
                       strstr(instr_name, "Sqrt") || strstr(instr_name, "Rsqrt")) {
                return VEC_UNARY_LATENCY;
            }
            return VEC_BINARY_LATENCY;
    }
    
    return SCALAR_LATENCY;
}
