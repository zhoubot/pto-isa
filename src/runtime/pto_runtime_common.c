/**
 * PTO Runtime System - Common Implementation (Platform Independent)
 * 
 * This file contains platform-independent implementations:
 * - Runtime initialization (basic structure)
 * - Task allocation and argument management
 * - TensorMap operations (dependency discovery)
 * - Record & Replay core logic
 * - Cycle trace recording
 * - Debug dump functions
 */

#include "pto_runtime_common.h"
#include <time.h>

// =============================================================================
// Global Variables
// =============================================================================

CycleTrace* pto_global_trace = NULL;

// =============================================================================
// Runtime Initialization (Platform Independent Parts)
// =============================================================================

void pto_runtime_init(PTORuntime* rt) {
    if (!rt) return;
    
    // Initialize task table (sliding window)
    memset(rt->pend_task, 0, sizeof(rt->pend_task));
    
    rt->next_task_id = 0;
    rt->active_task_count = 0;
    
    // Initialize sliding window tracking
    rt->window_oldest_pending = 0;
    rt->window_aborted = false;
    rt->runtime_mode = PTO_MODE_BENCHMARK_ONLY;  // Default: no window check
    
    // Initialize tensor map
    memset(rt->tensor_map, 0, sizeof(rt->tensor_map));
    
    // Allocate tensor map memory pool
    rt->tensormap_pool = (TensorMapEntry*)malloc(PTO_TENSORMAP_POOL_SIZE * sizeof(TensorMapEntry));
    rt->tensormap_pool_next = 0;
    
    // Initialize legacy ready queue
    memset(rt->ready_queue, 0, sizeof(rt->ready_queue));
    rt->ready_head = 0;
    rt->ready_tail = 0;
    rt->ready_count = 0;
    
    // Initialize dual ready queues (for a2a3_sim mode)
    memset(rt->vector_ready_queue, 0, sizeof(rt->vector_ready_queue));
    rt->vector_ready_head = 0;
    rt->vector_ready_tail = 0;
    rt->vector_ready_count = 0;
    
    memset(rt->cube_ready_queue, 0, sizeof(rt->cube_ready_queue));
    rt->cube_ready_head = 0;
    rt->cube_ready_tail = 0;
    rt->cube_ready_count = 0;
    
    // Initialize statistics
    rt->total_tasks_scheduled = 0;
    rt->total_tasks_completed = 0;
    
    // Initialize thread synchronization primitives
    pthread_mutex_init(&rt->queue_mutex, NULL);
    pthread_mutex_init(&rt->task_mutex, NULL);
    pthread_cond_init(&rt->queue_not_empty, NULL);
    pthread_cond_init(&rt->all_done, NULL);
    pthread_cond_init(&rt->vector_queue_not_empty, NULL);
    pthread_cond_init(&rt->cube_queue_not_empty, NULL);
    pthread_cond_init(&rt->window_not_full, NULL);
    
    // Initialize worker state
    rt->num_workers = 0;
    rt->num_vector_workers = 0;
    rt->num_cube_workers = 0;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    rt->execution_task_threshold = 0;
    rt->simulation_mode = false;
    rt->dual_queue_mode = false;
    memset(rt->workers, 0, sizeof(rt->workers));
    memset(rt->func_registry, 0, sizeof(rt->func_registry));
    
    DEBUG_PRINT("[PTO Runtime] Initialized (window_size=%d, tensormap_size=%d)\n",
           PTO_TASK_WINDOW_SIZE, PTO_TENSORMAP_SIZE);
}

void pto_runtime_shutdown(PTORuntime* rt) {
    if (!rt) return;
    
    // Cleanup core simulator (if used)
    pto_cleanup_core_sim();
    
    // Free tensor map memory pool
    if (rt->tensormap_pool) {
        free(rt->tensormap_pool);
        rt->tensormap_pool = NULL;
    }
    rt->tensormap_pool_next = 0;
    memset(rt->tensor_map, 0, sizeof(rt->tensor_map));
    
    // Destroy thread synchronization primitives
    pthread_mutex_destroy(&rt->queue_mutex);
    pthread_mutex_destroy(&rt->task_mutex);
    pthread_cond_destroy(&rt->queue_not_empty);
    pthread_cond_destroy(&rt->all_done);
    pthread_cond_destroy(&rt->vector_queue_not_empty);
    pthread_cond_destroy(&rt->cube_queue_not_empty);
    pthread_cond_destroy(&rt->window_not_full);
    
    DEBUG_PRINT("[PTO Runtime] Shutdown (scheduled=%lld, completed=%lld)\n",
           (long long)rt->total_tasks_scheduled,
           (long long)rt->total_tasks_completed);
}

void pto_runtime_set_mode(PTORuntime* rt, PTORuntimeMode mode) {
    if (!rt) return;
    rt->runtime_mode = mode;
    DEBUG_PRINT("[PTO Runtime] Mode set to %d\n", mode);
}

bool pto_runtime_was_aborted(PTORuntime* rt) {
    if (!rt) return false;
    return rt->window_aborted;
}

void pto_runtime_reset(PTORuntime* rt) {
    if (!rt) return;
    
    // Clear task table
    memset(rt->pend_task, 0, sizeof(rt->pend_task));
    
    rt->next_task_id = 0;
    rt->active_task_count = 0;
    rt->window_oldest_pending = 0;
    rt->window_aborted = false;
    
    // Clear tensor map (just reset hash table and pool index, entries are in pool)
    memset(rt->tensor_map, 0, sizeof(rt->tensor_map));
    rt->tensormap_pool_next = 0;
    
    // Reset ready queues
    rt->ready_head = 0;
    rt->ready_tail = 0;
    rt->ready_count = 0;
    rt->vector_ready_head = 0;
    rt->vector_ready_tail = 0;
    rt->vector_ready_count = 0;
    rt->cube_ready_head = 0;
    rt->cube_ready_tail = 0;
    rt->cube_ready_count = 0;
    
    // Reset statistics
    rt->total_tasks_scheduled = 0;
    rt->total_tasks_completed = 0;
    
    DEBUG_PRINT("[PTO Runtime] Reset complete\n");
}

void pto_runtime_stats(PTORuntime* rt) {
    printf("\n[PTO Runtime Statistics]\n");
    printf("  Total tasks scheduled: %lld\n", (long long)rt->total_tasks_scheduled);
    printf("  Total tasks completed: %lld\n", (long long)rt->total_tasks_completed);
    printf("  Active tasks:          %d\n", rt->active_task_count);
    printf("  Ready queue size:      %d\n", rt->ready_count);
    printf("\n");
}

// =============================================================================
// TensorMap Implementation (Platform Independent)
// 
// Optimization: Entries with producer_id < window_oldest_pending are considered
// "stale" and can be reused or skipped. This keeps the tensor map size bounded
// by the sliding window size rather than growing unboundedly.
// =============================================================================

uint32_t pto_tensormap_hash(TensorRegion* region) {
    // Fast hash with good distribution - no loops, minimal operations
    uint64_t ptr_val = (uint64_t)region->raw_tensor;
    
    // Combine all fields with shifts and XORs (very fast)
    uint64_t h = ptr_val;
    h ^= (uint64_t)region->row_offset * 0x9E3779B97F4A7C15ULL;  // Golden ratio constant
    h ^= (uint64_t)region->col_offset * 0xC6A4A7935BD1E995ULL;  // Another prime
    h ^= ((uint64_t)region->rows << 32) | (uint64_t)region->cols;
    
    // Final mix (from MurmurHash finalizer)
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    
    // Use bitwise AND for power-of-2 table size
    return (uint32_t)h & (PTO_TENSORMAP_SIZE - 1);
}

bool pto_region_match(TensorRegion* a, TensorRegion* b) {
    return a->raw_tensor == b->raw_tensor &&
           a->row_offset == b->row_offset &&
           a->col_offset == b->col_offset &&
           a->rows == b->rows &&
           a->cols == b->cols;
}

// Check if an entry is stale (producer task has already completed and left the window)
static inline bool pto_tensormap_entry_is_stale(PTORuntime* rt, TensorMapEntry* entry) {
    return entry->producer_id < rt->window_oldest_pending;
}

int32_t pto_tensormap_lookup(PTORuntime* rt, TensorRegion* region) {
    uint32_t hash = pto_tensormap_hash(region);
    TensorMapEntry* entry = rt->tensor_map[hash];
    
    while (entry) {
        if (pto_region_match(&entry->region, region)) {
            // Check if producer is stale (outside window)
            if (pto_tensormap_entry_is_stale(rt, entry)) {
                return -1;  // Producer no longer valid
            }
            return entry->producer_id;
        }
        entry = entry->next;
    }
    
    return -1; // Not found
}

void pto_tensormap_insert(PTORuntime* rt, TensorRegion* region, int32_t task_id) {
    uint32_t hash = pto_tensormap_hash(region);
    TensorMapEntry* entry = rt->tensor_map[hash];
    
    while (entry) {
        // Exact match - update in place
        if (pto_region_match(&entry->region, region)) {
            entry->producer_id = task_id;
            return;
        }
        
        // Stale entry - overwrite in place (no GC needed)
        if (pto_tensormap_entry_is_stale(rt, entry)) {
            entry->region = *region;
            entry->producer_id = task_id;
            return;
        }
        
        entry = entry->next;
    }
    
    // No reusable entry found - allocate from pool
    if (rt->tensormap_pool && rt->tensormap_pool_next < PTO_TENSORMAP_POOL_SIZE) {
        TensorMapEntry* new_entry = &rt->tensormap_pool[rt->tensormap_pool_next++];
        new_entry->region = *region;
        new_entry->producer_id = task_id;
        new_entry->next = rt->tensor_map[hash];
        rt->tensor_map[hash] = new_entry;
    } else {
        // Fallback to malloc if pool exhausted
        TensorMapEntry* new_entry = (TensorMapEntry*)malloc(sizeof(TensorMapEntry));
        if (!new_entry) {
            fprintf(stderr, "[PTO Runtime] ERROR: Failed to allocate TensorMapEntry\n");
            return;
        }
        new_entry->region = *region;
        new_entry->producer_id = task_id;
        new_entry->next = rt->tensor_map[hash];
        rt->tensor_map[hash] = new_entry;
    }
}

void pto_tensormap_clear(PTORuntime* rt) {
    // With memory pool, just clear the hash table pointers
    // Pool entries are reused when pool index is reset
    memset(rt->tensor_map, 0, sizeof(rt->tensor_map));
    rt->tensormap_pool_next = 0;
}

// Garbage collect - no longer needed
// Stale entries are reused in-place during insert
void pto_tensormap_gc(PTORuntime* rt) {
    (void)rt;  // No-op: stale entries overwritten during insert
}

// =============================================================================
// Task Allocation and Arguments (Platform Independent)
// =============================================================================

int32_t pto_task_alloc_impl(PTORuntime* rt, const char* func_name, void* func_ptr,
                            int32_t buffer_bytes, int32_t reuse_bytes, bool is_cube) {
    // Check if window is full
    int32_t tasks_in_flight = rt->next_task_id - rt->window_oldest_pending;
    
    if (tasks_in_flight >= PTO_TASK_WINDOW_SIZE) {
        // Window is full - behavior depends on runtime mode
        switch (rt->runtime_mode) {
            case PTO_MODE_BENCHMARK_ONLY:
                // Benchmark mode: simulate window advancement to enable TensorMap cleanup
                // Since no tasks actually complete, window_oldest_pending stays at 0,
                // causing TensorMap to grow unboundedly. By advancing it here, we allow
                // stale entries to be reclaimed, keeping TensorMap size bounded.
                rt->window_oldest_pending = rt->next_task_id - (PTO_TASK_WINDOW_SIZE / 2);
                break;
                
            case PTO_MODE_DUMP_GRAPH:
                // Dump/graph mode: abort orchestration
                if (!rt->window_aborted) {
                    rt->window_aborted = true;
                    fprintf(stderr, "[PTO Runtime] Window full (size=%d), aborting orchestration for dump/graph\n",
                            PTO_TASK_WINDOW_SIZE);
                }
                return -1;
                
            case PTO_MODE_EXECUTE:
            case PTO_MODE_SIMULATE:
                // Execute/simulate mode: wait for workers to complete tasks
                pthread_mutex_lock(&rt->task_mutex);
                while ((rt->next_task_id - rt->window_oldest_pending) >= PTO_TASK_WINDOW_SIZE) {
                    DEBUG_PRINT("[PTO Runtime] Window full, waiting... (oldest=%d, next=%d)\n",
                           rt->window_oldest_pending, rt->next_task_id);
                    pthread_cond_wait(&rt->window_not_full, &rt->task_mutex);
                }
                pthread_mutex_unlock(&rt->task_mutex);
                break;
        }
    }
    
    // Allocate task ID and get slot in window
    int32_t task_id = rt->next_task_id++;
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    // Initialize task
    task->task_id = task_id;
    task->func_name = func_name;
    task->func_ptr = func_ptr;
    task->cycle_func = NULL;  // Set via pto_task_set_cycle_func if needed
    task->num_args = 0;
    task->buffer_size_bytes = buffer_bytes;
    task->buffer_size_with_reuse = reuse_bytes;
    task->fanin = 0;
    task->fanout_count = 0;
    task->is_active = true;
    task->is_complete = false;
    task->is_cube = is_cube;
    task->earliest_start_cycle = 0;
    task->end_cycle = 0;
    
    // Clear fanout list
    memset(task->fanout, 0, sizeof(task->fanout));
    
    rt->active_task_count++;
    rt->total_tasks_scheduled++;
    
    DEBUG_PRINT("[PTO Runtime] Allocated task %d (slot %d): %s (buf=%d B, reuse=%d B, is_cube=%d)\n", 
           task_id, slot, func_name, buffer_bytes, reuse_bytes, is_cube);
    
    return task_id;
}

void pto_task_set_cycle_func(PTORuntime* rt, int32_t task_id, CycleCostFunc cycle_func) {
    if (!rt || task_id < 0 || task_id >= rt->next_task_id) return;
    rt->pend_task[PTO_TASK_SLOT(task_id)].cycle_func = cycle_func;
}

void pto_task_add_input(PTORuntime* rt, int32_t task_id,
                        void* tensor, int64_t row_off, int64_t col_off,
                        int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
    
    if (task->num_args >= PTO_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many arguments for task %d\n", task_id);
        return;
    }
    
    // Create tensor region
    TensorRegion region = {
        .raw_tensor = tensor,
        .row_offset = row_off,
        .col_offset = col_off,
        .rows = rows,
        .cols = cols
    };
    
    // Add argument
    TaskArg* arg = &task->args[task->num_args++];
    arg->region = region;
    arg->is_output = false;
    
    // Look up producer in TensorMap
    int32_t producer_id = pto_tensormap_lookup(rt, &region);
    
    if (producer_id >= 0 && producer_id != task_id) {
        // Found producer - add dependency (needs mutex for pipelined execution)
        pthread_mutex_lock(&rt->task_mutex);
        
        PendingTask* producer = &rt->pend_task[PTO_TASK_SLOT(producer_id)];
        
        // Check if producer is already complete (pipelined execution race condition)
        if (producer->is_complete) {
            // Producer already done - no need to add dependency
            pthread_mutex_unlock(&rt->task_mutex);
            DEBUG_PRINT("[PTO Runtime] Task %d: producer %d already complete, no dependency added\n",
                   task_id, producer_id);
        } else {
            // Add current task to producer's fanout
            if (producer->fanout_count < PTO_MAX_FANOUT) {
                producer->fanout[producer->fanout_count++] = task_id;
            } else {
                fprintf(stderr, "[PTO Runtime] WARNING: Fanout overflow for task %d\n", producer_id);
            }
            
            // Increment fanin (dependency count)
            task->fanin++;
            
            pthread_mutex_unlock(&rt->task_mutex);
            DEBUG_PRINT("[PTO Runtime] Task %d depends on task %d (tensor=%p, offset=[%lld,%lld])\n",
                   task_id, producer_id, tensor, (long long)row_off, (long long)col_off);
        }
    } else {
        DEBUG_PRINT("[PTO Runtime] Task %d input (tensor=%p, offset=[%lld,%lld]) - no producer\n",
               task_id, tensor, (long long)row_off, (long long)col_off);
    }
}

void pto_task_add_output(PTORuntime* rt, int32_t task_id,
                         void* tensor, int64_t row_off, int64_t col_off,
                         int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
    
    if (task->num_args >= PTO_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many arguments for task %d\n", task_id);
        return;
    }
    
    // Create tensor region
    TensorRegion region = {
        .raw_tensor = tensor,
        .row_offset = row_off,
        .col_offset = col_off,
        .rows = rows,
        .cols = cols
    };
    
    // Add argument
    TaskArg* arg = &task->args[task->num_args++];
    arg->region = region;
    arg->is_output = true;
    
    // Register in TensorMap (this task produces this region)
    pto_tensormap_insert(rt, &region, task_id);
    
    DEBUG_PRINT("[PTO Runtime] Task %d output (tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld])\n",
           task_id, tensor, (long long)row_off, (long long)col_off,
           (long long)rows, (long long)cols);
}

// =============================================================================
// Cycle Trace Recording Implementation (Platform Independent)
// =============================================================================

void pto_trace_init(int32_t num_workers) {
    if (pto_global_trace) {
        free(pto_global_trace);
    }
    pto_global_trace = (CycleTrace*)calloc(1, sizeof(CycleTrace));
    if (!pto_global_trace) return;
    
    pto_global_trace->count = 0;
    pto_global_trace->num_workers = num_workers > 0 ? num_workers : 1;
    pto_global_trace->num_vector_workers = 0;
    pto_global_trace->num_cube_workers = 0;
    pto_global_trace->enabled = true;
    
    // Initialize per-worker cycle counters
    for (int i = 0; i < PTO_MAX_WORKERS; i++) {
        pto_global_trace->per_worker_cycle[i] = 0;
    }
}

void pto_trace_init_dual(int32_t num_vector_workers, int32_t num_cube_workers) {
    int32_t total = num_vector_workers + num_cube_workers;
    pto_trace_init(total);
    if (pto_global_trace) {
        pto_global_trace->num_vector_workers = num_vector_workers;
        pto_global_trace->num_cube_workers = num_cube_workers;
    }
}

void pto_trace_record(int32_t worker_id, const char* func_name, int64_t cycle_cost) {
    if (!pto_global_trace || !pto_global_trace->enabled) return;
    if (pto_global_trace->count >= PTO_MAX_TRACE_ENTRIES) return;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return;
    
    int idx = pto_global_trace->count++;
    CycleTraceEntry* entry = &pto_global_trace->entries[idx];
    
    // Copy function name
    strncpy(entry->func_name, func_name ? func_name : "unknown", PTO_MAX_FUNC_NAME_LEN - 1);
    entry->func_name[PTO_MAX_FUNC_NAME_LEN - 1] = '\0';
    
    entry->worker_id = worker_id;
    entry->start_cycle = pto_global_trace->per_worker_cycle[worker_id];
    entry->end_cycle = entry->start_cycle + cycle_cost;
    
    // Update worker cycle counter
    pto_global_trace->per_worker_cycle[worker_id] = entry->end_cycle;
}

void pto_trace_record_with_time(int32_t worker_id, const char* func_name, 
                                 int64_t start_cycle, int64_t end_cycle) {
    if (!pto_global_trace || !pto_global_trace->enabled) return;
    if (pto_global_trace->count >= PTO_MAX_TRACE_ENTRIES) return;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return;
    
    int idx = pto_global_trace->count++;
    CycleTraceEntry* entry = &pto_global_trace->entries[idx];
    
    // Copy function name
    strncpy(entry->func_name, func_name ? func_name : "unknown", PTO_MAX_FUNC_NAME_LEN - 1);
    entry->func_name[PTO_MAX_FUNC_NAME_LEN - 1] = '\0';
    
    entry->worker_id = worker_id;
    entry->start_cycle = start_cycle;
    entry->end_cycle = end_cycle;
    
    // Update worker cycle counter to the end of this task
    pto_global_trace->per_worker_cycle[worker_id] = end_cycle;
}

int64_t pto_trace_get_cycle(int32_t worker_id) {
    if (!pto_global_trace) return 0;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return 0;
    return pto_global_trace->per_worker_cycle[worker_id];
}

void pto_trace_cleanup(void) {
    if (pto_global_trace) {
        free(pto_global_trace);
        pto_global_trace = NULL;
    }
}

char* pto_trace_to_chrome_json(void) {
    if (!pto_global_trace) return NULL;
    
    int32_t num_vec = pto_global_trace->num_vector_workers;
    int32_t num_cube = pto_global_trace->num_cube_workers;
    bool dual_mode = (num_vec > 0 && num_cube > 0);
    
    // Estimate output size (generous allocation)
    // Add extra space for thread name metadata if dual mode
    size_t metadata_size = dual_mode ? (num_vec + num_cube) * 128 : 0;
    size_t buf_size = 2048 + pto_global_trace->count * 256 + metadata_size;
    char* buf = (char*)malloc(buf_size);
    if (!buf) return NULL;
    
    char* ptr = buf;
    ptr += sprintf(ptr, "{\n  \"traceEvents\": [\n");
    
    // Add thread name metadata events for dual-queue mode
    if (dual_mode) {
        // Vector workers: pid=0, tid=0 to num_vec-1
        for (int i = 0; i < num_vec; i++) {
            ptr += sprintf(ptr, "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 0, \"tid\": %d, "
                           "\"args\": {\"name\": \"Vector-%d\"}},\n", i, i);
        }
        // Cube workers: pid=1, tid=0 to num_cube-1
        for (int i = 0; i < num_cube; i++) {
            ptr += sprintf(ptr, "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": %d, "
                           "\"args\": {\"name\": \"Cube-%d\"}},\n", i, i);
        }
        // Process name metadata
        ptr += sprintf(ptr, "    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 0, "
                       "\"args\": {\"name\": \"Vector Workers (%d)\"}},\n", num_vec);
        ptr += sprintf(ptr, "    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 1, "
                       "\"args\": {\"name\": \"Cube Workers (%d)\"}},\n", num_cube);
    }
    
    for (int i = 0; i < pto_global_trace->count; i++) {
        CycleTraceEntry* e = &pto_global_trace->entries[i];
        int64_t duration = e->end_cycle - e->start_cycle;
        
        // Determine pid and tid based on worker type
        int pid = 0;
        int tid = e->worker_id;
        
        if (dual_mode && e->worker_id >= num_vec) {
            // Cube worker (worker_id >= num_vector_workers)
            pid = 1;
            tid = e->worker_id - num_vec;
        }
        
        // Chrome Tracing format (duration event)
        ptr += sprintf(ptr, "    {\"name\": \"%s\", \"cat\": \"task\", \"ph\": \"X\", "
                       "\"ts\": %lld, \"dur\": %lld, \"pid\": %d, \"tid\": %d}%s\n",
                       e->func_name,
                       (long long)(e->start_cycle),     // timestamp in microseconds (we use cycles)
                       (long long)duration,              // duration
                       pid, tid,
                       (i < pto_global_trace->count - 1) ? "," : "");
    }
    
    ptr += sprintf(ptr, "  ],\n");
    ptr += sprintf(ptr, "  \"displayTimeUnit\": \"ns\",\n");
    ptr += sprintf(ptr, "  \"metadata\": {\n");
    ptr += sprintf(ptr, "    \"num_workers\": %d,\n", pto_global_trace->num_workers);
    if (dual_mode) {
        ptr += sprintf(ptr, "    \"num_vector_workers\": %d,\n", num_vec);
        ptr += sprintf(ptr, "    \"num_cube_workers\": %d,\n", num_cube);
    }
    ptr += sprintf(ptr, "    \"total_entries\": %d\n", pto_global_trace->count);
    ptr += sprintf(ptr, "  }\n");
    ptr += sprintf(ptr, "}\n");
    
    return buf;
}

void pto_trace_write_json(const char* filename) {
    char* json = pto_trace_to_chrome_json();
    if (!json) {
        fprintf(stderr, "Error: Failed to generate trace JSON\n");
        return;
    }
    
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Failed to open %s for writing\n", filename);
        free(json);
        return;
    }
    
    fputs(json, f);
    fclose(f);
    free(json);
    
    printf("Trace written to: %s\n", filename);
    printf("  Open in Chrome: chrome://tracing and load the file\n");
}

void pto_trace_print_summary(void) {
    if (!pto_global_trace) {
        printf("Trace: not initialized\n");
        return;
    }
    
    printf("\n=== Cycle Trace Summary ===\n");
    printf("Total entries: %d\n", pto_global_trace->count);
    printf("Num workers: %d\n", pto_global_trace->num_workers);
    
    // Per-worker statistics
    int64_t max_cycle = 0;
    for (int w = 0; w < pto_global_trace->num_workers; w++) {
        int64_t cycle = pto_global_trace->per_worker_cycle[w];
        printf("  Worker %d: %lld cycles\n", w, (long long)cycle);
        if (cycle > max_cycle) max_cycle = cycle;
    }
    printf("Max cycle (makespan): %lld\n", (long long)max_cycle);
    
    // Function breakdown
    printf("\nFunction breakdown:\n");
    
    // Simple aggregation (could be made more efficient with hash table)
    typedef struct { char name[PTO_MAX_FUNC_NAME_LEN]; int64_t total_cycles; int count; } FuncStats;
    FuncStats stats[100];
    int num_stats = 0;
    
    for (int i = 0; i < pto_global_trace->count; i++) {
        CycleTraceEntry* e = &pto_global_trace->entries[i];
        int64_t dur = e->end_cycle - e->start_cycle;
        
        // Find or create entry
        int found = -1;
        for (int s = 0; s < num_stats; s++) {
            if (strcmp(stats[s].name, e->func_name) == 0) {
                found = s;
                break;
            }
        }
        
        if (found >= 0) {
            stats[found].total_cycles += dur;
            stats[found].count++;
        } else if (num_stats < 100) {
            strncpy(stats[num_stats].name, e->func_name, PTO_MAX_FUNC_NAME_LEN - 1);
            stats[num_stats].total_cycles = dur;
            stats[num_stats].count = 1;
            num_stats++;
        }
    }
    
    for (int s = 0; s < num_stats; s++) {
        printf("  %s: %lld cycles (%d calls)\n", 
               stats[s].name, (long long)stats[s].total_cycles, stats[s].count);
    }
    printf("===========================\n\n");
}

// =============================================================================
// Debug Dump Implementation (Platform Independent)
// =============================================================================

int pto_runtime_dump(PTORuntime* rt, const char* filename) {
    if (!rt || !filename) return -1;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "[PTO Runtime] ERROR: Cannot open file %s for writing\n", filename);
        return -1;
    }
    
    // Header
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "PTO RUNTIME DUMP\n");
    fprintf(fp, "================================================================================\n\n");
    
    // Summary statistics
    fprintf(fp, "SUMMARY\n");
    fprintf(fp, "--------------------------------------------------------------------------------\n");
    fprintf(fp, "  Total tasks scheduled:  %lld\n", (long long)rt->total_tasks_scheduled);
    fprintf(fp, "  Total tasks completed:  %lld\n", (long long)rt->total_tasks_completed);
    fprintf(fp, "  Active task count:      %d\n", rt->active_task_count);
    fprintf(fp, "  Next task ID:           %d\n", rt->next_task_id);
    fprintf(fp, "  Ready queue size:       %d\n", rt->ready_count);
    fprintf(fp, "\n");
    
    // Task Table - only dump tasks within current window
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "TASK TABLE (sliding window, size=%d)\n", PTO_TASK_WINDOW_SIZE);
    fprintf(fp, "================================================================================\n\n");
    
    // Determine dump range (limited by window)
    int32_t dump_start = rt->window_oldest_pending;
    int32_t dump_end = rt->next_task_id;
    int32_t dump_count = dump_end - dump_start;
    if (dump_count > PTO_TASK_WINDOW_SIZE) {
        dump_count = PTO_TASK_WINDOW_SIZE;
        dump_start = dump_end - PTO_TASK_WINDOW_SIZE;
    }
    
    fprintf(fp, "  Window: tasks %d to %d (%d tasks)\n\n", dump_start, dump_end - 1, dump_count);
    
    for (int32_t i = dump_start; i < dump_end; i++) {
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(i)];
        
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "TASK %d (slot %d)\n", task->task_id, PTO_TASK_SLOT(i));
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "  Function:     %s\n", task->func_name ? task->func_name : "(null)");
        fprintf(fp, "  Func Ptr:     %p\n", task->func_ptr);
        fprintf(fp, "  Is Active:    %s\n", task->is_active ? "true" : "false");
        fprintf(fp, "  Is Complete:  %s\n", task->is_complete ? "true" : "false");
        fprintf(fp, "\n");
        
        // Buffer size estimation
        fprintf(fp, "  BUFFER SIZE (InCore Tile Buffers)\n");
        fprintf(fp, "  ----------------------------------\n");
        fprintf(fp, "    without_reuse = %d bytes (%.2f KB)\n", 
                task->buffer_size_bytes, task->buffer_size_bytes / 1024.0);
        fprintf(fp, "    with_reuse    = %d bytes (%.2f KB)\n", 
                task->buffer_size_with_reuse, task->buffer_size_with_reuse / 1024.0);
        if (task->buffer_size_bytes > 0) {
            int savings = task->buffer_size_bytes - task->buffer_size_with_reuse;
            float pct = 100.0 * savings / task->buffer_size_bytes;
            fprintf(fp, "    savings       = %d bytes (%.1f%%)\n", savings, pct);
        }
        fprintf(fp, "\n");
        
        // Fanin counter
        fprintf(fp, "  FANIN COUNTER\n");
        fprintf(fp, "  -------------\n");
        fprintf(fp, "    fanin = %d\n", task->fanin);
        fprintf(fp, "\n");
        
        // Fanout list
        fprintf(fp, "  FANOUT LIST (consumers that depend on this task)\n");
        fprintf(fp, "  ------------------------------------------------\n");
        fprintf(fp, "    fanout_count = %d\n", task->fanout_count);
        if (task->fanout_count > 0) {
            fprintf(fp, "    fanout[] = [");
            for (int j = 0; j < task->fanout_count; j++) {
                fprintf(fp, "%d", task->fanout[j]);
                if (j < task->fanout_count - 1) fprintf(fp, ", ");
            }
            fprintf(fp, "]\n");
            
            // Detailed fanout info
            fprintf(fp, "    Consumers:\n");
            for (int j = 0; j < task->fanout_count; j++) {
                int32_t consumer_id = task->fanout[j];
                PendingTask* consumer = &rt->pend_task[PTO_TASK_SLOT(consumer_id)];
                fprintf(fp, "      -> Task %d (%s)\n", consumer_id, 
                        consumer->func_name ? consumer->func_name : "(null)");
            }
        } else {
            fprintf(fp, "    fanout[] = [] (no consumers)\n");
        }
        fprintf(fp, "\n");
        
        // Arguments
        fprintf(fp, "  ARGUMENTS (num_args = %d)\n", task->num_args);
        fprintf(fp, "  -------------------------\n");
        for (int j = 0; j < task->num_args; j++) {
            TaskArg* arg = &task->args[j];
            fprintf(fp, "    [%d] %s:\n", j, arg->is_output ? "OUTPUT" : "INPUT");
            fprintf(fp, "        tensor:     %p\n", arg->region.raw_tensor);
            fprintf(fp, "        row_offset: %lld\n", (long long)arg->region.row_offset);
            fprintf(fp, "        col_offset: %lld\n", (long long)arg->region.col_offset);
            fprintf(fp, "        rows:       %lld\n", (long long)arg->region.rows);
            fprintf(fp, "        cols:       %lld\n", (long long)arg->region.cols);
        }
        fprintf(fp, "\n");
    }
    
    // Ready Queue
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "READY QUEUE\n");
    fprintf(fp, "================================================================================\n\n");
    fprintf(fp, "  Head:  %d\n", rt->ready_head);
    fprintf(fp, "  Tail:  %d\n", rt->ready_tail);
    fprintf(fp, "  Count: %d\n", rt->ready_count);
    if (rt->ready_count > 0) {
        fprintf(fp, "  Queue: [");
        int idx = rt->ready_head;
        for (int i = 0; i < rt->ready_count; i++) {
            fprintf(fp, "%d", rt->ready_queue[idx]);
            if (i < rt->ready_count - 1) fprintf(fp, ", ");
            idx = (idx + 1) % PTO_MAX_READY_QUEUE;
        }
        fprintf(fp, "]\n");
    } else {
        fprintf(fp, "  Queue: [] (empty)\n");
    }
    fprintf(fp, "\n");
    
    // TensorMap (active entries)
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "TENSOR MAP (non-empty buckets)\n");
    fprintf(fp, "================================================================================\n\n");
    int tensor_count = 0;
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        TensorMapEntry* entry = rt->tensor_map[i];
        while (entry) {
            fprintf(fp, "  [bucket %d] tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld] -> producer: Task %d\n",
                    i,
                    entry->region.raw_tensor,
                    (long long)entry->region.row_offset,
                    (long long)entry->region.col_offset,
                    (long long)entry->region.rows,
                    (long long)entry->region.cols,
                    entry->producer_id);
            tensor_count++;
            entry = entry->next;
        }
    }
    if (tensor_count == 0) {
        fprintf(fp, "  (empty)\n");
    }
    fprintf(fp, "\n  Total tensor entries: %d\n", tensor_count);
    fprintf(fp, "\n");
    
    // Dependency Graph (ASCII representation)
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "DEPENDENCY GRAPH (Producer -> Consumer)\n");
    fprintf(fp, "================================================================================\n\n");
    for (int32_t i = dump_start; i < dump_end; i++) {
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(i)];
        if (!task->is_active) continue;
        
        // Status indicator
        const char* status = task->is_complete ? "[DONE]" : 
                            (task->fanin == 0 ? "[READY]" : "[WAIT]");
        
        fprintf(fp, "  Task %d (%s) %s\n", i, 
                task->func_name ? task->func_name : "?", status);
        
        if (task->fanout_count > 0) {
            for (int j = 0; j < task->fanout_count; j++) {
                int32_t consumer_id = task->fanout[j];
                PendingTask* consumer = &rt->pend_task[PTO_TASK_SLOT(consumer_id)];
                fprintf(fp, "    └──> Task %d (%s)\n", consumer_id,
                        consumer->func_name ? consumer->func_name : "?");
            }
        }
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "END OF DUMP\n");
    fprintf(fp, "================================================================================\n");
    
    fclose(fp);
    DEBUG_PRINT("[PTO Runtime] Dumped runtime state to %s\n", filename);
    return 0;
}

int pto_runtime_dump_stdout(PTORuntime* rt) {
    if (!rt) return -1;
    
    printf("================================================================================\n");
    printf("PTO RUNTIME DUMP\n");
    printf("================================================================================\n\n");
    
    printf("SUMMARY\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  Total tasks scheduled:  %lld\n", (long long)rt->total_tasks_scheduled);
    printf("  Total tasks completed:  %lld\n", (long long)rt->total_tasks_completed);
    printf("  Active task count:      %d\n", rt->active_task_count);
    printf("  Next task ID:           %d\n", rt->next_task_id);
    printf("  Window oldest pending:  %d\n", rt->window_oldest_pending);
    printf("  Ready queue size:       %d\n", rt->ready_count);
    printf("\n");
    
    // Determine dump range (limited by window)
    int32_t dump_start = rt->window_oldest_pending;
    int32_t dump_end = rt->next_task_id;
    int32_t dump_count = dump_end - dump_start;
    if (dump_count > PTO_TASK_WINDOW_SIZE) {
        dump_count = PTO_TASK_WINDOW_SIZE;
        dump_start = dump_end - PTO_TASK_WINDOW_SIZE;
    }
    
    printf("TASK TABLE (window: %d to %d, %d tasks)\n", dump_start, dump_end - 1, dump_count);
    printf("--------------------------------------------------------------------------------\n");
    for (int32_t i = dump_start; i < dump_end; i++) {
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(i)];
        const char* status = task->is_complete ? "DONE" : 
                            (task->fanin == 0 ? "READY" : "WAIT");
        
        printf("  Task %d: %-20s [%s] fanin=%d buf=%.1fKB fanout=[",
               i, task->func_name ? task->func_name : "?", status, task->fanin,
               task->buffer_size_with_reuse / 1024.0);
        for (int j = 0; j < task->fanout_count; j++) {
            printf("%d", task->fanout[j]);
            if (j < task->fanout_count - 1) printf(",");
        }
        printf("]\n");
    }
    printf("\n");
    
    return 0;
}

// =============================================================================
// Cycle-Accurate Simulation Implementation
// =============================================================================

// A2A3 Core Simulator Integration (optional)
// When A2A3_CORE_SIM_AVAILABLE is defined and liba2a3_core.a is linked,
// cycle costs are computed by the cycle-accurate core model.
// Otherwise, heuristic estimates are used.

#ifdef A2A3_CORE_SIM_AVAILABLE
#include "ascend_a2a3_core_model/a2a3_sim_integration.h"
static bool g_core_sim_initialized = false;
#endif

// Heuristic-based cycle cost estimation (fallback)
static int64_t pto_estimate_cycle_cost_heuristic(const char* func_name) {
    if (!func_name) return 10;
    
    // MatMul operations (Cube Engine) - most expensive
    if (strstr(func_name, "matmul") || strstr(func_name, "gemm") || 
        strstr(func_name, "MATMUL") || strstr(func_name, "GEMM")) {
        return 50;  // Base matmul cost per tile
    }
    
    // Reduction operations
    if (strstr(func_name, "rowsum") || strstr(func_name, "rowmax") ||
        strstr(func_name, "colsum") || strstr(func_name, "colmax") ||
        strstr(func_name, "rowmin") || strstr(func_name, "colmin")) {
        return 20;
    }
    
    // Broadcast/expand operations
    if (strstr(func_name, "expand") || strstr(func_name, "broadcast")) {
        return 16;
    }
    
    // Activation functions
    if (strstr(func_name, "silu") || strstr(func_name, "gelu") ||
        strstr(func_name, "relu") || strstr(func_name, "sigmoid") ||
        strstr(func_name, "tanh")) {
        return 20;
    }
    
    // Softmax components
    if (strstr(func_name, "softmax") || strstr(func_name, "exp")) {
        return 24;
    }
    
    // Normalization
    if (strstr(func_name, "norm") || strstr(func_name, "rsqrt") ||
        strstr(func_name, "rmsnorm") || strstr(func_name, "layernorm")) {
        return 30;
    }
    
    // RoPE (rotary position embedding)
    if (strstr(func_name, "rope") || strstr(func_name, "rotary")) {
        return 40;
    }
    
    // Attention components
    if (strstr(func_name, "attention") || strstr(func_name, "score")) {
        return 60;
    }
    
    // Memory operations
    if (strstr(func_name, "load") || strstr(func_name, "store") ||
        strstr(func_name, "LOAD") || strstr(func_name, "STORE")) {
        return 100;  // GM access is expensive
    }
    
    // Basic element-wise ops
    if (strstr(func_name, "add") || strstr(func_name, "sub") ||
        strstr(func_name, "mul") || strstr(func_name, "div")) {
        return 8;
    }
    
    // Default
    return 10;
}

// Main cycle cost estimation function
// Uses core simulator when available, otherwise falls back to heuristics
int64_t pto_estimate_cycle_cost(const char* func_name) {
#ifdef A2A3_CORE_SIM_AVAILABLE
    // Initialize core simulator on first use
    if (!g_core_sim_initialized) {
        a2a3_sim_init();
        g_core_sim_initialized = true;
    }
    // Use core simulator for cycle estimation
    bool is_cube = (func_name && (strstr(func_name, "matmul") || strstr(func_name, "gemm") ||
                                   strstr(func_name, "MATMUL") || strstr(func_name, "GEMM")));
    return a2a3_sim_get_task_cycles(func_name, is_cube, 32 * 128);
#else
    // Fallback to heuristic estimation
    return pto_estimate_cycle_cost_heuristic(func_name);
#endif
}

// Cleanup function for core simulator (call during runtime shutdown)
void pto_cleanup_core_sim(void) {
#ifdef A2A3_CORE_SIM_AVAILABLE
    if (g_core_sim_initialized) {
        a2a3_sim_cleanup();
        g_core_sim_initialized = false;
    }
#endif
}

void pto_simulate_all(PTORuntime* rt) {
    if (!rt) return;
    
    // Use platform-configured number of workers
    // For A2A3: num_vector_workers + num_cube_workers (typically 48 + 24 = 72)
    // For other platforms: num_workers or default to 4
    int NUM_WORKERS = rt->num_workers;
    if (NUM_WORKERS <= 0) {
        // Fallback: check if A2A3 dual-queue mode with separate counts
        if (rt->dual_queue_mode && (rt->num_vector_workers > 0 || rt->num_cube_workers > 0)) {
            NUM_WORKERS = rt->num_vector_workers + rt->num_cube_workers;
        } else {
            NUM_WORKERS = 4;  // Default fallback
        }
    }
    
    // Safety: cap to PTO_MAX_WORKERS to prevent buffer overflow
    if (NUM_WORKERS > PTO_MAX_WORKERS) {
        fprintf(stderr, "[PTO Simulator] WARNING: Requested %d workers exceeds PTO_MAX_WORKERS (%d), capping\n",
                NUM_WORKERS, PTO_MAX_WORKERS);
        NUM_WORKERS = PTO_MAX_WORKERS;
    }
    
    printf("\n[PTO Simulator] ======== Starting Cycle-Accurate Simulation ========\n");
    printf("  Total tasks: %lld\n", (long long)rt->total_tasks_scheduled);
    if (rt->dual_queue_mode) {
        printf("  Workers: %d (%d vector + %d cube)\n", 
               NUM_WORKERS, rt->num_vector_workers, rt->num_cube_workers);
    } else {
        printf("  Workers: %d\n", NUM_WORKERS);
    }
    
    int64_t worker_cycles[PTO_MAX_WORKERS] = {0};
    int tasks_per_worker[PTO_MAX_WORKERS] = {0};
    int worker_round_robin = 0;
    
    // Process tasks in dependency order (within window)
    // For simulation, we assume all tasks are within the window (single-shot simulation)
    int32_t sim_start = rt->window_oldest_pending;
    int32_t sim_end = rt->next_task_id;
    int32_t total_to_simulate = sim_end - sim_start;
    
    if (total_to_simulate > PTO_TASK_WINDOW_SIZE) {
        fprintf(stderr, "[PTO Simulator] WARNING: %d tasks exceed window size %d, simulating last %d\n",
                total_to_simulate, PTO_TASK_WINDOW_SIZE, PTO_TASK_WINDOW_SIZE);
        sim_start = sim_end - PTO_TASK_WINDOW_SIZE;
        total_to_simulate = PTO_TASK_WINDOW_SIZE;
    }
    
    int tasks_completed = 0;
    int max_iterations = total_to_simulate * 2;  // Safety limit
    int iteration = 0;
    
    while (tasks_completed < total_to_simulate && iteration < max_iterations) {
        iteration++;
        bool made_progress = false;
        
        // Find a ready task (fanin == 0 and not complete)
        for (int32_t task_id = sim_start; task_id < sim_end; task_id++) {
            PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
            
            if (task->is_complete || task->fanin > 0) {
                continue;
            }
            
            // Found a ready task - simulate its execution
            const char* func_name = task->func_name ? task->func_name : "unknown";
            
            // Estimate cycle cost
            int64_t cycle_cost = task->cycle_func 
                ? task->cycle_func(NULL, 0) 
                : pto_estimate_cycle_cost(func_name);
            
            // Assign to worker based on task type (dual-queue mode) or simple load balancing
            int best_worker = 0;
            if (rt->dual_queue_mode && rt->num_vector_workers > 0 && rt->num_cube_workers > 0) {
                // Dual-queue mode: respect is_cube flag
                int worker_start, worker_end;
                if (task->is_cube) {
                    // Cube task → assign to cube workers only
                    worker_start = rt->num_vector_workers;
                    worker_end = NUM_WORKERS;
                } else {
                    // Vector task → assign to vector workers only
                    worker_start = 0;
                    worker_end = rt->num_vector_workers;
                }
                
                // Find worker with lowest cycle count within the appropriate range
                best_worker = worker_start;
                for (int w = worker_start + 1; w < worker_end; w++) {
                    if (worker_cycles[w] < worker_cycles[best_worker]) {
                        best_worker = w;
                    }
                }
            } else {
                // Single-queue mode: simple load balancing across all workers
                for (int w = 1; w < NUM_WORKERS; w++) {
                    if (worker_cycles[w] < worker_cycles[best_worker]) {
                        best_worker = w;
                    }
                }
            }
            
            // Calculate actual start time (max of worker availability and task's earliest start)
            int64_t task_earliest = task->earliest_start_cycle;
            int64_t actual_start = (worker_cycles[best_worker] > task_earliest) 
                                  ? worker_cycles[best_worker] 
                                  : task_earliest;
            int64_t actual_end = actual_start + cycle_cost;
            
            // Update worker state
            worker_cycles[best_worker] = actual_end;
            tasks_per_worker[best_worker]++;
            
            // Record trace
            if (pto_global_trace) {
                pto_trace_record_with_time(best_worker, func_name, actual_start, actual_end);
            }
            
            // Update task state
            task->end_cycle = actual_end;
            task->is_complete = true;
            tasks_completed++;
            made_progress = true;
            
            // Update dependencies - reduce fanin of dependent tasks
            for (int i = 0; i < task->fanout_count; i++) {
                int32_t dep_id = task->fanout[i];
                if (dep_id >= sim_start && dep_id < sim_end) {
                    PendingTask* dep_task = &rt->pend_task[PTO_TASK_SLOT(dep_id)];
                    if (dep_task->fanin > 0) {
                        dep_task->fanin--;
                    }
                    // Update dependent task's earliest start time
                    if (actual_end > dep_task->earliest_start_cycle) {
                        dep_task->earliest_start_cycle = actual_end;
                    }
                }
            }
            
            DEBUG_PRINT("[Sim] Task %d: %s on Worker %d, cycles=[%lld, %lld], cost=%lld\n",
                   task_id, func_name, best_worker, 
                   (long long)actual_start, (long long)actual_end, (long long)cycle_cost);
        }
        
        if (!made_progress) {
            // Check if there are incomplete tasks with non-zero fanin (deadlock)
            int waiting_tasks = 0;
            for (int32_t i = sim_start; i < sim_end; i++) {
                if (!rt->pend_task[PTO_TASK_SLOT(i)].is_complete) {
                    waiting_tasks++;
                }
            }
            if (waiting_tasks > 0) {
                fprintf(stderr, "[PTO Simulator] WARNING: %d tasks waiting with dependencies - possible deadlock\n", 
                        waiting_tasks);
            }
            break;
        }
    }
    
    // Calculate statistics
    int64_t max_cycle = 0;
    for (int w = 0; w < NUM_WORKERS; w++) {
        if (worker_cycles[w] > max_cycle) {
            max_cycle = worker_cycles[w];
        }
    }
    
    printf("\n[PTO Simulator] ======== Simulation Complete ========\n");
    printf("  Tasks simulated: %d / %d\n", tasks_completed, total_to_simulate);
    printf("  Makespan (total cycles): %lld\n", (long long)max_cycle);
    
    // Count active workers (those that executed at least one task)
    int active_workers = 0;
    for (int w = 0; w < NUM_WORKERS; w++) {
        if (tasks_per_worker[w] > 0) active_workers++;
    }
    
    printf("  Active workers: %d / %d\n", active_workers, NUM_WORKERS);
    printf("  Worker utilization (top workers):\n");
    
    // Only show workers that did work (limit output for large worker counts)
    int shown = 0;
    int max_to_show = (active_workers > 20) ? 10 : active_workers;
    for (int w = 0; w < NUM_WORKERS && shown < max_to_show; w++) {
        if (tasks_per_worker[w] > 0) {
            double util = max_cycle > 0 ? (100.0 * worker_cycles[w] / max_cycle) : 0;
            printf("    Worker %d: %lld cycles (%d tasks, %.1f%% utilization)\n",
                   w, (long long)worker_cycles[w], tasks_per_worker[w], util);
            shown++;
        }
    }
    if (active_workers > max_to_show) {
        printf("    ... and %d more workers\n", active_workers - max_to_show);
    }
    printf("\n");
    
    rt->total_tasks_completed = tasks_completed;
}
