/**
 * PTO Runtime - Ascend A2/A3 Host Layer Implementation
 * 
 * This file implements the host CPU interface for A2A3 execution.
 */

#include "a2a3_host.h"
#include <time.h>

// =============================================================================
// Public API Implementation
// =============================================================================

void pto_runtime_enable_a2a3_sim(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers) {
    if (!rt) return;
    a2a3_orch_init(rt, num_vector_workers, num_cube_workers);
    DEBUG_PRINT("[A2A3 Host] Simulation mode enabled: %d vector + %d cube workers\n",
                num_vector_workers, num_cube_workers);
}

// =============================================================================
// Task Execution
// =============================================================================

static void execute_task_a2a3(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[A2A3 Host] Worker %d executing task %d: %s\n", 
                worker_id, task_id, task->func_name);
    
    // Build argument array
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < task->num_args; i++) {
        TaskArg* arg = &task->args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Simulation mode: use cycle cost function
    if (rt->simulation_mode && task->cycle_func) {
        int64_t cycle_cost = task->cycle_func(args, task->num_args);
        
        int64_t worker_current = pto_trace_get_cycle(worker_id);
        int64_t actual_start = (worker_current > task->earliest_start_cycle) ? 
            worker_current : task->earliest_start_cycle;
        int64_t actual_end = actual_start + cycle_cost;
        
        task->end_cycle = actual_end;
        
        pto_trace_record_with_time(worker_id, task->func_name, actual_start, actual_end);
        DEBUG_PRINT("[A2A3 Host] Task %d simulated: %lld cycles\n", 
                    task_id, (long long)cycle_cost);
    }
    // Normal mode: execute actual function
    else if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    }
}

// =============================================================================
// Worker Threads
// =============================================================================

typedef struct {
    PTORuntime* rt;
    int worker_id;
    bool is_cube_worker;
} A2A3WorkerContext;

static void* vector_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Host] Vector worker %d started\n", worker_id);
    
    while (!rt->shutdown_requested) {
        int32_t task_id = a2a3_orch_get_vector_task_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        execute_task_a2a3(rt, task_id, worker_id);
        a2a3_orch_complete_task_threadsafe(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Host] Vector worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}

static void* cube_worker_func(void* arg) {
    A2A3WorkerContext* ctx = (A2A3WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[A2A3 Host] Cube worker %d started\n", worker_id);
    
    while (!rt->shutdown_requested) {
        int32_t task_id = a2a3_orch_get_cube_task_blocking(rt);
        
        if (task_id < 0) {
            if (rt->shutdown_requested) break;
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) break;
            continue;
        }
        
        execute_task_a2a3(rt, task_id, worker_id);
        a2a3_orch_complete_task_threadsafe(rt, task_id);
    }
    
    DEBUG_PRINT("[A2A3 Host] Cube worker %d exiting\n", worker_id);
    free(ctx);
    return NULL;
}

// =============================================================================
// Runtime Entry Point
// =============================================================================

int runtime_entry_a2a3(PTOOrchFunc orch_func, void* user_data, 
                       int num_vector_workers, int num_cube_workers,
                       int execution_task_threshold) {
    if (!orch_func) {
        fprintf(stderr, "[A2A3 Host] ERROR: No orchestration function\n");
        return -1;
    }
    
    // Apply defaults
    if (num_vector_workers < 1) num_vector_workers = A2A3_DEFAULT_VECTOR_WORKERS;
    if (num_cube_workers < 1) num_cube_workers = A2A3_DEFAULT_CUBE_WORKERS;
    
    int total_workers = num_vector_workers + num_cube_workers;
    if (total_workers > PTO_MAX_WORKERS) {
        fprintf(stderr, "[A2A3 Host] ERROR: Total workers (%d) exceeds max (%d)\n",
                total_workers, PTO_MAX_WORKERS);
        return -1;
    }
    if (execution_task_threshold < 0) execution_task_threshold = 0;
    
    printf("[A2A3 Host] ========================================\n");
    printf("[A2A3 Host] Ascend A2/A3 Dual-Queue Execution\n");
    printf("[A2A3 Host] Vector workers: %d\n", num_vector_workers);
    printf("[A2A3 Host] Cube workers:   %d\n", num_cube_workers);
    if (execution_task_threshold > 0) {
        printf("[A2A3 Host] Pipeline threshold: %d tasks\n", execution_task_threshold);
    }
    printf("[A2A3 Host] ========================================\n");
    
    // Allocate runtime
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "[A2A3 Host] ERROR: Failed to allocate runtime\n");
        return -1;
    }
    
    // Initialize
    pto_runtime_init(rt);
    pto_runtime_enable_a2a3_sim(rt, num_vector_workers, num_cube_workers);
    rt->num_workers = total_workers;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    rt->execution_task_threshold = execution_task_threshold;
    
    // Spawn vector workers
    printf("[A2A3 Host] Spawning %d vector workers...\n", num_vector_workers);
    for (int i = 0; i < num_vector_workers; i++) {
        A2A3WorkerContext* ctx = (A2A3WorkerContext*)malloc(sizeof(A2A3WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[A2A3 Host] ERROR: Failed to allocate worker context\n");
            rt->shutdown_requested = true;
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        ctx->rt = rt;
        ctx->worker_id = i;
        ctx->is_cube_worker = false;
        
        if (pthread_create(&rt->workers[i], NULL, vector_worker_func, ctx) != 0) {
            fprintf(stderr, "[A2A3 Host] ERROR: Failed to create vector worker %d\n", i);
            free(ctx);
            rt->shutdown_requested = true;
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
    }
    
    // Spawn cube workers
    printf("[A2A3 Host] Spawning %d cube workers...\n", num_cube_workers);
    for (int i = 0; i < num_cube_workers; i++) {
        int worker_idx = num_vector_workers + i;
        A2A3WorkerContext* ctx = (A2A3WorkerContext*)malloc(sizeof(A2A3WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[A2A3 Host] ERROR: Failed to allocate worker context\n");
            rt->shutdown_requested = true;
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        ctx->rt = rt;
        ctx->worker_id = worker_idx;
        ctx->is_cube_worker = true;
        
        if (pthread_create(&rt->workers[worker_idx], NULL, cube_worker_func, ctx) != 0) {
            fprintf(stderr, "[A2A3 Host] ERROR: Failed to create cube worker %d\n", i);
            free(ctx);
            rt->shutdown_requested = true;
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
    }
    
    // Wait for workers to start
    struct timespec start_delay = {0, 10000000};  // 10ms
    nanosleep(&start_delay, NULL);
    printf("[A2A3 Host] Workers started, building task graph...\n");
    
    // Build task graph
    orch_func(rt, user_data);
    
    // Mark orchestration complete
    pthread_mutex_lock(&rt->task_mutex);
    rt->execution_started = true;
    int64_t total_tasks = rt->total_tasks_scheduled;
    pthread_mutex_unlock(&rt->task_mutex);
    
    printf("[A2A3 Host] Task graph built: %lld tasks\n", (long long)total_tasks);
    
    // Wake up workers
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->vector_queue_not_empty);
    pthread_cond_broadcast(&rt->cube_queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    // Wait for completion
    struct timespec poll_interval = {0, 1000000};  // 1ms
    while (1) {
        pthread_mutex_lock(&rt->task_mutex);
        bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
        int64_t completed = rt->total_tasks_completed;
        pthread_mutex_unlock(&rt->task_mutex);
        
        if (all_done) {
            printf("[A2A3 Host] All %lld tasks completed!\n", (long long)completed);
            break;
        }
        
        static int64_t last_reported = 0;
        if (completed > last_reported + 1000 || completed == total_tasks) {
            printf("[A2A3 Host] Progress: %lld / %lld (%.1f%%)\n",
                   (long long)completed, (long long)total_tasks,
                   100.0 * completed / total_tasks);
            last_reported = completed;
        }
        
        nanosleep(&poll_interval, NULL);
    }
    
    // Shutdown
    printf("[A2A3 Host] Shutting down workers...\n");
    rt->shutdown_requested = true;
    
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->vector_queue_not_empty);
    pthread_cond_broadcast(&rt->cube_queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    for (int i = 0; i < total_workers; i++) {
        pthread_join(rt->workers[i], NULL);
    }
    
    // Print statistics
    printf("[A2A3 Host] ========================================\n");
    printf("[A2A3 Host] Execution Statistics\n");
    printf("[A2A3 Host]   Total tasks:     %lld\n", (long long)rt->total_tasks_scheduled);
    printf("[A2A3 Host]   Completed:       %lld\n", (long long)rt->total_tasks_completed);
    printf("[A2A3 Host]   Vector workers:  %d\n", num_vector_workers);
    printf("[A2A3 Host]   Cube workers:    %d\n", num_cube_workers);
    printf("[A2A3 Host] ========================================\n");
    
    a2a3_orch_print_stats(rt);
    
    pto_runtime_shutdown(rt);
    free(rt);
    
    return 0;
}

// =============================================================================
// Memory Management (Simulation mode uses host memory)
// =============================================================================

void* a2a3_host_malloc(size_t size_bytes) {
    return calloc(1, size_bytes);
}

void a2a3_host_free(void* ptr) {
    free(ptr);
}

int a2a3_host_copy_to_device(void* dst_device, const void* src_host, size_t size_bytes) {
    memcpy(dst_device, src_host, size_bytes);
    return 0;
}

int a2a3_host_copy_from_device(void* dst_host, const void* src_device, size_t size_bytes) {
    memcpy(dst_host, src_device, size_bytes);
    return 0;
}

void a2a3_host_synchronize(void) {
    // No-op in simulation mode
}

// =============================================================================
// Device Query
// =============================================================================

void a2a3_host_get_device_info(A2A3DeviceInfo* info) {
    if (!info) return;
    
    info->name = "Ascend 910B (Simulated)";
    info->num_vector_cores = A2A3_NUM_VECTOR_CORES;
    info->num_cube_cores = A2A3_NUM_CUBE_CORES;
    info->global_memory_bytes = (int64_t)A2A3_GLOBAL_MEMORY_GB * 1024 * 1024 * 1024;
    info->l2_cache_bytes = (int64_t)A2A3_L2_CACHE_MB * 1024 * 1024;
    info->l1_buffer_bytes = A2A3_L1_SIZE_BYTES;
    info->compute_capability = 320.0;  // ~320 TFLOPS FP16
}

void a2a3_host_print_device_info(void) {
    A2A3DeviceInfo info;
    a2a3_host_get_device_info(&info);
    
    printf("[A2A3 Device] %s\n", info.name);
    printf("  Vector cores: %d\n", info.num_vector_cores);
    printf("  Cube cores:   %d\n", info.num_cube_cores);
    printf("  Global memory: %lld GB\n", (long long)(info.global_memory_bytes / (1024*1024*1024)));
    printf("  L2 cache:     %lld MB\n", (long long)(info.l2_cache_bytes / (1024*1024)));
    printf("  L1 buffer:    %lld KB\n", (long long)(info.l1_buffer_bytes / 1024));
    printf("  Compute:      %.0f TFLOPS (FP16)\n", info.compute_capability);
}
