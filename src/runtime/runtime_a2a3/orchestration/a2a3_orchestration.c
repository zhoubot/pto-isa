/**
 * PTO Runtime - Ascend A2/A3 Orchestration Implementation
 * 
 * This file implements the dual-queue orchestration for heterogeneous
 * Vector and Cube core execution on Ascend A2/A3 NPU.
 */

#include "a2a3_orchestration.h"
#include <time.h>

// =============================================================================
// Dual Ready Queue Implementation
// =============================================================================

static void vector_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->vector_ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[A2A3 Orch] ERROR: Vector queue overflow\n");
        return;
    }
    
    rt->vector_ready_queue[rt->vector_ready_tail] = task_id;
    rt->vector_ready_tail = (rt->vector_ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count++;
}

static int32_t vector_queue_pop(PTORuntime* rt) {
    if (rt->vector_ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->vector_ready_queue[rt->vector_ready_head];
    rt->vector_ready_head = (rt->vector_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count--;
    return task_id;
}

static void cube_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->cube_ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[A2A3 Orch] ERROR: Cube queue overflow\n");
        return;
    }
    
    rt->cube_ready_queue[rt->cube_ready_tail] = task_id;
    rt->cube_ready_tail = (rt->cube_ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count++;
}

static int32_t cube_queue_pop(PTORuntime* rt) {
    if (rt->cube_ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->cube_ready_queue[rt->cube_ready_head];
    rt->cube_ready_head = (rt->cube_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count--;
    return task_id;
}

// =============================================================================
// Public API Implementation
// =============================================================================

void a2a3_orch_init(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers) {
    if (!rt) return;
    
    rt->simulation_mode = true;
    rt->dual_queue_mode = true;
    rt->num_vector_workers = num_vector_workers;
    rt->num_cube_workers = num_cube_workers;
    
    // Initialize trace with dual-mode (vector=pid0, cube=pid1)
    pto_trace_init_dual(num_vector_workers, num_cube_workers);
    
    DEBUG_PRINT("[A2A3 Orch] Initialized: %d vector + %d cube workers\n",
                num_vector_workers, num_cube_workers);
}

void a2a3_orch_route_to_queue(PTORuntime* rt, int32_t task_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    bool is_cube = rt->pend_task[slot].is_cube;
    
    if (is_cube) {
        cube_queue_push(rt, task_id);
        DEBUG_PRINT("[A2A3 Orch] Task %d -> CUBE queue\n", task_id);
    } else {
        vector_queue_push(rt, task_id);
        DEBUG_PRINT("[A2A3 Orch] Task %d -> VECTOR queue\n", task_id);
    }
}

void a2a3_orch_route_to_queue_threadsafe(PTORuntime* rt, int32_t task_id) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    bool is_cube = rt->pend_task[slot].is_cube;
    
    if (is_cube) {
        if (rt->cube_ready_count >= PTO_MAX_READY_QUEUE) {
            fprintf(stderr, "[A2A3 Orch] ERROR: Cube queue overflow\n");
            pthread_mutex_unlock(&rt->queue_mutex);
            return;
        }
        rt->cube_ready_queue[rt->cube_ready_tail] = task_id;
        rt->cube_ready_tail = (rt->cube_ready_tail + 1) % PTO_MAX_READY_QUEUE;
        rt->cube_ready_count++;
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        DEBUG_PRINT("[A2A3 Orch] Task %d -> CUBE queue (count=%d)\n", task_id, rt->cube_ready_count);
    } else {
        if (rt->vector_ready_count >= PTO_MAX_READY_QUEUE) {
            fprintf(stderr, "[A2A3 Orch] ERROR: Vector queue overflow\n");
            pthread_mutex_unlock(&rt->queue_mutex);
            return;
        }
        rt->vector_ready_queue[rt->vector_ready_tail] = task_id;
        rt->vector_ready_tail = (rt->vector_ready_tail + 1) % PTO_MAX_READY_QUEUE;
        rt->vector_ready_count++;
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        DEBUG_PRINT("[A2A3 Orch] Task %d -> VECTOR queue (count=%d)\n", task_id, rt->vector_ready_count);
    }
    
    pthread_mutex_unlock(&rt->queue_mutex);
}

void a2a3_orch_submit_task(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[A2A3 Orch] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
    
    DEBUG_PRINT("[A2A3 Orch] Submit task %d: %s (fanin=%d, is_cube=%d)\n",
                task_id, task->func_name, task->fanin, task->is_cube);
    
    // If no dependencies, route to ready queue
    if (task->fanin == 0) {
        a2a3_orch_route_to_queue_threadsafe(rt, task_id);
    }
}

void a2a3_orch_complete_task(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[A2A3 Orch] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    // Advance window
    while (rt->window_oldest_pending < rt->next_task_id) {
        int32_t oldest_slot = PTO_TASK_SLOT(rt->window_oldest_pending);
        if (!rt->pend_task[oldest_slot].is_complete) break;
        rt->window_oldest_pending++;
    }
    
    DEBUG_PRINT("[A2A3 Orch] Complete task %d: %s\n", task_id, task->func_name);
    
    // Process dependents
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        int32_t dep_slot = PTO_TASK_SLOT(dep_id);
        PendingTask* dep_task = &rt->pend_task[dep_slot];
        
        // Update earliest_start_cycle
        if (task->end_cycle > dep_task->earliest_start_cycle) {
            dep_task->earliest_start_cycle = task->end_cycle;
        }
        
        dep_task->fanin--;
        
        if (dep_task->fanin == 0 && !dep_task->is_complete) {
            a2a3_orch_route_to_queue(rt, dep_id);
        }
    }
}

void a2a3_orch_complete_task_threadsafe(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[A2A3 Orch] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    pthread_mutex_lock(&rt->task_mutex);
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    // Advance window
    bool window_advanced = false;
    while (rt->window_oldest_pending < rt->next_task_id) {
        int32_t oldest_slot = PTO_TASK_SLOT(rt->window_oldest_pending);
        if (!rt->pend_task[oldest_slot].is_complete) break;
        rt->window_oldest_pending++;
        window_advanced = true;
    }
    
    DEBUG_PRINT("[A2A3 Orch] Complete task %d (threadsafe): %s\n", task_id, task->func_name);
    
    // Collect newly ready tasks
    int32_t newly_ready[PTO_MAX_FANOUT];
    int32_t newly_ready_count = 0;
    
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        int32_t dep_slot = PTO_TASK_SLOT(dep_id);
        PendingTask* dep_task = &rt->pend_task[dep_slot];
        
        dep_task->fanin--;
        
        if (dep_task->fanin == 0 && !dep_task->is_complete) {
            newly_ready[newly_ready_count++] = dep_id;
        }
    }
    
    bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
    
    if (window_advanced) {
        pthread_cond_broadcast(&rt->window_not_full);
    }
    
    pthread_mutex_unlock(&rt->task_mutex);
    
    // Route newly ready tasks
    for (int i = 0; i < newly_ready_count; i++) {
        a2a3_orch_route_to_queue_threadsafe(rt, newly_ready[i]);
    }
    
    // Signal completion
    if (all_done) {
        pthread_mutex_lock(&rt->queue_mutex);
        pthread_cond_broadcast(&rt->all_done);
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        pthread_mutex_unlock(&rt->queue_mutex);
    }
}

// =============================================================================
// Queue Access
// =============================================================================

int32_t a2a3_orch_get_vector_task(PTORuntime* rt) {
    return vector_queue_pop(rt);
}

int32_t a2a3_orch_get_cube_task(PTORuntime* rt) {
    return cube_queue_pop(rt);
}

int32_t a2a3_orch_get_vector_task_blocking(PTORuntime* rt) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    while ((rt->vector_ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;
        }
        
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;  // 100µs
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->vector_queue_not_empty, &rt->queue_mutex, &timeout);
        
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->vector_ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->vector_ready_queue[rt->vector_ready_head];
    rt->vector_ready_head = (rt->vector_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

int32_t a2a3_orch_get_cube_task_blocking(PTORuntime* rt) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    while ((rt->cube_ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;
        }
        
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;  // 100µs
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->cube_queue_not_empty, &rt->queue_mutex, &timeout);
        
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->cube_ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->cube_ready_queue[rt->cube_ready_head];
    rt->cube_ready_head = (rt->cube_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

// =============================================================================
// Statistics
// =============================================================================

void a2a3_orch_get_stats(PTORuntime* rt, A2A3OrchStats* stats) {
    if (!rt || !stats) return;
    
    memset(stats, 0, sizeof(A2A3OrchStats));
    
    // Count completed tasks by type
    for (int i = 0; i < rt->next_task_id; i++) {
        int32_t slot = PTO_TASK_SLOT(i);
        PendingTask* task = &rt->pend_task[slot];
        if (task->is_complete) {
            if (task->is_cube) {
                stats->cube_tasks_executed++;
            } else {
                stats->vector_tasks_executed++;
            }
        }
    }
}

void a2a3_orch_print_stats(PTORuntime* rt) {
    A2A3OrchStats stats;
    a2a3_orch_get_stats(rt, &stats);
    
    printf("[A2A3 Orch] Statistics:\n");
    printf("  Vector tasks: %lld\n", (long long)stats.vector_tasks_executed);
    printf("  Cube tasks:   %lld\n", (long long)stats.cube_tasks_executed);
}
