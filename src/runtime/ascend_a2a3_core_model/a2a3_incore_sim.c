/**
 * Ascend A2/A3 InCore Function Simulator - Implementation
 */

#include "a2a3_incore_sim.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

// =============================================================================
// Internal Helpers
// =============================================================================

static char* trim_whitespace(char* str) {
    if (!str) return str;
    
    // Leading whitespace
    while (isspace((unsigned char)*str)) str++;
    
    if (*str == 0) return str;
    
    // Trailing whitespace
    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    
    return str;
}

static bool is_comment_or_empty(const char* line) {
    if (!line || !*line) return true;
    
    const char* p = line;
    while (isspace((unsigned char)*p)) p++;
    
    return (*p == '\0' || *p == '/' || *p == '#');
}

// =============================================================================
// Simulator Lifecycle
// =============================================================================

IncoreSimulator* a2a3_incore_sim_create(void) {
    IncoreSimulator* sim = (IncoreSimulator*)calloc(1, sizeof(IncoreSimulator));
    if (!sim) return NULL;
    
    // Create core models
    sim->cube_core = a2a3_core_create(CORE_TYPE_CUBE, 0);
    sim->vector_core = a2a3_core_create(CORE_TYPE_VECTOR, 0);
    
    if (!sim->cube_core || !sim->vector_core) {
        a2a3_incore_sim_destroy(sim);
        return NULL;
    }
    
    // Initialize function registry
    sim->capacity = 64;
    sim->functions = (IncoreFunction**)calloc(sim->capacity, sizeof(IncoreFunction*));
    sim->num_functions = 0;
    
    sim->trace_enabled = false;
    sim->trace_file = NULL;
    
    return sim;
}

void a2a3_incore_sim_destroy(IncoreSimulator* sim) {
    if (!sim) return;
    
    if (sim->cube_core) a2a3_core_destroy(sim->cube_core);
    if (sim->vector_core) a2a3_core_destroy(sim->vector_core);
    
    if (sim->functions) {
        for (int i = 0; i < sim->num_functions; i++) {
            free(sim->functions[i]);
        }
        free(sim->functions);
    }
    
    free(sim);
}

void a2a3_incore_sim_reset(IncoreSimulator* sim) {
    if (!sim) return;
    
    a2a3_core_reset(sim->cube_core);
    a2a3_core_reset(sim->vector_core);
    
    // Invalidate cached results
    for (int i = 0; i < sim->num_functions; i++) {
        if (sim->functions[i]) {
            sim->functions[i]->cache_valid = false;
        }
    }
    
    sim->total_simulations = 0;
    sim->total_cycles_simulated = 0;
}

// =============================================================================
// Function Management
// =============================================================================

int a2a3_incore_sim_register(IncoreSimulator* sim, const char* name,
                             CoreType core_type, const char** instructions,
                             int num_instructions, int tile_rows, int tile_cols) {
    if (!sim || !name || !instructions || num_instructions <= 0) return -1;
    
    // Expand registry if needed
    if (sim->num_functions >= sim->capacity) {
        int new_capacity = sim->capacity * 2;
        IncoreFunction** new_funcs = (IncoreFunction**)realloc(
            sim->functions, new_capacity * sizeof(IncoreFunction*));
        if (!new_funcs) return -1;
        sim->functions = new_funcs;
        sim->capacity = new_capacity;
    }
    
    // Allocate function
    IncoreFunction* func = (IncoreFunction*)calloc(1, sizeof(IncoreFunction));
    if (!func) return -1;
    
    strncpy(func->name, name, MAX_INCORE_NAME - 1);
    func->core_type = core_type;
    func->tile_rows = tile_rows;
    func->tile_cols = tile_cols;
    func->element_size = 4;  // Assume float32
    func->cache_valid = false;
    
    // Parse instructions
    func->num_instructions = 0;
    for (int i = 0; i < num_instructions && i < MAX_INCORE_INSTRUCTIONS; i++) {
        if (is_comment_or_empty(instructions[i])) continue;
        
        ParsedInstruction* pi = &func->instructions[func->num_instructions];
        strncpy(pi->text, instructions[i], sizeof(pi->text) - 1);
        
        if (a2a3_parse_instruction(instructions[i], &pi->decoded, core_type)) {
            func->num_instructions++;
        }
    }
    
    // Add to registry
    int func_id = sim->num_functions;
    sim->functions[sim->num_functions++] = func;
    
    return func_id;
}

int a2a3_incore_sim_register_code(IncoreSimulator* sim, const char* name,
                                  CoreType core_type, const char* code_block,
                                  int tile_rows, int tile_cols) {
    if (!sim || !name || !code_block) return -1;
    
    // Count lines
    int num_lines = 1;
    for (const char* p = code_block; *p; p++) {
        if (*p == '\n') num_lines++;
    }
    
    // Allocate line array
    const char** lines = (const char**)malloc(num_lines * sizeof(const char*));
    char* code_copy = strdup(code_block);
    if (!lines || !code_copy) {
        free(lines);
        free(code_copy);
        return -1;
    }
    
    // Split by newlines
    int line_count = 0;
    char* line = strtok(code_copy, "\n");
    while (line && line_count < num_lines) {
        lines[line_count++] = trim_whitespace(line);
        line = strtok(NULL, "\n");
    }
    
    // Register
    int result = a2a3_incore_sim_register(sim, name, core_type, lines,
                                          line_count, tile_rows, tile_cols);
    
    free(lines);
    free(code_copy);
    
    return result;
}

int a2a3_incore_sim_find(IncoreSimulator* sim, const char* name) {
    if (!sim || !name) return -1;
    
    for (int i = 0; i < sim->num_functions; i++) {
        if (sim->functions[i] && strcmp(sim->functions[i]->name, name) == 0) {
            return i;
        }
    }
    
    return -1;
}

// =============================================================================
// Simulation
// =============================================================================

int64_t a2a3_incore_sim_execute(IncoreSimulator* sim, int func_id) {
    if (!sim || func_id < 0 || func_id >= sim->num_functions) return 0;
    
    IncoreFunction* func = sim->functions[func_id];
    if (!func) return 0;
    
    // Return cached result if valid
    if (func->cache_valid) {
        return func->cached_cycles;
    }
    
    // Select core based on function type
    A2A3Core* core = (func->core_type == CORE_TYPE_CUBE) 
                     ? sim->cube_core : sim->vector_core;
    
    // Reset core for simulation
    a2a3_core_reset(core);
    
    // Enable tracing if requested
    if (sim->trace_enabled && sim->trace_file) {
        a2a3_core_enable_trace(core, sim->trace_file);
        fprintf(sim->trace_file, "\n=== Simulating InCore function: %s ===\n", func->name);
    }
    
    // Execute each instruction
    for (int i = 0; i < func->num_instructions; i++) {
        a2a3_core_execute(core, &func->instructions[i].decoded);
    }
    
    // Drain all pending operations
    int64_t total_cycles = a2a3_core_drain(core);
    
    // Cache result
    func->cached_cycles = total_cycles;
    func->cache_valid = true;
    
    // Update statistics
    sim->total_simulations++;
    sim->total_cycles_simulated += total_cycles;
    
    if (sim->trace_enabled && sim->trace_file) {
        fprintf(sim->trace_file, "=== Function %s completed: %lld cycles ===\n\n",
                func->name, (long long)total_cycles);
        a2a3_core_disable_trace(core);
    }
    
    return total_cycles;
}

int64_t a2a3_incore_sim_execute_by_name(IncoreSimulator* sim, const char* name) {
    int func_id = a2a3_incore_sim_find(sim, name);
    if (func_id < 0) {
        // Function not registered - use heuristic estimate
        return a2a3_get_incore_cycle_cost(name, 32 * 128);
    }
    return a2a3_incore_sim_execute(sim, func_id);
}

int64_t a2a3_incore_sim_estimate(IncoreSimulator* sim, int func_id) {
    if (!sim || func_id < 0 || func_id >= sim->num_functions) return 0;
    
    IncoreFunction* func = sim->functions[func_id];
    if (!func) return 0;
    
    // Simple heuristic: sum of instruction latencies
    int64_t total = 0;
    for (int i = 0; i < func->num_instructions; i++) {
        total += func->instructions[i].decoded.latency;
    }
    
    return total;
}

// =============================================================================
// Cycle Cost Query
// =============================================================================

int64_t a2a3_get_incore_cycle_cost(const char* func_name, int64_t tile_size) {
    if (!func_name) return 10;
    
    // Heuristic-based cycle estimation for common functions
    // This is used when detailed simulation is not available
    
    // Matrix multiply (Cube)
    if (strstr(func_name, "matmul") || strstr(func_name, "MATMUL") ||
        strstr(func_name, "linear") || strstr(func_name, "gemm")) {
        return CUBE_MATMUL_LATENCY + (tile_size / 64);
    }
    
    // RMSNorm (multiple vector ops + reduction)
    if (strstr(func_name, "rmsnorm") || strstr(func_name, "layernorm")) {
        return VEC_REDUCE_LATENCY * 2 + VEC_BINARY_LATENCY * 3;
    }
    
    // Softmax (reduction + element-wise)
    if (strstr(func_name, "softmax")) {
        return VEC_REDUCE_LATENCY * 2 + VEC_UNARY_LATENCY + VEC_BINARY_LATENCY * 2;
    }
    
    // RoPE (element-wise)
    if (strstr(func_name, "rope") || strstr(func_name, "rotary")) {
        return VEC_BINARY_LATENCY * 4 + VEC_UNARY_LATENCY * 2;
    }
    
    // SwiGLU (activation + multiply)
    if (strstr(func_name, "swiglu") || strstr(func_name, "silu")) {
        return VEC_ACTIVATION_LATENCY + VEC_BINARY_LATENCY;
    }
    
    // Attention score/output (matmul + softmax)
    if (strstr(func_name, "attention") || strstr(func_name, "score")) {
        return CUBE_MATMUL_LATENCY + VEC_REDUCE_LATENCY * 2;
    }
    
    // Reduction ops
    if (strstr(func_name, "rowsum") || strstr(func_name, "rowmax") ||
        strstr(func_name, "colsum") || strstr(func_name, "reduce")) {
        return VEC_REDUCE_LATENCY;
    }
    
    // Simple element-wise ops
    if (strstr(func_name, "add") || strstr(func_name, "mul") ||
        strstr(func_name, "sub") || strstr(func_name, "div")) {
        return VEC_BINARY_LATENCY;
    }
    
    // Activation functions
    if (strstr(func_name, "relu") || strstr(func_name, "gelu") ||
        strstr(func_name, "sigmoid") || strstr(func_name, "tanh")) {
        return VEC_ACTIVATION_LATENCY;
    }
    
    // Math functions
    if (strstr(func_name, "exp") || strstr(func_name, "log") ||
        strstr(func_name, "sqrt") || strstr(func_name, "rsqrt")) {
        return VEC_UNARY_LATENCY;
    }
    
    // Default
    return VEC_BINARY_LATENCY;
}

// =============================================================================
// Tracing
// =============================================================================

void a2a3_incore_sim_enable_trace(IncoreSimulator* sim, FILE* trace_file) {
    if (!sim) return;
    sim->trace_enabled = true;
    sim->trace_file = trace_file ? trace_file : stdout;
}

void a2a3_incore_sim_disable_trace(IncoreSimulator* sim) {
    if (!sim) return;
    sim->trace_enabled = false;
}

void a2a3_incore_sim_print_stats(const IncoreSimulator* sim) {
    if (!sim) return;
    
    printf("\n=== InCore Simulator Statistics ===\n");
    printf("Registered functions: %d\n", sim->num_functions);
    printf("Total simulations: %lld\n", (long long)sim->total_simulations);
    printf("Total cycles simulated: %lld\n", (long long)sim->total_cycles_simulated);
    
    printf("\nFunction cache:\n");
    for (int i = 0; i < sim->num_functions; i++) {
        IncoreFunction* func = sim->functions[i];
        if (func) {
            printf("  %s: %d instructions, %s, cached=%s",
                   func->name, func->num_instructions,
                   (func->core_type == CORE_TYPE_CUBE) ? "CUBE" : "VECTOR",
                   func->cache_valid ? "yes" : "no");
            if (func->cache_valid) {
                printf(" (%lld cycles)", (long long)func->cached_cycles);
            }
            printf("\n");
        }
    }
    printf("\n");
}

// =============================================================================
// Instruction Parsing
// =============================================================================

bool a2a3_parse_instruction(const char* text, A2A3Instruction* out, CoreType core_type) {
    if (!text || !out) return false;
    
    // Skip empty or comment lines
    if (is_comment_or_empty(text)) return false;
    
    memset(out, 0, sizeof(A2A3Instruction));
    strncpy(out->name, text, A2A3_MAX_INSTR_NAME - 1);
    
    // Decode category and target pipe
    out->category = a2a3_decode_instr_category(text, core_type);
    out->target_pipe = a2a3_get_target_pipe(text, core_type);
    
    // Estimate latency
    out->latency = a2a3_estimate_latency(text, 32 * 128 * 4);  // Default tile size
    
    // Parse sync instructions
    if (strstr(text, "SET_FLAG") || strstr(text, "set_flag")) {
        out->category = INSTR_CAT_SYNC;
        // Try to parse flag_id from "SET_FLAG(N)" format
        const char* p = strchr(text, '(');
        if (p) {
            out->flag_id = atoi(p + 1);
        }
    } else if (strstr(text, "WAIT_FLAG") || strstr(text, "wait_flag")) {
        out->category = INSTR_CAT_SYNC;
        const char* p = strchr(text, '(');
        if (p) {
            out->flag_id = atoi(p + 1);
        }
    }
    
    // Parse MTE transfer size from DataCopy(dst, src, size)
    if (out->category == INSTR_CAT_MTE) {
        const char* p = text;
        int paren_count = 0;
        while (*p) {
            if (*p == '(') paren_count++;
            if (*p == ')') paren_count--;
            if (*p == ',' && paren_count == 1) {
                // Look for size parameter
                const char* size_start = p + 1;
                while (*size_start && isspace((unsigned char)*size_start)) size_start++;
                // Skip second comma if present
                const char* comma2 = strchr(size_start, ',');
                if (comma2) {
                    size_start = comma2 + 1;
                    while (*size_start && isspace((unsigned char)*size_start)) size_start++;
                }
                out->transfer_size = atoi(size_start) * 4;  // Assume elements, convert to bytes
                break;
            }
            p++;
        }
        if (out->transfer_size == 0) {
            out->transfer_size = 32 * 128 * 4;  // Default tile size
        }
    }
    
    return true;
}

int a2a3_parse_code_block(const char* code_block, ParsedInstruction* out,
                          int max_instructions, CoreType core_type) {
    if (!code_block || !out || max_instructions <= 0) return 0;
    
    char* code_copy = strdup(code_block);
    if (!code_copy) return 0;
    
    int count = 0;
    char* line = strtok(code_copy, "\n");
    
    while (line && count < max_instructions) {
        line = trim_whitespace(line);
        if (!is_comment_or_empty(line)) {
            strncpy(out[count].text, line, sizeof(out[count].text) - 1);
            if (a2a3_parse_instruction(line, &out[count].decoded, core_type)) {
                count++;
            }
        }
        line = strtok(NULL, "\n");
    }
    
    free(code_copy);
    return count;
}
