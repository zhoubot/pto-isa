/**
 * Ascend A2/A3 InCore Function Simulator
 * 
 * This module provides simulation of InCore functions using the A2A3 core model.
 * It parses generated Ascend instructions and simulates their execution to
 * estimate cycle counts.
 * 
 * Usage:
 * 1. Create an InCore simulator context
 * 2. Load the instruction stream (from generated code)
 * 3. Simulate execution to get cycle count
 * 4. Use cycle count for task scheduling in orchestration runtime
 */

#ifndef A2A3_INCORE_SIM_H
#define A2A3_INCORE_SIM_H

#include "a2a3_core_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Configuration
// =============================================================================

#define MAX_INCORE_INSTRUCTIONS     1024    // Max instructions per InCore function
#define MAX_INCORE_NAME             64      // Max function name length

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Parsed instruction for simulation
 */
typedef struct {
    char text[128];             // Original instruction text
    A2A3Instruction decoded;    // Decoded instruction for core model
} ParsedInstruction;

/**
 * InCore function representation
 */
typedef struct {
    char name[MAX_INCORE_NAME];
    CoreType core_type;         // CUBE or VECTOR
    
    // Instructions
    ParsedInstruction instructions[MAX_INCORE_INSTRUCTIONS];
    int num_instructions;
    
    // Tile configuration
    int tile_rows;
    int tile_cols;
    int element_size;           // Bytes per element
    
    // Cached simulation result
    int64_t cached_cycles;
    bool cache_valid;
} IncoreFunction;

/**
 * InCore simulator context
 */
typedef struct {
    // Core models (reused across functions)
    A2A3Core* cube_core;
    A2A3Core* vector_core;
    
    // Function registry
    IncoreFunction** functions;
    int num_functions;
    int capacity;
    
    // Statistics
    int64_t total_simulations;
    int64_t total_cycles_simulated;
    
    // Trace control
    bool trace_enabled;
    FILE* trace_file;
} IncoreSimulator;

// =============================================================================
// Simulator Lifecycle
// =============================================================================

/**
 * Create a new InCore simulator
 */
IncoreSimulator* a2a3_incore_sim_create(void);

/**
 * Destroy the simulator and free resources
 */
void a2a3_incore_sim_destroy(IncoreSimulator* sim);

/**
 * Reset simulator state (clears function cache)
 */
void a2a3_incore_sim_reset(IncoreSimulator* sim);

// =============================================================================
// Function Management
// =============================================================================

/**
 * Register an InCore function from instruction text
 * @param sim Simulator context
 * @param name Function name
 * @param core_type CORE_TYPE_CUBE or CORE_TYPE_VECTOR
 * @param instructions Array of instruction strings
 * @param num_instructions Number of instructions
 * @param tile_rows Tile row dimension
 * @param tile_cols Tile column dimension
 * @return Function ID (>= 0) or -1 on error
 */
int a2a3_incore_sim_register(IncoreSimulator* sim, const char* name,
                             CoreType core_type, const char** instructions,
                             int num_instructions, int tile_rows, int tile_cols);

/**
 * Register an InCore function from a single code block
 * Parses newline-separated instructions
 */
int a2a3_incore_sim_register_code(IncoreSimulator* sim, const char* name,
                                  CoreType core_type, const char* code_block,
                                  int tile_rows, int tile_cols);

/**
 * Find a function by name
 * @return Function ID or -1 if not found
 */
int a2a3_incore_sim_find(IncoreSimulator* sim, const char* name);

// =============================================================================
// Simulation API
// =============================================================================

/**
 * Simulate an InCore function execution
 * @param sim Simulator context
 * @param func_id Function ID from register/find
 * @return Cycle count for execution
 */
int64_t a2a3_incore_sim_execute(IncoreSimulator* sim, int func_id);

/**
 * Simulate an InCore function by name
 * Uses cached result if available
 */
int64_t a2a3_incore_sim_execute_by_name(IncoreSimulator* sim, const char* name);

/**
 * Get estimated cycle count for a function (without full simulation)
 * Uses heuristics based on instruction count and types
 */
int64_t a2a3_incore_sim_estimate(IncoreSimulator* sim, int func_id);

// =============================================================================
// Cycle Cost Query (for runtime integration)
// =============================================================================

/**
 * Get cycle cost for a named InCore function
 * This is the main API for runtime task scheduling
 * 
 * @param func_name InCore function name
 * @param tile_size Number of elements in tile (rows * cols)
 * @return Estimated cycle count
 */
int64_t a2a3_get_incore_cycle_cost(const char* func_name, int64_t tile_size);

// =============================================================================
// Tracing
// =============================================================================

/**
 * Enable simulation tracing
 */
void a2a3_incore_sim_enable_trace(IncoreSimulator* sim, FILE* trace_file);

/**
 * Disable simulation tracing
 */
void a2a3_incore_sim_disable_trace(IncoreSimulator* sim);

/**
 * Print simulator statistics
 */
void a2a3_incore_sim_print_stats(const IncoreSimulator* sim);

// =============================================================================
// Instruction Parsing
// =============================================================================

/**
 * Parse an instruction string into decoded form
 */
bool a2a3_parse_instruction(const char* text, A2A3Instruction* out, CoreType core_type);

/**
 * Parse a code block into instructions
 * @return Number of instructions parsed
 */
int a2a3_parse_code_block(const char* code_block, ParsedInstruction* out,
                          int max_instructions, CoreType core_type);

#ifdef __cplusplus
}
#endif

#endif // A2A3_INCORE_SIM_H
