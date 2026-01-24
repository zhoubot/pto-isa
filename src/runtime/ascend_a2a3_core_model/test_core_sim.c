/**
 * Test program for A2A3 Core Simulator
 */

#include <stdio.h>
#include <stdlib.h>
#include "a2a3_core_model.h"
#include "a2a3_incore_sim.h"

// Example RMSNorm tile function instructions
const char* rmsnorm_instructions[] = {
    "// Load input tile from GM to UB",
    "DataCopy(x, input, 4096);",
    "// Compute x^2",
    "Mul(x_sq, x, x, 4096);",
    "// Row sum of x^2",
    "ReduceSum(row_sum, x_sq, 4096);",
    "// Divide by cols and add epsilon",
    "Divs(mean_sq, row_sum, 128.0, 32);",
    "Adds(mean_sq_eps, mean_sq, 1e-6, 32);",
    "// rsqrt",
    "Rsqrt(norm_factor, mean_sq_eps, 32);",
    "// Normalize: x * norm_factor (broadcast)",
    "BroadcastMul(x_norm, x, norm_factor, 4096, 8);",
    "// Load weights and multiply",
    "DataCopy(w, weights, 4096);",
    "Mul(result, x_norm, w, 4096);",
    "// Store result",
    "DataCopy(output, result, 4096);",
    "pipe_barrier();",
    NULL
};

// Example MatMul tile function instructions
const char* matmul_instructions[] = {
    "// Load A tile from GM to L1",
    "DataCopy(a_l1, input_a, 16384);",
    "SET_FLAG(0);",
    "// Load B tile from GM to L1",
    "DataCopy(b_l1, input_b, 16384);",
    "SET_FLAG(1);",
    "// Wait for loads",
    "WAIT_FLAG(0);",
    "WAIT_FLAG(1);",
    "// Move to L0",
    "DataCopy(a_l0a, a_l1, 16384);",
    "DataCopy(b_l0b, b_l1, 16384);",
    "// Matrix multiply",
    "Matmul(c_l0c, a_l0a, b_l0b, 64, 64);",
    "// Move result to L1",
    "DataCopy(c_l1, c_l0c, 16384);",
    "// Store to GM",
    "DataCopy(output, c_l1, 16384);",
    "pipe_barrier();",
    NULL
};

void test_core_model() {
    printf("=== Testing Core Model ===\n\n");
    
    // Create a vector core
    A2A3Core* vec_core = a2a3_core_create(CORE_TYPE_VECTOR, 0);
    if (!vec_core) {
        fprintf(stderr, "Failed to create vector core\n");
        return;
    }
    
    // Create a cube core
    A2A3Core* cube_core = a2a3_core_create(CORE_TYPE_CUBE, 1);
    if (!cube_core) {
        fprintf(stderr, "Failed to create cube core\n");
        a2a3_core_destroy(vec_core);
        return;
    }
    
    // Test vector core with some instructions
    printf("Testing Vector Core:\n");
    a2a3_core_enable_trace(vec_core, stdout);
    
    A2A3Instruction instr;
    
    // MTE: Load from GM to UB
    strcpy(instr.name, "DataCopy(x, input, 4096)");
    instr.category = INSTR_CAT_MTE;
    instr.target_pipe = VEC_PIPE_MTE_GM2UB;
    instr.latency = MTE_GM2UB_LATENCY;
    instr.transfer_size = 4096 * 4;
    a2a3_core_execute(vec_core, &instr);
    
    // Vector: Multiply
    strcpy(instr.name, "Mul(x_sq, x, x, 4096)");
    instr.category = INSTR_CAT_VECTOR;
    instr.target_pipe = VEC_PIPE_VECTOR;
    instr.latency = VEC_BINARY_LATENCY;
    a2a3_core_execute(vec_core, &instr);
    
    // Barrier
    a2a3_core_pipe_barrier(vec_core);
    
    a2a3_core_disable_trace(vec_core);
    a2a3_core_print_stats(vec_core);
    
    // Test cube core
    printf("\nTesting Cube Core:\n");
    a2a3_core_enable_trace(cube_core, stdout);
    
    // MTE: Load A
    strcpy(instr.name, "DataCopy(a_l1, input_a, 16384)");
    instr.category = INSTR_CAT_MTE;
    instr.target_pipe = CUBE_PIPE_MTE_GM2L1;
    instr.latency = MTE_GM2L1_LATENCY;
    instr.transfer_size = 16384 * 4;
    a2a3_core_execute(cube_core, &instr);
    
    // Cube: MatMul
    strcpy(instr.name, "Matmul(c, a, b, 64, 64)");
    instr.category = INSTR_CAT_CUBE;
    instr.target_pipe = CUBE_PIPE_CUBE;
    instr.latency = CUBE_MATMUL_LATENCY;
    a2a3_core_execute(cube_core, &instr);
    
    // Barrier
    a2a3_core_pipe_barrier(cube_core);
    
    a2a3_core_disable_trace(cube_core);
    a2a3_core_print_stats(cube_core);
    
    // Cleanup
    a2a3_core_destroy(vec_core);
    a2a3_core_destroy(cube_core);
}

void test_incore_simulator() {
    printf("\n=== Testing InCore Simulator ===\n\n");
    
    // Create simulator
    IncoreSimulator* sim = a2a3_incore_sim_create();
    if (!sim) {
        fprintf(stderr, "Failed to create InCore simulator\n");
        return;
    }
    
    // Count instructions
    int rmsnorm_count = 0;
    while (rmsnorm_instructions[rmsnorm_count]) rmsnorm_count++;
    
    int matmul_count = 0;
    while (matmul_instructions[matmul_count]) matmul_count++;
    
    // Register RMSNorm function
    int rmsnorm_id = a2a3_incore_sim_register(sim, "rmsnorm_tile", 
                                               CORE_TYPE_VECTOR,
                                               rmsnorm_instructions, rmsnorm_count,
                                               32, 128);
    printf("Registered rmsnorm_tile: id=%d\n", rmsnorm_id);
    
    // Register MatMul function
    int matmul_id = a2a3_incore_sim_register(sim, "tile_matmul",
                                              CORE_TYPE_CUBE,
                                              matmul_instructions, matmul_count,
                                              64, 64);
    printf("Registered tile_matmul: id=%d\n", matmul_id);
    
    // Simulate RMSNorm
    printf("\nSimulating rmsnorm_tile:\n");
    a2a3_incore_sim_enable_trace(sim, stdout);
    int64_t rmsnorm_cycles = a2a3_incore_sim_execute(sim, rmsnorm_id);
    printf("rmsnorm_tile: %lld cycles\n", (long long)rmsnorm_cycles);
    
    // Simulate MatMul
    printf("\nSimulating tile_matmul:\n");
    int64_t matmul_cycles = a2a3_incore_sim_execute(sim, matmul_id);
    printf("tile_matmul: %lld cycles\n", (long long)matmul_cycles);
    
    a2a3_incore_sim_disable_trace(sim);
    
    // Print statistics
    a2a3_incore_sim_print_stats(sim);
    
    // Test heuristic estimates
    printf("\n=== Heuristic Cycle Estimates ===\n");
    printf("rmsnorm_tile: %lld cycles\n", 
           (long long)a2a3_get_incore_cycle_cost("rmsnorm_tile", 32*128));
    printf("tile_matmul: %lld cycles\n",
           (long long)a2a3_get_incore_cycle_cost("tile_matmul", 64*64));
    printf("softmax_tile: %lld cycles\n",
           (long long)a2a3_get_incore_cycle_cost("softmax_tile", 32*128));
    printf("rope_tile: %lld cycles\n",
           (long long)a2a3_get_incore_cycle_cost("rope_tile", 32*128));
    printf("swiglu_tile: %lld cycles\n",
           (long long)a2a3_get_incore_cycle_cost("swiglu_tile", 32*128));
    
    // Cleanup
    a2a3_incore_sim_destroy(sim);
}

int main() {
    printf("========================================\n");
    printf("Ascend A2/A3 Core Simulator Test Suite\n");
    printf("========================================\n\n");
    
    test_core_model();
    test_incore_simulator();
    
    printf("\n========================================\n");
    printf("All tests completed!\n");
    printf("========================================\n");
    
    return 0;
}
