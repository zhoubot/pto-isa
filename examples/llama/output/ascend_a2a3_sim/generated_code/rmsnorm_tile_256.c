// =============================================================================
// InCore Function: rmsnorm_tile_256
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* rmsnorm_tile_256_instructions = 
    "// PTO Program: rmsnorm_tile_256\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: rmsnorm_tile_256\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     8\n"
    "//   Total capacity (no reuse): 658,432 bytes (643.0 KB)\n"
    "//   Total capacity (w/ reuse): 394,240 bytes (385.0 KB)\n"
    "//   Reuse savings:            264,192 bytes (40.1%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void rmsnorm_tile_256(__gm__ float* input, __gm__ float* weights, __gm__ float* output, float eps, float inv_cols) {\n"
    "    __ub__ float x[32768];\n"
    "    __ub__ float x_sq[32768];\n"
    "    __ub__ float row_sum[256];\n"
    "    __ub__ float row_mean[256];\n"
    "    __ub__ float row_rsqrt[256];\n"
    "    __ub__ float x_norm[32768];\n"
    "    __ub__ float gamma[32768];\n"
    "    __ub__ float result[32768];\n"
    "\n"
    "    // Loop fusion: 1 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 32768], 32768);\n"
    "\n"
    "    // TLOAD: gamma = load(weights[0, 0])\n"
    "    DataCopy(gamma, weights[(0) * 32768], 32768);\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 256x128 = 32768 elements\n"
    "    Mul(x_sq, x, x, 32768);\n"
    "\n"
    "    // TROWSUM: reduction operation\n"
    "    ReduceSum(row_sum, x_sq, 256);\n"
    "\n"
    "    // LI: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 256x1 = 256 elements\n"
    "    Muls(row_mean, row_sum, , 256);\n"
    "\n"
    "    // LI: Not implemented\n"
    "\n"
    "    // Fused vector operations: 2 operations\n"
    "    // Tile size: 256x1 = 256 elements\n"
    "    Adds(row_mean, row_mean, , 256);\n"
    "    Rsqrt(row_rsqrt, row_mean, 256);\n"
    "\n"
    "    // TROWEXPANDMUL: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 256x128 = 32768 elements\n"
    "    Mul(result, x_norm, gamma, 32768);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int rmsnorm_tile_256_sim_registered = 0;

void register_rmsnorm_tile_256_sim(IncoreSimulator* sim) {
    if (!rmsnorm_tile_256_sim_registered) {
        a2a3_incore_sim_register_code(sim, "rmsnorm_tile_256",
            CORE_TYPE_VECTOR, rmsnorm_tile_256_instructions, 32, 128);
        rmsnorm_tile_256_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t rmsnorm_tile_256_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("rmsnorm_tile_256", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: rmsnorm_tile_256
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 658,432 bytes (643.0 KB)
//   Total capacity (w/ reuse): 394,240 bytes (385.0 KB)
//   Reuse savings:            264,192 bytes (40.1%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void rmsnorm_tile_256(__gm__ float* input, __gm__ float* weights, __gm__ float* output, float eps, float inv_cols) {
    __ub__ float x[32768];
    __ub__ float x_sq[32768];
    __ub__ float row_sum[256];
    __ub__ float row_mean[256];
    __ub__ float row_rsqrt[256];
    __ub__ float x_norm[32768];
    __ub__ float gamma[32768];
    __ub__ float result[32768];

    // Loop fusion: 1 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 32768], 32768);

    // TLOAD: gamma = load(weights[0, 0])
    DataCopy(gamma, weights[(0) * 32768], 32768);

    // Fused vector operations: 1 operations
    // Tile size: 256x128 = 32768 elements
    Mul(x_sq, x, x, 32768);

    // TROWSUM: reduction operation
    ReduceSum(row_sum, x_sq, 256);

    // LI: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 256x1 = 256 elements
    Muls(row_mean, row_sum, , 256);

    // LI: Not implemented

    // Fused vector operations: 2 operations
    // Tile size: 256x1 = 256 elements
    Adds(row_mean, row_mean, , 256);
    Rsqrt(row_rsqrt, row_mean, 256);

    // TROWEXPANDMUL: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 256x128 = 32768 elements
    Mul(result, x_norm, gamma, 32768);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/