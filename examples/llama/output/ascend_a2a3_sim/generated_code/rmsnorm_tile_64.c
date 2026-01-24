// =============================================================================
// InCore Function: rmsnorm_tile_64
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* rmsnorm_tile_64_instructions = 
    "// PTO Program: rmsnorm_tile_64\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: rmsnorm_tile_64\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     8\n"
    "//   Total capacity (no reuse): 164,608 bytes (160.8 KB)\n"
    "//   Total capacity (w/ reuse): 98,560 bytes (96.2 KB)\n"
    "//   Reuse savings:            66,048 bytes (40.1%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void rmsnorm_tile_64(__gm__ float* input, __gm__ float* weights, __gm__ float* output, float eps, float inv_cols) {\n"
    "    __ub__ float x[8192];\n"
    "    __ub__ float x_sq[8192];\n"
    "    __ub__ float row_sum[64];\n"
    "    __ub__ float row_mean[64];\n"
    "    __ub__ float row_rsqrt[64];\n"
    "    __ub__ float x_norm[8192];\n"
    "    __ub__ float gamma[8192];\n"
    "    __ub__ float result[8192];\n"
    "\n"
    "    // Loop fusion: 1 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 8192], 8192);\n"
    "\n"
    "    // TLOAD: gamma = load(weights[0, 0])\n"
    "    DataCopy(gamma, weights[(0) * 8192], 8192);\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Mul(x_sq, x, x, 8192);\n"
    "\n"
    "    // TROWSUM: reduction operation\n"
    "    ReduceSum(row_sum, x_sq, 64);\n"
    "\n"
    "    // LI: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 64x1 = 64 elements\n"
    "    Muls(row_mean, row_sum, , 64);\n"
    "\n"
    "    // LI: Not implemented\n"
    "\n"
    "    // Fused vector operations: 2 operations\n"
    "    // Tile size: 64x1 = 64 elements\n"
    "    Adds(row_mean, row_mean, , 64);\n"
    "    Rsqrt(row_rsqrt, row_mean, 64);\n"
    "\n"
    "    // TROWEXPANDMUL: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Mul(result, x_norm, gamma, 8192);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int rmsnorm_tile_64_sim_registered = 0;

void register_rmsnorm_tile_64_sim(IncoreSimulator* sim) {
    if (!rmsnorm_tile_64_sim_registered) {
        a2a3_incore_sim_register_code(sim, "rmsnorm_tile_64",
            CORE_TYPE_VECTOR, rmsnorm_tile_64_instructions, 32, 128);
        rmsnorm_tile_64_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t rmsnorm_tile_64_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("rmsnorm_tile_64", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: rmsnorm_tile_64
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 164,608 bytes (160.8 KB)
//   Total capacity (w/ reuse): 98,560 bytes (96.2 KB)
//   Reuse savings:            66,048 bytes (40.1%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void rmsnorm_tile_64(__gm__ float* input, __gm__ float* weights, __gm__ float* output, float eps, float inv_cols) {
    __ub__ float x[8192];
    __ub__ float x_sq[8192];
    __ub__ float row_sum[64];
    __ub__ float row_mean[64];
    __ub__ float row_rsqrt[64];
    __ub__ float x_norm[8192];
    __ub__ float gamma[8192];
    __ub__ float result[8192];

    // Loop fusion: 1 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 8192], 8192);

    // TLOAD: gamma = load(weights[0, 0])
    DataCopy(gamma, weights[(0) * 8192], 8192);

    // Fused vector operations: 1 operations
    // Tile size: 64x128 = 8192 elements
    Mul(x_sq, x, x, 8192);

    // TROWSUM: reduction operation
    ReduceSum(row_sum, x_sq, 64);

    // LI: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 64x1 = 64 elements
    Muls(row_mean, row_sum, , 64);

    // LI: Not implemented

    // Fused vector operations: 2 operations
    // Tile size: 64x1 = 64 elements
    Adds(row_mean, row_mean, , 64);
    Rsqrt(row_rsqrt, row_mean, 64);

    // TROWEXPANDMUL: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 64x128 = 8192 elements
    Mul(result, x_norm, gamma, 8192);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/