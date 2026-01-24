// =============================================================================
// InCore Function: rmsnorm_tile_128
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* rmsnorm_tile_128_instructions = 
    "// PTO Program: rmsnorm_tile_128\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: rmsnorm_tile_128\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     8\n"
    "//   Total capacity (no reuse): 329,216 bytes (321.5 KB)\n"
    "//   Total capacity (w/ reuse): 197,120 bytes (192.5 KB)\n"
    "//   Reuse savings:            132,096 bytes (40.1%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void rmsnorm_tile_128(__gm__ float* input, __gm__ float* weights, __gm__ float* output, float eps, float inv_cols) {\n"
    "    __ub__ float x[16384];\n"
    "    __ub__ float x_sq[16384];\n"
    "    __ub__ float row_sum[128];\n"
    "    __ub__ float row_mean[128];\n"
    "    __ub__ float row_rsqrt[128];\n"
    "    __ub__ float x_norm[16384];\n"
    "    __ub__ float gamma[16384];\n"
    "    __ub__ float result[16384];\n"
    "\n"
    "    // Loop fusion: 1 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 16384], 16384);\n"
    "\n"
    "    // TLOAD: gamma = load(weights[0, 0])\n"
    "    DataCopy(gamma, weights[(0) * 16384], 16384);\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 128x128 = 16384 elements\n"
    "    Mul(x_sq, x, x, 16384);\n"
    "\n"
    "    // TROWSUM: reduction operation\n"
    "    ReduceSum(row_sum, x_sq, 128);\n"
    "\n"
    "    // LI: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 128x1 = 128 elements\n"
    "    Muls(row_mean, row_sum, , 128);\n"
    "\n"
    "    // LI: Not implemented\n"
    "\n"
    "    // Fused vector operations: 2 operations\n"
    "    // Tile size: 128x1 = 128 elements\n"
    "    Adds(row_mean, row_mean, , 128);\n"
    "    Rsqrt(row_rsqrt, row_mean, 128);\n"
    "\n"
    "    // TROWEXPANDMUL: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 128x128 = 16384 elements\n"
    "    Mul(result, x_norm, gamma, 16384);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int rmsnorm_tile_128_sim_registered = 0;

void register_rmsnorm_tile_128_sim(IncoreSimulator* sim) {
    if (!rmsnorm_tile_128_sim_registered) {
        a2a3_incore_sim_register_code(sim, "rmsnorm_tile_128",
            CORE_TYPE_VECTOR, rmsnorm_tile_128_instructions, 32, 128);
        rmsnorm_tile_128_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t rmsnorm_tile_128_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("rmsnorm_tile_128", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: rmsnorm_tile_128
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 329,216 bytes (321.5 KB)
//   Total capacity (w/ reuse): 197,120 bytes (192.5 KB)
//   Reuse savings:            132,096 bytes (40.1%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void rmsnorm_tile_128(__gm__ float* input, __gm__ float* weights, __gm__ float* output, float eps, float inv_cols) {
    __ub__ float x[16384];
    __ub__ float x_sq[16384];
    __ub__ float row_sum[128];
    __ub__ float row_mean[128];
    __ub__ float row_rsqrt[128];
    __ub__ float x_norm[16384];
    __ub__ float gamma[16384];
    __ub__ float result[16384];

    // Loop fusion: 1 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 16384], 16384);

    // TLOAD: gamma = load(weights[0, 0])
    DataCopy(gamma, weights[(0) * 16384], 16384);

    // Fused vector operations: 1 operations
    // Tile size: 128x128 = 16384 elements
    Mul(x_sq, x, x, 16384);

    // TROWSUM: reduction operation
    ReduceSum(row_sum, x_sq, 128);

    // LI: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 128x1 = 128 elements
    Muls(row_mean, row_sum, , 128);

    // LI: Not implemented

    // Fused vector operations: 2 operations
    // Tile size: 128x1 = 128 elements
    Adds(row_mean, row_mean, , 128);
    Rsqrt(row_rsqrt, row_mean, 128);

    // TROWEXPANDMUL: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 128x128 = 16384 elements
    Mul(result, x_norm, gamma, 16384);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/