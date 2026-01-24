// =============================================================================
// InCore Function: tile_silu_256
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_silu_256_instructions = 
    "// PTO Program: tile_silu_256\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_silu_256\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     6\n"
    "//   Total capacity (no reuse): 786,432 bytes (768.0 KB)\n"
    "//   Total capacity (w/ reuse): 393,216 bytes (384.0 KB)\n"
    "//   Reuse savings:            393,216 bytes (50.0%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void tile_silu_256(__gm__ float* input, __gm__ float* output) {\n"
    "    __ub__ float x[32768];\n"
    "    __ub__ float neg_x[32768];\n"
    "    __ub__ float exp_neg_x[32768];\n"
    "    __ub__ float one_plus_exp[32768];\n"
    "    __ub__ float sigmoid[32768];\n"
    "    __ub__ float result[32768];\n"
    "\n"
    "    // Loop fusion: 4 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 32768], 32768);\n"
    "\n"
    "    // Fused vector operations: 5 operations\n"
    "    // Tile size: 256x128 = 32768 elements\n"
    "    Neg(neg_x, x, 32768);\n"
    "    Exp(exp_neg_x, neg_x, 32768);\n"
    "    Adds(one_plus_exp, exp_neg_x, , 32768);\n"
    "    Reciprocal(sigmoid, one_plus_exp, 32768);\n"
    "    Mul(result, x, sigmoid, 32768);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_silu_256_sim_registered = 0;

void register_tile_silu_256_sim(IncoreSimulator* sim) {
    if (!tile_silu_256_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_silu_256",
            CORE_TYPE_VECTOR, tile_silu_256_instructions, 32, 128);
        tile_silu_256_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_silu_256_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_silu_256", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_silu_256
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_silu_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 786,432 bytes (768.0 KB)
//   Total capacity (w/ reuse): 393,216 bytes (384.0 KB)
//   Reuse savings:            393,216 bytes (50.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void tile_silu_256(__gm__ float* input, __gm__ float* output) {
    __ub__ float x[32768];
    __ub__ float neg_x[32768];
    __ub__ float exp_neg_x[32768];
    __ub__ float one_plus_exp[32768];
    __ub__ float sigmoid[32768];
    __ub__ float result[32768];

    // Loop fusion: 4 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 32768], 32768);

    // Fused vector operations: 5 operations
    // Tile size: 256x128 = 32768 elements
    Neg(neg_x, x, 32768);
    Exp(exp_neg_x, neg_x, 32768);
    Adds(one_plus_exp, exp_neg_x, , 32768);
    Reciprocal(sigmoid, one_plus_exp, 32768);
    Mul(result, x, sigmoid, 32768);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/