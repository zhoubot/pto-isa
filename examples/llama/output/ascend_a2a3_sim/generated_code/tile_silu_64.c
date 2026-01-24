// =============================================================================
// InCore Function: tile_silu_64
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_silu_64_instructions = 
    "// PTO Program: tile_silu_64\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_silu_64\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     6\n"
    "//   Total capacity (no reuse): 196,608 bytes (192.0 KB)\n"
    "//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)\n"
    "//   Reuse savings:            98,304 bytes (50.0%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void tile_silu_64(__gm__ float* input, __gm__ float* output) {\n"
    "    __ub__ float x[8192];\n"
    "    __ub__ float neg_x[8192];\n"
    "    __ub__ float exp_neg_x[8192];\n"
    "    __ub__ float one_plus_exp[8192];\n"
    "    __ub__ float sigmoid[8192];\n"
    "    __ub__ float result[8192];\n"
    "\n"
    "    // Loop fusion: 4 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 8192], 8192);\n"
    "\n"
    "    // Fused vector operations: 5 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Neg(neg_x, x, 8192);\n"
    "    Exp(exp_neg_x, neg_x, 8192);\n"
    "    Adds(one_plus_exp, exp_neg_x, , 8192);\n"
    "    Reciprocal(sigmoid, one_plus_exp, 8192);\n"
    "    Mul(result, x, sigmoid, 8192);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_silu_64_sim_registered = 0;

void register_tile_silu_64_sim(IncoreSimulator* sim) {
    if (!tile_silu_64_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_silu_64",
            CORE_TYPE_VECTOR, tile_silu_64_instructions, 32, 128);
        tile_silu_64_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_silu_64_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_silu_64", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_silu_64
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_silu_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            98,304 bytes (50.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void tile_silu_64(__gm__ float* input, __gm__ float* output) {
    __ub__ float x[8192];
    __ub__ float neg_x[8192];
    __ub__ float exp_neg_x[8192];
    __ub__ float one_plus_exp[8192];
    __ub__ float sigmoid[8192];
    __ub__ float result[8192];

    // Loop fusion: 4 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 8192], 8192);

    // Fused vector operations: 5 operations
    // Tile size: 64x128 = 8192 elements
    Neg(neg_x, x, 8192);
    Exp(exp_neg_x, neg_x, 8192);
    Adds(one_plus_exp, exp_neg_x, , 8192);
    Reciprocal(sigmoid, one_plus_exp, 8192);
    Mul(result, x, sigmoid, 8192);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/