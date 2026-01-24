// =============================================================================
// InCore Function: softmax_tile_64
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* softmax_tile_64_instructions = 
    "// PTO Program: softmax_tile_64\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: softmax_tile_64\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     6\n"
    "//   Total capacity (no reuse): 131,584 bytes (128.5 KB)\n"
    "//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)\n"
    "//   Reuse savings:            65,792 bytes (50.0%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void softmax_tile_64(__gm__ float* input, __gm__ float* output) {\n"
    "    __ub__ float x[8192];\n"
    "    __ub__ float row_max[64];\n"
    "    __ub__ float x_shifted[8192];\n"
    "    __ub__ float exp_x[8192];\n"
    "    __ub__ float row_sum[64];\n"
    "    __ub__ float result[8192];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 8192], 8192);\n"
    "\n"
    "    // TROWMAX: reduction max operation\n"
    "    ReduceMax(row_max, x, 64);\n"
    "\n"
    "    // TROWEXPANDSUB: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Exp(exp_x, x_shifted, 8192);\n"
    "\n"
    "    // TROWSUM: reduction operation\n"
    "    ReduceSum(row_sum, exp_x, 64);\n"
    "\n"
    "    // TROWEXPANDDIV: Not implemented\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int softmax_tile_64_sim_registered = 0;

void register_softmax_tile_64_sim(IncoreSimulator* sim) {
    if (!softmax_tile_64_sim_registered) {
        a2a3_incore_sim_register_code(sim, "softmax_tile_64",
            CORE_TYPE_VECTOR, softmax_tile_64_instructions, 32, 128);
        softmax_tile_64_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t softmax_tile_64_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("softmax_tile_64", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: softmax_tile_64
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: softmax_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 131,584 bytes (128.5 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            65,792 bytes (50.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void softmax_tile_64(__gm__ float* input, __gm__ float* output) {
    __ub__ float x[8192];
    __ub__ float row_max[64];
    __ub__ float x_shifted[8192];
    __ub__ float exp_x[8192];
    __ub__ float row_sum[64];
    __ub__ float result[8192];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 8192], 8192);

    // TROWMAX: reduction max operation
    ReduceMax(row_max, x, 64);

    // TROWEXPANDSUB: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 64x128 = 8192 elements
    Exp(exp_x, x_shifted, 8192);

    // TROWSUM: reduction operation
    ReduceSum(row_sum, exp_x, 64);

    // TROWEXPANDDIV: Not implemented

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/