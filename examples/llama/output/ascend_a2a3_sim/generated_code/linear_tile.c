// =============================================================================
// InCore Function: linear_tile
// Core Type: Cube
// =============================================================================

// Instruction code for core simulator parsing
static const char* linear_tile_instructions = 
    "// PTO Program: linear_tile\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: linear_tile\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     3\n"
    "//   Total capacity (no reuse): 98,304 bytes (96.0 KB)\n"
    "//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)\n"
    "//   Reuse savings:            0 bytes (0.0%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void linear_tile(__gm__ float* input, __gm__ float* weight, __gm__ float* output) {\n"
    "    __ub__ float x[4096];\n"
    "    __ub__ float w[16384];\n"
    "    __ub__ float result[4096];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 4096], 4096);\n"
    "\n"
    "    // TLOAD: w = load(weight[0, 0])\n"
    "    DataCopy(w, weight[(0) * 16384], 16384);\n"
    "\n"
    "    // TMATMUL: result = x @ w\n"
    "    Matmul(result, x, w, 32, 128);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int linear_tile_sim_registered = 0;

void register_linear_tile_sim(IncoreSimulator* sim) {
    if (!linear_tile_sim_registered) {
        a2a3_incore_sim_register_code(sim, "linear_tile",
            CORE_TYPE_CUBE, linear_tile_instructions, 32, 128);
        linear_tile_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t linear_tile_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("linear_tile", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: linear_tile
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: linear_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void linear_tile(__gm__ float* input, __gm__ float* weight, __gm__ float* output) {
    __ub__ float x[4096];
    __ub__ float w[16384];
    __ub__ float result[4096];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 4096], 4096);

    // TLOAD: w = load(weight[0, 0])
    DataCopy(w, weight[(0) * 16384], 16384);

    // TMATMUL: result = x @ w
    Matmul(result, x, w, 32, 128);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/