// =============================================================================
// InCore Function: attention_output_tile_64
// Core Type: Cube
// =============================================================================

// Instruction code for core simulator parsing
static const char* attention_output_tile_64_instructions = 
    "// PTO Program: attention_output_tile_64\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: attention_output_tile_64\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     3\n"
    "//   Total capacity (no reuse): 131,072 bytes (128.0 KB)\n"
    "//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)\n"
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
    "__aicore__ void attention_output_tile_64(__gm__ float* input_weights, __gm__ float* input_v, __gm__ float* output) {\n"
    "    __ub__ float weights[8192];\n"
    "    __ub__ float v[16384];\n"
    "    __ub__ float result[8192];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: weights = load(input_weights[0, 0])\n"
    "    DataCopy(weights, input_weights[(0) * 8192], 8192);\n"
    "\n"
    "    // TLOAD: v = load(input_v[0, 0])\n"
    "    DataCopy(v, input_v[(0) * 16384], 16384);\n"
    "\n"
    "    // TMATMUL: result = weights @ v\n"
    "    Matmul(result, weights, v, 64, 128);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int attention_output_tile_64_sim_registered = 0;

void register_attention_output_tile_64_sim(IncoreSimulator* sim) {
    if (!attention_output_tile_64_sim_registered) {
        a2a3_incore_sim_register_code(sim, "attention_output_tile_64",
            CORE_TYPE_CUBE, attention_output_tile_64_instructions, 32, 128);
        attention_output_tile_64_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t attention_output_tile_64_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("attention_output_tile_64", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: attention_output_tile_64
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: attention_output_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void attention_output_tile_64(__gm__ float* input_weights, __gm__ float* input_v, __gm__ float* output) {
    __ub__ float weights[8192];
    __ub__ float v[16384];
    __ub__ float result[8192];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: weights = load(input_weights[0, 0])
    DataCopy(weights, input_weights[(0) * 8192], 8192);

    // TLOAD: v = load(input_v[0, 0])
    DataCopy(v, input_v[(0) * 16384], 16384);

    // TMATMUL: result = weights @ v
    Matmul(result, weights, v, 64, 128);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/