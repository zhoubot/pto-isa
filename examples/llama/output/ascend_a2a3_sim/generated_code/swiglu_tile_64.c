// =============================================================================
// InCore Function: swiglu_tile_64
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* swiglu_tile_64_instructions = 
    "// PTO Program: swiglu_tile_64\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: swiglu_tile_64\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     8\n"
    "//   Total capacity (no reuse): 262,144 bytes (256.0 KB)\n"
    "//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)\n"
    "//   Reuse savings:            131,072 bytes (50.0%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void swiglu_tile_64(__gm__ float* input_gate, __gm__ float* input_up, __gm__ float* output) {\n"
    "    __ub__ float gate[8192];\n"
    "    __ub__ float up[8192];\n"
    "    __ub__ float neg_gate[8192];\n"
    "    __ub__ float exp_neg_gate[8192];\n"
    "    __ub__ float one_plus_exp[8192];\n"
    "    __ub__ float sigmoid_gate[8192];\n"
    "    __ub__ float gate_silu[8192];\n"
    "    __ub__ float result[8192];\n"
    "\n"
    "    // Loop fusion: 5 loop overheads saved\n"
    "\n"
    "    // TLOAD: gate = load(input_gate[0, 0])\n"
    "    DataCopy(gate, input_gate[(0) * 8192], 8192);\n"
    "\n"
    "    // TLOAD: up = load(input_up[0, 0])\n"
    "    DataCopy(up, input_up[(0) * 8192], 8192);\n"
    "\n"
    "    // Fused vector operations: 6 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Neg(neg_gate, gate, 8192);\n"
    "    Exp(exp_neg_gate, neg_gate, 8192);\n"
    "    Adds(one_plus_exp, exp_neg_gate, , 8192);\n"
    "    Reciprocal(sigmoid_gate, one_plus_exp, 8192);\n"
    "    Mul(gate_silu, gate, sigmoid_gate, 8192);\n"
    "    Mul(result, gate_silu, up, 8192);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int swiglu_tile_64_sim_registered = 0;

void register_swiglu_tile_64_sim(IncoreSimulator* sim) {
    if (!swiglu_tile_64_sim_registered) {
        a2a3_incore_sim_register_code(sim, "swiglu_tile_64",
            CORE_TYPE_VECTOR, swiglu_tile_64_instructions, 32, 128);
        swiglu_tile_64_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t swiglu_tile_64_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("swiglu_tile_64", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: swiglu_tile_64
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 262,144 bytes (256.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            131,072 bytes (50.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void swiglu_tile_64(__gm__ float* input_gate, __gm__ float* input_up, __gm__ float* output) {
    __ub__ float gate[8192];
    __ub__ float up[8192];
    __ub__ float neg_gate[8192];
    __ub__ float exp_neg_gate[8192];
    __ub__ float one_plus_exp[8192];
    __ub__ float sigmoid_gate[8192];
    __ub__ float gate_silu[8192];
    __ub__ float result[8192];

    // Loop fusion: 5 loop overheads saved

    // TLOAD: gate = load(input_gate[0, 0])
    DataCopy(gate, input_gate[(0) * 8192], 8192);

    // TLOAD: up = load(input_up[0, 0])
    DataCopy(up, input_up[(0) * 8192], 8192);

    // Fused vector operations: 6 operations
    // Tile size: 64x128 = 8192 elements
    Neg(neg_gate, gate, 8192);
    Exp(exp_neg_gate, neg_gate, 8192);
    Adds(one_plus_exp, exp_neg_gate, , 8192);
    Reciprocal(sigmoid_gate, one_plus_exp, 8192);
    Mul(gate_silu, gate, sigmoid_gate, 8192);
    Mul(result, gate_silu, up, 8192);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/