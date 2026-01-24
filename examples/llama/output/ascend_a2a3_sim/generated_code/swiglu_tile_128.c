// =============================================================================
// InCore Function: swiglu_tile_128
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* swiglu_tile_128_instructions = 
    "// PTO Program: swiglu_tile_128\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: swiglu_tile_128\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     8\n"
    "//   Total capacity (no reuse): 524,288 bytes (512.0 KB)\n"
    "//   Total capacity (w/ reuse): 262,144 bytes (256.0 KB)\n"
    "//   Reuse savings:            262,144 bytes (50.0%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void swiglu_tile_128(__gm__ float* input_gate, __gm__ float* input_up, __gm__ float* output) {\n"
    "    __ub__ float gate[16384];\n"
    "    __ub__ float up[16384];\n"
    "    __ub__ float neg_gate[16384];\n"
    "    __ub__ float exp_neg_gate[16384];\n"
    "    __ub__ float one_plus_exp[16384];\n"
    "    __ub__ float sigmoid_gate[16384];\n"
    "    __ub__ float gate_silu[16384];\n"
    "    __ub__ float result[16384];\n"
    "\n"
    "    // Loop fusion: 5 loop overheads saved\n"
    "\n"
    "    // TLOAD: gate = load(input_gate[0, 0])\n"
    "    DataCopy(gate, input_gate[(0) * 16384], 16384);\n"
    "\n"
    "    // TLOAD: up = load(input_up[0, 0])\n"
    "    DataCopy(up, input_up[(0) * 16384], 16384);\n"
    "\n"
    "    // Fused vector operations: 6 operations\n"
    "    // Tile size: 128x128 = 16384 elements\n"
    "    Neg(neg_gate, gate, 16384);\n"
    "    Exp(exp_neg_gate, neg_gate, 16384);\n"
    "    Adds(one_plus_exp, exp_neg_gate, , 16384);\n"
    "    Reciprocal(sigmoid_gate, one_plus_exp, 16384);\n"
    "    Mul(gate_silu, gate, sigmoid_gate, 16384);\n"
    "    Mul(result, gate_silu, up, 16384);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int swiglu_tile_128_sim_registered = 0;

void register_swiglu_tile_128_sim(IncoreSimulator* sim) {
    if (!swiglu_tile_128_sim_registered) {
        a2a3_incore_sim_register_code(sim, "swiglu_tile_128",
            CORE_TYPE_VECTOR, swiglu_tile_128_instructions, 32, 128);
        swiglu_tile_128_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t swiglu_tile_128_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("swiglu_tile_128", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: swiglu_tile_128
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 524,288 bytes (512.0 KB)
//   Total capacity (w/ reuse): 262,144 bytes (256.0 KB)
//   Reuse savings:            262,144 bytes (50.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void swiglu_tile_128(__gm__ float* input_gate, __gm__ float* input_up, __gm__ float* output) {
    __ub__ float gate[16384];
    __ub__ float up[16384];
    __ub__ float neg_gate[16384];
    __ub__ float exp_neg_gate[16384];
    __ub__ float one_plus_exp[16384];
    __ub__ float sigmoid_gate[16384];
    __ub__ float gate_silu[16384];
    __ub__ float result[16384];

    // Loop fusion: 5 loop overheads saved

    // TLOAD: gate = load(input_gate[0, 0])
    DataCopy(gate, input_gate[(0) * 16384], 16384);

    // TLOAD: up = load(input_up[0, 0])
    DataCopy(up, input_up[(0) * 16384], 16384);

    // Fused vector operations: 6 operations
    // Tile size: 128x128 = 16384 elements
    Neg(neg_gate, gate, 16384);
    Exp(exp_neg_gate, neg_gate, 16384);
    Adds(one_plus_exp, exp_neg_gate, , 16384);
    Reciprocal(sigmoid_gate, one_plus_exp, 16384);
    Mul(gate_silu, gate, sigmoid_gate, 16384);
    Mul(result, gate_silu, up, 16384);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/