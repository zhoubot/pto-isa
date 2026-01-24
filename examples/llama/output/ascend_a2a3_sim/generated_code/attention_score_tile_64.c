// =============================================================================
// InCore Function: attention_score_tile_64
// Core Type: Cube
// =============================================================================

// Instruction code for core simulator parsing
static const char* attention_score_tile_64_instructions = 
    "// PTO Program: attention_score_tile_64\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: attention_score_tile_64\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     4\n"
    "//   Total capacity (no reuse): 163,840 bytes (160.0 KB)\n"
    "//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)\n"
    "//   Reuse savings:            32,768 bytes (20.0%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void attention_score_tile_64(__gm__ float* input_q, __gm__ float* input_kt, __gm__ float* output, float scale) {\n"
    "    __ub__ float q[8192];\n"
    "    __ub__ float k_t[16384];\n"
    "    __ub__ float scores[8192];\n"
    "    __ub__ float scaled_scores[8192];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: q = load(input_q[0, 0])\n"
    "    DataCopy(q, input_q[(0) * 8192], 8192);\n"
    "\n"
    "    // TLOAD: k_t = load(input_kt[0, 0])\n"
    "    DataCopy(k_t, input_kt[(0) * 16384], 16384);\n"
    "\n"
    "    // TMATMUL: scores = q @ k_t\n"
    "    Matmul(scores, q, k_t, 64, 128);\n"
    "\n"
    "    // LI: Not implemented\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Muls(scaled_scores, scores, , 8192);\n"
    "\n"
    "    // TSTORE: store(scaled_scores) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], scaled_scores, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int attention_score_tile_64_sim_registered = 0;

void register_attention_score_tile_64_sim(IncoreSimulator* sim) {
    if (!attention_score_tile_64_sim_registered) {
        a2a3_incore_sim_register_code(sim, "attention_score_tile_64",
            CORE_TYPE_CUBE, attention_score_tile_64_instructions, 32, 128);
        attention_score_tile_64_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t attention_score_tile_64_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("attention_score_tile_64", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: attention_score_tile_64
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: attention_score_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 163,840 bytes (160.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            32,768 bytes (20.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void attention_score_tile_64(__gm__ float* input_q, __gm__ float* input_kt, __gm__ float* output, float scale) {
    __ub__ float q[8192];
    __ub__ float k_t[16384];
    __ub__ float scores[8192];
    __ub__ float scaled_scores[8192];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: q = load(input_q[0, 0])
    DataCopy(q, input_q[(0) * 8192], 8192);

    // TLOAD: k_t = load(input_kt[0, 0])
    DataCopy(k_t, input_kt[(0) * 16384], 16384);

    // TMATMUL: scores = q @ k_t
    Matmul(scores, q, k_t, 64, 128);

    // LI: Not implemented

    // Fused vector operations: 1 operations
    // Tile size: 64x128 = 8192 elements
    Muls(scaled_scores, scores, , 8192);

    // TSTORE: store(scaled_scores) -> output[0, 0]
    DataCopy(output[(0) * 64], scaled_scores, 64);

}
*/