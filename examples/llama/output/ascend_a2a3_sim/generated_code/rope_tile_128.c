// =============================================================================
// InCore Function: rope_tile_128
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* rope_tile_128_instructions = 
    "// PTO Program: rope_tile_128\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: rope_tile_128\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     6\n"
    "//   Total capacity (no reuse): 393,216 bytes (384.0 KB)\n"
    "//   Total capacity (w/ reuse): 262,144 bytes (256.0 KB)\n"
    "//   Reuse savings:            131,072 bytes (33.3%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void rope_tile_128(__gm__ float* input, __gm__ float* cos_cache, __gm__ float* sin_cache, __gm__ float* output) {\n"
    "    __ub__ float x[16384];\n"
    "    __ub__ float cos_pos[16384];\n"
    "    __ub__ float sin_pos[16384];\n"
    "    __ub__ float x_cos[16384];\n"
    "    __ub__ float x_sin[16384];\n"
    "    __ub__ float result[16384];\n"
    "\n"
    "    // Loop fusion: 2 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 16384], 16384);\n"
    "\n"
    "    // TLOAD: cos_pos = load(cos_cache[0, 0])\n"
    "    DataCopy(cos_pos, cos_cache[(0) * 16384], 16384);\n"
    "\n"
    "    // TLOAD: sin_pos = load(sin_cache[0, 0])\n"
    "    DataCopy(sin_pos, sin_cache[(0) * 16384], 16384);\n"
    "\n"
    "    // Fused vector operations: 3 operations\n"
    "    // Tile size: 128x128 = 16384 elements\n"
    "    Mul(x_cos, x, cos_pos, 16384);\n"
    "    Mul(x_sin, x, sin_pos, 16384);\n"
    "    Add(result, x_cos, x_sin, 16384);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int rope_tile_128_sim_registered = 0;

void register_rope_tile_128_sim(IncoreSimulator* sim) {
    if (!rope_tile_128_sim_registered) {
        a2a3_incore_sim_register_code(sim, "rope_tile_128",
            CORE_TYPE_VECTOR, rope_tile_128_instructions, 32, 128);
        rope_tile_128_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t rope_tile_128_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("rope_tile_128", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: rope_tile_128
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: rope_tile_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 393,216 bytes (384.0 KB)
//   Total capacity (w/ reuse): 262,144 bytes (256.0 KB)
//   Reuse savings:            131,072 bytes (33.3%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void rope_tile_128(__gm__ float* input, __gm__ float* cos_cache, __gm__ float* sin_cache, __gm__ float* output) {
    __ub__ float x[16384];
    __ub__ float cos_pos[16384];
    __ub__ float sin_pos[16384];
    __ub__ float x_cos[16384];
    __ub__ float x_sin[16384];
    __ub__ float result[16384];

    // Loop fusion: 2 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 16384], 16384);

    // TLOAD: cos_pos = load(cos_cache[0, 0])
    DataCopy(cos_pos, cos_cache[(0) * 16384], 16384);

    // TLOAD: sin_pos = load(sin_cache[0, 0])
    DataCopy(sin_pos, sin_cache[(0) * 16384], 16384);

    // Fused vector operations: 3 operations
    // Tile size: 128x128 = 16384 elements
    Mul(x_cos, x, cos_pos, 16384);
    Mul(x_sin, x, sin_pos, 16384);
    Add(result, x_cos, x_sin, 16384);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/