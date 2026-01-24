// =============================================================================
// InCore Function: rope_tile_256
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* rope_tile_256_instructions = 
    "// PTO Program: rope_tile_256\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: rope_tile_256\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     6\n"
    "//   Total capacity (no reuse): 786,432 bytes (768.0 KB)\n"
    "//   Total capacity (w/ reuse): 524,288 bytes (512.0 KB)\n"
    "//   Reuse savings:            262,144 bytes (33.3%)\n"
    "//\n"
    "// ======================================================================\n"
    "\n"
    "// Auto-generated Ascend C code from PTO ISA Compiler\n"
    "// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)\n"
    "#include \"kernel_operator.h\"\n"
    "\n"
    "using namespace AscendC;\n"
    "\n"
    "__aicore__ void rope_tile_256(__gm__ float* input, __gm__ float* cos_cache, __gm__ float* sin_cache, __gm__ float* output) {\n"
    "    __ub__ float x[32768];\n"
    "    __ub__ float cos_pos[32768];\n"
    "    __ub__ float sin_pos[32768];\n"
    "    __ub__ float x_cos[32768];\n"
    "    __ub__ float x_sin[32768];\n"
    "    __ub__ float result[32768];\n"
    "\n"
    "    // Loop fusion: 2 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 32768], 32768);\n"
    "\n"
    "    // TLOAD: cos_pos = load(cos_cache[0, 0])\n"
    "    DataCopy(cos_pos, cos_cache[(0) * 32768], 32768);\n"
    "\n"
    "    // TLOAD: sin_pos = load(sin_cache[0, 0])\n"
    "    DataCopy(sin_pos, sin_cache[(0) * 32768], 32768);\n"
    "\n"
    "    // Fused vector operations: 3 operations\n"
    "    // Tile size: 256x128 = 32768 elements\n"
    "    Mul(x_cos, x, cos_pos, 32768);\n"
    "    Mul(x_sin, x, sin_pos, 32768);\n"
    "    Add(result, x_cos, x_sin, 32768);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int rope_tile_256_sim_registered = 0;

void register_rope_tile_256_sim(IncoreSimulator* sim) {
    if (!rope_tile_256_sim_registered) {
        a2a3_incore_sim_register_code(sim, "rope_tile_256",
            CORE_TYPE_VECTOR, rope_tile_256_instructions, 32, 128);
        rope_tile_256_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t rope_tile_256_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("rope_tile_256", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: rope_tile_256
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: rope_tile_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 786,432 bytes (768.0 KB)
//   Total capacity (w/ reuse): 524,288 bytes (512.0 KB)
//   Reuse savings:            262,144 bytes (33.3%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void rope_tile_256(__gm__ float* input, __gm__ float* cos_cache, __gm__ float* sin_cache, __gm__ float* output) {
    __ub__ float x[32768];
    __ub__ float cos_pos[32768];
    __ub__ float sin_pos[32768];
    __ub__ float x_cos[32768];
    __ub__ float x_sin[32768];
    __ub__ float result[32768];

    // Loop fusion: 2 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 32768], 32768);

    // TLOAD: cos_pos = load(cos_cache[0, 0])
    DataCopy(cos_pos, cos_cache[(0) * 32768], 32768);

    // TLOAD: sin_pos = load(sin_cache[0, 0])
    DataCopy(sin_pos, sin_cache[(0) * 32768], 32768);

    // Fused vector operations: 3 operations
    // Tile size: 256x128 = 32768 elements
    Mul(x_cos, x, cos_pos, 32768);
    Mul(x_sin, x, sin_pos, 32768);
    Add(result, x_cos, x_sin, 32768);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/