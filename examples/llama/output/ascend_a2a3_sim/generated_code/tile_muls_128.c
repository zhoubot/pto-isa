// =============================================================================
// InCore Function: tile_muls_128
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_muls_128_instructions = 
    "// PTO Program: tile_muls_128\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_muls_128\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     2\n"
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
    "__aicore__ void tile_muls_128(__gm__ float* input, __gm__ float* output, float scale) {\n"
    "    __ub__ float a[16384];\n"
    "    __ub__ float result[16384];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: a = load(input[0, 0])\n"
    "    DataCopy(a, input[(0) * 16384], 16384);\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 128x128 = 16384 elements\n"
    "    Muls(result, a, , 16384);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_muls_128_sim_registered = 0;

void register_tile_muls_128_sim(IncoreSimulator* sim) {
    if (!tile_muls_128_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_muls_128",
            CORE_TYPE_VECTOR, tile_muls_128_instructions, 32, 128);
        tile_muls_128_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_muls_128_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_muls_128", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_muls_128
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_muls_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void tile_muls_128(__gm__ float* input, __gm__ float* output, float scale) {
    __ub__ float a[16384];
    __ub__ float result[16384];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(input[0, 0])
    DataCopy(a, input[(0) * 16384], 16384);

    // Fused vector operations: 1 operations
    // Tile size: 128x128 = 16384 elements
    Muls(result, a, , 16384);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/