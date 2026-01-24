// =============================================================================
// InCore Function: tile_muls
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_muls_instructions = 
    "// PTO Program: tile_muls\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_muls\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     2\n"
    "//   Total capacity (no reuse): 32,768 bytes (32.0 KB)\n"
    "//   Total capacity (w/ reuse): 32,768 bytes (32.0 KB)\n"
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
    "__aicore__ void tile_muls(__gm__ float* input, __gm__ float* output, float scale) {\n"
    "    __ub__ float a[4096];\n"
    "    __ub__ float result[4096];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: a = load(input[0, 0])\n"
    "    DataCopy(a, input[(0) * 4096], 4096);\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 32x128 = 4096 elements\n"
    "    Muls(result, a, , 4096);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_muls_sim_registered = 0;

void register_tile_muls_sim(IncoreSimulator* sim) {
    if (!tile_muls_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_muls",
            CORE_TYPE_VECTOR, tile_muls_instructions, 32, 128);
        tile_muls_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_muls_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_muls", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_muls
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_muls
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 32,768 bytes (32.0 KB)
//   Total capacity (w/ reuse): 32,768 bytes (32.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void tile_muls(__gm__ float* input, __gm__ float* output, float scale) {
    __ub__ float a[4096];
    __ub__ float result[4096];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(input[0, 0])
    DataCopy(a, input[(0) * 4096], 4096);

    // Fused vector operations: 1 operations
    // Tile size: 32x128 = 4096 elements
    Muls(result, a, , 4096);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/