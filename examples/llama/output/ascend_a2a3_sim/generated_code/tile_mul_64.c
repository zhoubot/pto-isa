// =============================================================================
// InCore Function: tile_mul_64
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_mul_64_instructions = 
    "// PTO Program: tile_mul_64\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_mul_64\n"
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
    "__aicore__ void tile_mul_64(__gm__ float* input_a, __gm__ float* input_b, __gm__ float* output) {\n"
    "    __ub__ float a[8192];\n"
    "    __ub__ float b[8192];\n"
    "    __ub__ float result[8192];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: a = load(input_a[0, 0])\n"
    "    DataCopy(a, input_a[(0) * 8192], 8192);\n"
    "\n"
    "    // TLOAD: b = load(input_b[0, 0])\n"
    "    DataCopy(b, input_b[(0) * 8192], 8192);\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Mul(result, a, b, 8192);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_mul_64_sim_registered = 0;

void register_tile_mul_64_sim(IncoreSimulator* sim) {
    if (!tile_mul_64_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_mul_64",
            CORE_TYPE_VECTOR, tile_mul_64_instructions, 32, 128);
        tile_mul_64_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_mul_64_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_mul_64", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_mul_64
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_mul_64
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

__aicore__ void tile_mul_64(__gm__ float* input_a, __gm__ float* input_b, __gm__ float* output) {
    __ub__ float a[8192];
    __ub__ float b[8192];
    __ub__ float result[8192];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(input_a[0, 0])
    DataCopy(a, input_a[(0) * 8192], 8192);

    // TLOAD: b = load(input_b[0, 0])
    DataCopy(b, input_b[(0) * 8192], 8192);

    // Fused vector operations: 1 operations
    // Tile size: 64x128 = 8192 elements
    Mul(result, a, b, 8192);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/