// =============================================================================
// InCore Function: tile_rowsum_128
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_rowsum_128_instructions = 
    "// PTO Program: tile_rowsum_128\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_rowsum_128\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     2\n"
    "//   Total capacity (no reuse): 66,048 bytes (64.5 KB)\n"
    "//   Total capacity (w/ reuse): 66,048 bytes (64.5 KB)\n"
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
    "__aicore__ void tile_rowsum_128(__gm__ float* input, __gm__ float* output) {\n"
    "    __ub__ float x[16384];\n"
    "    __ub__ float result[128];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: x = load(input[0, 0])\n"
    "    DataCopy(x, input[(0) * 16384], 16384);\n"
    "\n"
    "    // TROWSUM: reduction operation\n"
    "    ReduceSum(result, x, 128);\n"
    "\n"
    "    // TSTORE: store(result) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], result, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_rowsum_128_sim_registered = 0;

void register_tile_rowsum_128_sim(IncoreSimulator* sim) {
    if (!tile_rowsum_128_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_rowsum_128",
            CORE_TYPE_VECTOR, tile_rowsum_128_instructions, 32, 128);
        tile_rowsum_128_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_rowsum_128_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_rowsum_128", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_rowsum_128
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_rowsum_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 66,048 bytes (64.5 KB)
//   Total capacity (w/ reuse): 66,048 bytes (64.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void tile_rowsum_128(__gm__ float* input, __gm__ float* output) {
    __ub__ float x[16384];
    __ub__ float result[128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    DataCopy(x, input[(0) * 16384], 16384);

    // TROWSUM: reduction operation
    ReduceSum(result, x, 128);

    // TSTORE: store(result) -> output[0, 0]
    DataCopy(output[(0) * 64], result, 64);

}
*/