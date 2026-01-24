// =============================================================================
// InCore Function: tile_copy
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_copy_instructions = 
    "// PTO Program: tile_copy\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_copy\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     1\n"
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
    "__aicore__ void tile_copy(__gm__ float* A, __gm__ float* C) {\n"
    "    __ub__ float a[8192];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: a = load(A[0, 0])\n"
    "    DataCopy(a, A[(0) * 8192], 8192);\n"
    "\n"
    "    // TSTORE: store(a) -> C[0, 0]\n"
    "    DataCopy(C[(0) * 64], a, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_copy_sim_registered = 0;

void register_tile_copy_sim(IncoreSimulator* sim) {
    if (!tile_copy_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_copy",
            CORE_TYPE_VECTOR, tile_copy_instructions, 32, 128);
        tile_copy_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_copy_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_copy", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_copy
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_copy
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     1
//   Total capacity (no reuse): 32,768 bytes (32.0 KB)
//   Total capacity (w/ reuse): 32,768 bytes (32.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void tile_copy(__gm__ float* A, __gm__ float* C) {
    __ub__ float a[8192];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(A[0, 0])
    DataCopy(a, A[(0) * 8192], 8192);

    // TSTORE: store(a) -> C[0, 0]
    DataCopy(C[(0) * 64], a, 64);

}
*/