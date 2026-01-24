// =============================================================================
// InCore Function: tile_matmul_128
// Core Type: Cube
// =============================================================================

// Instruction code for core simulator parsing
static const char* tile_matmul_128_instructions = 
    "// PTO Program: tile_matmul_128\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: tile_matmul_128\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     3\n"
    "//   Total capacity (no reuse): 196,608 bytes (192.0 KB)\n"
    "//   Total capacity (w/ reuse): 196,608 bytes (192.0 KB)\n"
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
    "__aicore__ void tile_matmul_128(__gm__ float* input_a, __gm__ float* input_b, __gm__ float* output) {\n"
    "    __ub__ float a[16384];\n"
    "    __ub__ float b[16384];\n"
    "    __ub__ float c[16384];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: a = load(input_a[0, 0])\n"
    "    DataCopy(a, input_a[(0) * 16384], 16384);\n"
    "\n"
    "    // TLOAD: b = load(input_b[0, 0])\n"
    "    DataCopy(b, input_b[(0) * 16384], 16384);\n"
    "\n"
    "    // TMATMUL: c = a @ b\n"
    "    Matmul(c, a, b, 128, 128);\n"
    "\n"
    "    // TSTORE: store(c) -> output[0, 0]\n"
    "    DataCopy(output[(0) * 64], c, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int tile_matmul_128_sim_registered = 0;

void register_tile_matmul_128_sim(IncoreSimulator* sim) {
    if (!tile_matmul_128_sim_registered) {
        a2a3_incore_sim_register_code(sim, "tile_matmul_128",
            CORE_TYPE_CUBE, tile_matmul_128_instructions, 32, 128);
        tile_matmul_128_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t tile_matmul_128_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("tile_matmul_128", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: tile_matmul_128
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: tile_matmul_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 196,608 bytes (192.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void tile_matmul_128(__gm__ float* input_a, __gm__ float* input_b, __gm__ float* output) {
    __ub__ float a[16384];
    __ub__ float b[16384];
    __ub__ float c[16384];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(input_a[0, 0])
    DataCopy(a, input_a[(0) * 16384], 16384);

    // TLOAD: b = load(input_b[0, 0])
    DataCopy(b, input_b[(0) * 16384], 16384);

    // TMATMUL: c = a @ b
    Matmul(c, a, b, 128, 128);

    // TSTORE: store(c) -> output[0, 0]
    DataCopy(output[(0) * 64], c, 64);

}
*/