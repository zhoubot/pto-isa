// =============================================================================
// InCore Function: gemm_tile_acc
// Core Type: Cube
// =============================================================================

// Instruction code for core simulator parsing
static const char* gemm_tile_acc_instructions = 
    "// PTO Program: gemm_tile_acc\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: gemm_tile_acc\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     4\n"
    "//   Total capacity (no reuse): 114,688 bytes (112.0 KB)\n"
    "//   Total capacity (w/ reuse): 114,688 bytes (112.0 KB)\n"
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
    "__aicore__ void gemm_tile_acc(__gm__ float* A, __gm__ float* B, __gm__ float* C) {\n"
    "    __ub__ float a[4096];\n"
    "    __ub__ float b[8192];\n"
    "    __ub__ float c[8192];\n"
    "    __ub__ float tmp[8192];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: a = load(A[0, 0])\n"
    "    DataCopy(a, A[(0) * 4096], 4096);\n"
    "\n"
    "    // TLOAD: b = load(B[0, 0])\n"
    "    DataCopy(b, B[(0) * 8192], 8192);\n"
    "\n"
    "    // TLOAD: c = load(C[0, 0])\n"
    "    DataCopy(c, C[(0) * 8192], 8192);\n"
    "\n"
    "    // TMATMUL: tmp = a @ b\n"
    "    Matmul(tmp, a, b, 64, 128);\n"
    "\n"
    "    // Fused vector operations: 1 operations\n"
    "    // Tile size: 64x128 = 8192 elements\n"
    "    Add(c, c, tmp, 8192);\n"
    "\n"
    "    // TSTORE: store(c) -> C[0, 0]\n"
    "    DataCopy(C[(0) * 64], c, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int gemm_tile_acc_sim_registered = 0;

void register_gemm_tile_acc_sim(IncoreSimulator* sim) {
    if (!gemm_tile_acc_sim_registered) {
        a2a3_incore_sim_register_code(sim, "gemm_tile_acc",
            CORE_TYPE_CUBE, gemm_tile_acc_instructions, 32, 128);
        gemm_tile_acc_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t gemm_tile_acc_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("gemm_tile_acc", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: gemm_tile_acc
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: gemm_tile_acc
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 114,688 bytes (112.0 KB)
//   Total capacity (w/ reuse): 114,688 bytes (112.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void gemm_tile_acc(__gm__ float* A, __gm__ float* B, __gm__ float* C) {
    __ub__ float a[4096];
    __ub__ float b[8192];
    __ub__ float c[8192];
    __ub__ float tmp[8192];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(A[0, 0])
    DataCopy(a, A[(0) * 4096], 4096);

    // TLOAD: b = load(B[0, 0])
    DataCopy(b, B[(0) * 8192], 8192);

    // TLOAD: c = load(C[0, 0])
    DataCopy(c, C[(0) * 8192], 8192);

    // TMATMUL: tmp = a @ b
    Matmul(tmp, a, b, 64, 128);

    // Fused vector operations: 1 operations
    // Tile size: 64x128 = 8192 elements
    Add(c, c, tmp, 8192);

    // TSTORE: store(c) -> C[0, 0]
    DataCopy(C[(0) * 64], c, 64);

}
*/