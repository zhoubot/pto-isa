// =============================================================================
// InCore Function: flash_attn_init_state
// Core Type: Vector
// =============================================================================

// Instruction code for core simulator parsing
static const char* flash_attn_init_state_instructions = 
    "// PTO Program: flash_attn_init_state\n"
    "// Target: Ascend A2/A3\n"
    "// ======================================================================\n"
    "// TILE BUFFER ANALYSIS: flash_attn_init_state\n"
    "// ======================================================================\n"
    "//\n"
    "// SUMMARY:\n"
    "//   Total tiles declared:     3\n"
    "//   Total capacity (no reuse): 33,280 bytes (32.5 KB)\n"
    "//   Total capacity (w/ reuse): 33,280 bytes (32.5 KB)\n"
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
    "__aicore__ void flash_attn_init_state(__gm__ float* input_zeros_large, __gm__ float* input_zeros_small, __gm__ float* input_neg_inf, __gm__ float* output_o, __gm__ float* output_l, __gm__ float* output_m) {\n"
    "    __ub__ float o_init[8192];\n"
    "    __ub__ float l_init[64];\n"
    "    __ub__ float m_init[64];\n"
    "\n"
    "    // Loop fusion: 0 loop overheads saved\n"
    "\n"
    "    // TLOAD: o_init = load(input_zeros_large[0, 0])\n"
    "    DataCopy(o_init, input_zeros_large[(0) * 8192], 8192);\n"
    "\n"
    "    // TLOAD: l_init = load(input_zeros_small[0, 0])\n"
    "    DataCopy(l_init, input_zeros_small[(0) * 64], 64);\n"
    "\n"
    "    // TLOAD: m_init = load(input_neg_inf[0, 0])\n"
    "    DataCopy(m_init, input_neg_inf[(0) * 64], 64);\n"
    "\n"
    "    // TSTORE: store(o_init) -> output_o[0, 0]\n"
    "    DataCopy(output_o[(0) * 64], o_init, 64);\n"
    "\n"
    "    // TSTORE: store(l_init) -> output_l[0, 0]\n"
    "    DataCopy(output_l[(0) * 64], l_init, 64);\n"
    "\n"
    "    // TSTORE: store(m_init) -> output_m[0, 0]\n"
    "    DataCopy(output_m[(0) * 64], m_init, 64);\n"
    "\n"
    "}";

#ifdef A2A3_CORE_SIM_AVAILABLE
static int flash_attn_init_state_sim_registered = 0;

void register_flash_attn_init_state_sim(IncoreSimulator* sim) {
    if (!flash_attn_init_state_sim_registered) {
        a2a3_incore_sim_register_code(sim, "flash_attn_init_state",
            CORE_TYPE_VECTOR, flash_attn_init_state_instructions, 32, 128);
        flash_attn_init_state_sim_registered = 1;
    }
}
#endif

// Get cycle cost for this function
int64_t flash_attn_init_state_cycle_cost(int64_t tile_size) {
    return get_incore_cycle_cost_sim("flash_attn_init_state", tile_size);
}

/*
// Actual Ascend Instructions (for physical execution):
// PTO Program: flash_attn_init_state
// Target: Ascend A2/A3
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_init_state
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 33,280 bytes (32.5 KB)
//   Total capacity (w/ reuse): 33,280 bytes (32.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ void flash_attn_init_state(__gm__ float* input_zeros_large, __gm__ float* input_zeros_small, __gm__ float* input_neg_inf, __gm__ float* output_o, __gm__ float* output_l, __gm__ float* output_m) {
    __ub__ float o_init[8192];
    __ub__ float l_init[64];
    __ub__ float m_init[64];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: o_init = load(input_zeros_large[0, 0])
    DataCopy(o_init, input_zeros_large[(0) * 8192], 8192);

    // TLOAD: l_init = load(input_zeros_small[0, 0])
    DataCopy(l_init, input_zeros_small[(0) * 64], 64);

    // TLOAD: m_init = load(input_neg_inf[0, 0])
    DataCopy(m_init, input_neg_inf[(0) * 64], 64);

    // TSTORE: store(o_init) -> output_o[0, 0]
    DataCopy(output_o[(0) * 64], o_init, 64);

    // TSTORE: store(l_init) -> output_l[0, 0]
    DataCopy(output_l[(0) * 64], l_init, 64);

    // TSTORE: store(m_init) -> output_m[0, 0]
    DataCopy(output_m[(0) * 64], m_init, 64);

}
*/