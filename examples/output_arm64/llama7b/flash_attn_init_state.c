// PTO Program: flash_attn_init_state
// Function Type: InCore (tile-level computation)
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
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_init               64x1       f32       256   [  1,   4]           -
//   m_init               64x1       f32       256   [  2,   5]           -
//   o_init               64x128     f32     32768   [  0,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_init_state(float* input_zeros_large, float* input_zeros_small, float* input_neg_inf, float* output_o, float* output_l, float* output_m) {
    float o_init[64][128];
    float l_init[64][1];
    float m_init[64][1];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (1 ops): o_init=TLOAD(input_zeros_large,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_zeros_large[_row * 128 + _col]);
            vst1q_f32(&o_init[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_init[_row][_col] = input_zeros_large[_row * 128 + _col];
        }
    }

    // FUSED LOOP (2 ops): l_init=TLOAD(input_zeros_small,0,0); m_init=TLOAD(input_neg_inf,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_zeros_small[_row * 1 + _col]);
            vst1q_f32(&l_init[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&input_neg_inf[_row * 1 + _col]);
            vst1q_f32(&m_init[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            l_init[_row][_col] = input_zeros_small[_row * 1 + _col];
            m_init[_row][_col] = input_neg_inf[_row * 1 + _col];
        }
    }

    // FUSED LOOP (1 ops): output_o=TSTORE(o_init,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vs3 = vld1q_f32(&o_init[_row][_col]);
            vst1q_f32(&output_o[_row * 128 + _col], _vs3);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            output_o[_row * 128 + _col] = o_init[_row][_col];
        }
    }

    // FUSED LOOP (2 ops): output_l=TSTORE(l_init,0,0); output_m=TSTORE(m_init,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs4 = vld1q_f32(&l_init[_row][_col]);
            vst1q_f32(&output_l[_row * 1 + _col], _vs4);
            float32x4_t _vs5 = vld1q_f32(&m_init[_row][_col]);
            vst1q_f32(&output_m[_row * 1 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output_l[_row * 1 + _col] = l_init[_row][_col];
            output_m[_row * 1 + _col] = m_init[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "flash_attn_init_state"; }
enum { kPtoNumMemrefs = 6 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_zeros_large",
    "input_zeros_small",
    "input_neg_inf",
    "output_o",
    "output_l",
    "output_m",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(32768),
    (size_t)(256),
    (size_t)(256),
    (size_t)(32768),
    (size_t)(256),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
    0,
    1,
    1,
    1,
};
int pto_num_memrefs() { return kPtoNumMemrefs; }
const char* pto_memref_name(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefNames[idx];
}
size_t pto_memref_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefBytes[idx];
}
const char* pto_memref_dtype(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefDtypes[idx];
}
size_t pto_memref_elem_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefElemBytes[idx];
}
int pto_memref_is_output(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefIsOutput[idx];
}
void pto_launch(void **args, void *stream) {
    (void)stream;
    flash_attn_init_state((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4], (float*)args[5]);
}
#endif  // PTO_CPU_SMOKE_RUNNER