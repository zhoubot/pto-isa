// PTO Program: flash_attn_normalize
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_normalize
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 65,792 bytes (64.2 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_vec                64x1       f32       256   [  1,   2]           -
//   o_block              64x128     f32     32768   [  0,   2]           -
//   o_final              64x128     f32     32768   [  2,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_normalize(float* input_o, float* input_l, float* output) {
    float o_block[64][128];
    float l_vec[64][1];
    float o_final[64][128];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): o_block=TLOAD(input_o,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_o[_row * 128 + _col]);
            vst1q_f32(&o_block[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_block[_row][_col] = input_o[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): l_vec=TLOAD(input_l,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_l[_row * 1 + _col]);
            vst1q_f32(&l_vec[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            l_vec[_row][_col] = input_l[_row * 1 + _col];
        }
    }

    // FUSED LOOP (2 ops): o_final=TROWEXPANDDIV(o_block,l_vec); output=TSTORE(o_final,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v02 = vld1q_f32(&o_block[_row][_col]);
            float32x4_t _vb4 = vdupq_n_f32(l_vec[_row][0]);
            float32x4_t _vr3 = vdivq_f32(_v02, _vb4);
            vst1q_f32(&o_final[_row][_col], _vr3);
            float32x4_t _vs5 = vld1q_f32(&o_final[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_final[_row][_col] = o_block[_row][_col] / l_vec[_row][0];
            output[_row * 128 + _col] = o_final[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "flash_attn_normalize"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_o",
    "input_l",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(32768),
    (size_t)(256),
    (size_t)(32768),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
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
    flash_attn_normalize((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER