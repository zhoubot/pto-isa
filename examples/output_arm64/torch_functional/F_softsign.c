// PTO Program: F_softsign
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_softsign
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 1,024 bytes (1.0 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            256 bytes (25.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_x                8x8        f32       256   [  1,   2]           -
//   one_plus_abs         8x8        f32       256   [  2,   3]           -
//   result               8x8        f32       256   [  3,   4]           <- abs_x
//   x                    8x8        f32       256   [  0,   3]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of abs_x
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_softsign(float* input, float* output) {
    float x[8][8];
    float abs_x[8][8];
    float one_plus_abs[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): x=TLOAD(input,0,0); abs_x=TABS(x); one_plus_abs=TADDS(abs_x,1.0f); result=TDIV(x,one_plus_abs); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vabsq_f32(_v2);
            vst1q_f32(&abs_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&abs_x[_row][_col]);
            float32x4_t _vr5 = vaddq_f32(_v4, _vs0);
            vst1q_f32(&one_plus_abs[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&one_plus_abs[_row][_col]);
            float32x4_t _vr8 = vdivq_f32(_v6, _v7);
            vst1q_f32(&result[_row][_col], _vr8);
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            abs_x[_row][_col] = fabsf(x[_row][_col]);
            one_plus_abs[_row][_col] = abs_x[_row][_col] + 1.0f;
            result[_row][_col] = x[_row][_col] / one_plus_abs[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_softsign"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
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
    F_softsign((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER