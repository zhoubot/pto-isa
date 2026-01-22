// PTO Program: tile_silu_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_silu_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            98,304 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_x            64x128     f32     32768   [  2,   3]           -
//   neg_x                64x128     f32     32768   [  1,   2]           -
//   one_plus_exp         64x128     f32     32768   [  3,   4]           <- neg_x
//   result               64x128     f32     32768   [  5,   6]           <- one_plus_exp
//   sigmoid              64x128     f32     32768   [  4,   5]           <- exp_neg_x
//   x                    64x128     f32     32768   [  0,   5]           -
//
// BUFFER REUSE MAP:
//   one_plus_exp reuses buffer of neg_x
//   sigmoid reuses buffer of exp_neg_x
//   result reuses buffer of one_plus_exp
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_silu_64(float* input, float* output) {
    float x[64][128];
    float neg_x[64][128];
    float exp_neg_x[64][128];
    float one_plus_exp[64][128];
    float sigmoid[64][128];
    float result[64][128];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); one_plus_exp=TADDS(exp_neg_x,1.0f); sigmoid=TRECIP(one_plus_exp); result=TMUL(x,sigmoid); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vnegq_f32(_v2);
            vst1q_f32(&neg_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_neg_x[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&one_plus_exp[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_plus_exp[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&sigmoid[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&sigmoid[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v10, _v11);
            vst1q_f32(&result[_row][_col], _vr12);
            float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs13);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            one_plus_exp[_row][_col] = exp_neg_x[_row][_col] + 1.0f;
            sigmoid[_row][_col] = 1.0f / one_plus_exp[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid[_row][_col];
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tile_silu_64"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(32768),
    (size_t)(32768),
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
    tile_silu_64((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER