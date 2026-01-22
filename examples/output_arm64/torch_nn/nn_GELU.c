// PTO Program: nn_GELU
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: nn_GELU
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 1,792 bytes (1.8 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            1,024 bytes (57.1%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg              8x8        f32       256   [  3,   4]           <- scaled_x
//   neg_scaled           8x8        f32       256   [  2,   3]           -
//   one_plus             8x8        f32       256   [  4,   5]           <- neg_scaled
//   result               8x8        f32       256   [  6,   7]           <- one_plus
//   scaled_x             8x8        f32       256   [  1,   2]           -
//   sigmoid_out          8x8        f32       256   [  5,   6]           <- exp_neg
//   x                    8x8        f32       256   [  0,   6]           -
//
// BUFFER REUSE MAP:
//   exp_neg reuses buffer of scaled_x
//   one_plus reuses buffer of neg_scaled
//   sigmoid_out reuses buffer of exp_neg
//   result reuses buffer of one_plus
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void nn_GELU(float* input, float* output) {
    float x[8][8];
    float scaled_x[8][8];
    float neg_scaled[8][8];
    float exp_neg[8][8];
    float one_plus[8][8];
    float sigmoid_out[8][8];
    float result[8][8];

    // Loop fusion: 7 loop overheads saved

    // FUSED LOOP (8 ops): x=TLOAD(input,0,0); scaled_x=TMULS(x,1.702f); neg_scaled=TNEG(scaled_x); exp_neg=TEXP(neg_scaled); one_plus=TADDS(exp_neg,1.0f); sigmoid_out=TRECIP(one_plus); result=TMUL(x,sigmoid_out); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.702f);
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v3, _vs0);
            vst1q_f32(&scaled_x[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&scaled_x[_row][_col]);
            float32x4_t _vr6 = vnegq_f32(_v5);
            vst1q_f32(&neg_scaled[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&neg_scaled[_row][_col]);
            float32x4_t _vr8 = _v7;
            vst1q_f32(&exp_neg[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&exp_neg[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v9, _vs1);
            vst1q_f32(&one_plus[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr12 = _v11;
            vst1q_f32(&sigmoid_out[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v14 = vld1q_f32(&sigmoid_out[_row][_col]);
            float32x4_t _vr15 = vmulq_f32(_v13, _v14);
            vst1q_f32(&result[_row][_col], _vr15);
            float32x4_t _vs16 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs16);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            scaled_x[_row][_col] = x[_row][_col] * 1.702f;
            neg_scaled[_row][_col] = -scaled_x[_row][_col];
            exp_neg[_row][_col] = expf(neg_scaled[_row][_col]);
            one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
            sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "nn_GELU"; }
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
    nn_GELU((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER