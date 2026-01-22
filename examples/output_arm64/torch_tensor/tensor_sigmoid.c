// PTO Program: tensor_sigmoid
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_sigmoid
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 1,280 bytes (1.2 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            768 bytes (60.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg              8x8        f32       256   [  2,   3]           <- self
//   neg_self             8x8        f32       256   [  1,   2]           -
//   one_plus             8x8        f32       256   [  3,   4]           <- neg_self
//   result               8x8        f32       256   [  4,   5]           <- exp_neg
//   self                 8x8        f32       256   [  0,   1]           -
//
// BUFFER REUSE MAP:
//   exp_neg reuses buffer of self
//   one_plus reuses buffer of neg_self
//   result reuses buffer of exp_neg
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_sigmoid(float* input, float* output) {
    float self[8][8];
    float neg_self[8][8];
    float exp_neg[8][8];
    float one_plus[8][8];
    float result[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): self=TLOAD(input,0,0); neg_self=TNEG(self); exp_neg=TEXP(neg_self); one_plus=TADDS(exp_neg,1.0f); result=TRECIP(one_plus); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr3 = vnegq_f32(_v2);
            vst1q_f32(&neg_self[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&neg_self[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_neg[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&exp_neg[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&one_plus[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs10);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            neg_self[_row][_col] = -self[_row][_col];
            exp_neg[_row][_col] = expf(neg_self[_row][_col]);
            one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
            result[_row][_col] = 1.0f / one_plus[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_sigmoid"; }
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
    tensor_sigmoid((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER