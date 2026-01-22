// PTO Program: tensor_sinh
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tensor_sinh
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,536 bytes (1.5 KB)
//   Total capacity (w/ reuse): 768 bytes (0.8 KB)
//   Reuse savings:            768 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   diff                 8x8        f32       256   [  4,   5]           <- neg_x
//   exp_neg_x            8x8        f32       256   [  3,   4]           <- x
//   exp_x                8x8        f32       256   [  2,   4]           -
//   neg_x                8x8        f32       256   [  1,   3]           -
//   result               8x8        f32       256   [  5,   6]           <- exp_x
//   x                    8x8        f32       256   [  0,   2]           -
//
// BUFFER REUSE MAP:
//   exp_neg_x reuses buffer of x
//   diff reuses buffer of neg_x
//   result reuses buffer of exp_x
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tensor_sinh(float* input, float* output) {
    float x[8][8];
    float neg_x[8][8];
    float exp_x[8][8];
    float exp_neg_x[8][8];
    float diff[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); neg_x=TNEG(x); exp_x=TEXP(x); exp_neg_x=TEXP(neg_x); diff=TSUB(exp_x,exp_neg_x); result=TDIVS(diff,2.0f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(2.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vnegq_f32(_v2);
            vst1q_f32(&neg_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_x[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&exp_neg_x[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr10 = vsubq_f32(_v8, _v9);
            vst1q_f32(&diff[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr12 = vdivq_f32(_v11, _vs0);
            vst1q_f32(&result[_row][_col], _vr12);
            float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs13);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            neg_x[_row][_col] = -x[_row][_col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            diff[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
            result[_row][_col] = diff[_row][_col] / 2.0f;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "tensor_sinh"; }
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
    tensor_sinh((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER