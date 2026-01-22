// PTO Program: F_leaky_relu
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_leaky_relu
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
//   neg_part             8x8        f32       256   [  4,   5]           <- neg_x
//   neg_relu             8x8        f32       256   [  3,   4]           <- x
//   neg_x                8x8        f32       256   [  2,   3]           -
//   pos_part             8x8        f32       256   [  1,   6]           -
//   result               8x8        f32       256   [  6,   7]           <- neg_part
//   scaled_neg           8x8        f32       256   [  5,   6]           <- neg_relu
//   x                    8x8        f32       256   [  0,   2]           -
//
// BUFFER REUSE MAP:
//   neg_relu reuses buffer of x
//   neg_part reuses buffer of neg_x
//   scaled_neg reuses buffer of neg_relu
//   result reuses buffer of neg_part
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_leaky_relu(float* input, float* output) {
    float x[8][8];
    float pos_part[8][8];
    float neg_x[8][8];
    float neg_relu[8][8];
    float neg_part[8][8];
    float scaled_neg[8][8];
    float result[8][8];

    // Loop fusion: 7 loop overheads saved

    // FUSED LOOP (8 ops): x=TLOAD(input,0,0); pos_part=TRELU(x); neg_x=TNEG(x); neg_relu=TRELU(neg_x); neg_part=TNEG(neg_relu); scaled_neg=TMULS(neg_part,0.01f); result=TADD(pos_part,scaled_neg); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.01f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vmaxq_f32(_v2, vdupq_n_f32(0.0f));
            vst1q_f32(&pos_part[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = vnegq_f32(_v4);
            vst1q_f32(&neg_x[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr7 = vmaxq_f32(_v6, vdupq_n_f32(0.0f));
            vst1q_f32(&neg_relu[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&neg_relu[_row][_col]);
            float32x4_t _vr9 = vnegq_f32(_v8);
            vst1q_f32(&neg_part[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&neg_part[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v10, _vs0);
            vst1q_f32(&scaled_neg[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&pos_part[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&scaled_neg[_row][_col]);
            float32x4_t _vr14 = vaddq_f32(_v12, _v13);
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
            neg_x[_row][_col] = -x[_row][_col];
            neg_relu[_row][_col] = fmaxf(neg_x[_row][_col], 0.0f);
            neg_part[_row][_col] = -neg_relu[_row][_col];
            scaled_neg[_row][_col] = neg_part[_row][_col] * 0.01f;
            result[_row][_col] = pos_part[_row][_col] + scaled_neg[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_leaky_relu"; }
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
    F_leaky_relu((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER