// PTO Program: F_normalize
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_normalize
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 832 bytes (0.8 KB)
//   Total capacity (w/ reuse): 576 bytes (0.6 KB)
//   Reuse savings:            256 bytes (30.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   norm                 8x1        f32        32   [  3,   5]           -
//   result               8x8        f32       256   [  5,   6]           <- x_sq
//   row_sum              8x1        f32        32   [  2,   3]           -
//   x                    8x8        f32       256   [  0,   5]           -
//   x_sq                 8x8        f32       256   [  1,   2]           -
//
// BUFFER REUSE MAP:
//   result reuses buffer of x_sq
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_normalize(float* input, float* output) {
    float x[8][8];
    float x_sq[8][8];
    float row_sum[8][1];
    float norm[8][1];
    float result[8][8];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (2 ops): x=TLOAD(input,0,0); x_sq=TMUL(x,x)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vmulq_f32(_v1, _v2);
            vst1q_f32(&x_sq[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(x_sq)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += x_sq[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): norm=TSQRT(row_sum); norm=TADDS(norm,1e-12f)
    float32x4_t _vs4 = vdupq_n_f32(1e-12f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v5 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr6 = vsqrtq_f32(_v5);
            vst1q_f32(&norm[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&norm[_row][_col]);
            float32x4_t _vr8 = vaddq_f32(_v7, _vs4);
            vst1q_f32(&norm[_row][_col], _vr8);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            norm[_row][_col] = sqrtf(row_sum[_row][_col]);
            norm[_row][_col] = norm[_row][_col] + 1e-12f;
        }
    }

    // FUSED LOOP (2 ops): result=TROWEXPANDDIV(x,norm); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v09 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb11 = vdupq_n_f32(norm[_row][0]);
            float32x4_t _vr10 = vdivq_f32(_v09, _vb11);
            vst1q_f32(&result[_row][_col], _vr10);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] / norm[_row][0];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_normalize"; }
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
    F_normalize((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER