// PTO Program: F_softmax
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: F_softmax
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 1,088 bytes (1.1 KB)
//   Total capacity (w/ reuse): 544 bytes (0.5 KB)
//   Reuse savings:            544 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                8x8        f32       256   [  4,   6]           <- x
//   result               8x8        f32       256   [  6,   7]           <- x_shifted
//   row_max              8x1        f32        32   [  1,   3]           -
//   row_sum              8x1        f32        32   [  5,   6]           <- row_max
//   x                    8x8        f32       256   [  0,   3]           -
//   x_shifted            8x8        f32       256   [  3,   4]           -
//
// BUFFER REUSE MAP:
//   exp_x reuses buffer of x
//   row_sum reuses buffer of row_max
//   result reuses buffer of x_shifted
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void F_softmax(float* input, float* output) {
    float x[8][8];
    float row_max[8][1];
    float x_shifted[8][8];
    float exp_x[8][8];
    float row_sum[8][1];
    float result[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
        }
    }

    // TROWSUM: row_max = rowsum(x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += x[_row][_col];
        }
        row_max[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_max=TDIVS(row_max,8.0f)
    float32x4_t _vs1 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v2 = vld1q_f32(&row_max[_row][_col]);
            float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
            vst1q_f32(&row_max[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_max[_row][_col] = row_max[_row][_col] / 8.0f;
        }
    }

    // FUSED LOOP (2 ops): x_shifted=TROWEXPANDSUB(x,row_max); exp_x=TEXP(x_shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v04 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb6 = vdupq_n_f32(row_max[_row][0]);
            float32x4_t _vr5 = vsubq_f32(_v04, _vb6);
            vst1q_f32(&x_shifted[_row][_col], _vr5);
            float32x4_t _v7 = vld1q_f32(&x_shifted[_row][_col]);
            float32x4_t _vr8 = _v7;
            vst1q_f32(&exp_x[_row][_col], _vr8);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x_shifted[_row][_col] = x[_row][_col] - row_max[_row][0];
            exp_x[_row][_col] = expf(x_shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_x[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): result=TROWEXPANDDIV(exp_x,row_sum); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v09 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vb11 = vdupq_n_f32(row_sum[_row][0]);
            float32x4_t _vr10 = vdivq_f32(_v09, _vb11);
            vst1q_f32(&result[_row][_col], _vr10);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = exp_x[_row][_col] / row_sum[_row][0];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "F_softmax"; }
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
    F_softmax((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER