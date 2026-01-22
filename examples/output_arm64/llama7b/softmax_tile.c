// PTO Program: softmax_tile
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: softmax_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 65,792 bytes (64.2 KB)
//   Total capacity (w/ reuse): 32,896 bytes (32.1 KB)
//   Reuse savings:            32,896 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                32x128     f32     16384   [  3,   5]           <- x
//   result               32x128     f32     16384   [  5,   6]           <- x_shifted
//   row_max              32x1       f32       128   [  1,   2]           -
//   row_sum              32x1       f32       128   [  4,   5]           <- row_max
//   x                    32x128     f32     16384   [  0,   2]           -
//   x_shifted            32x128     f32     16384   [  2,   3]           -
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

void softmax_tile(float* input, float* output) {
    float x[32][128];
    float row_max[32][1];
    float x_shifted[32][128];
    float exp_x[32][128];
    float row_sum[32][1];
    float result[32][128];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 32; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 128 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
        }
    }

    // TROWMAX: row_max = rowmax(x)
    for (int _row = 0; _row < 32; _row++) {
        float _max = x[_row][0];
        for (int _col = 1; _col < 128; _col++) {
            if (x[_row][_col] > _max) _max = x[_row][_col];
        }
        row_max[_row][0] = _max;}

    // FUSED LOOP (2 ops): x_shifted=TROWEXPANDSUB(x,row_max); exp_x=TEXP(x_shifted)
    for (int _row = 0; _row < 32; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v01 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb3 = vdupq_n_f32(row_max[_row][0]);
            float32x4_t _vr2 = vsubq_f32(_v01, _vb3);
            vst1q_f32(&x_shifted[_row][_col], _vr2);
            float32x4_t _v4 = vld1q_f32(&x_shifted[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_x[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            x_shifted[_row][_col] = x[_row][_col] - row_max[_row][0];
            exp_x[_row][_col] = expf(x_shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_x)
    for (int _row = 0; _row < 32; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 128; _col++) {
            _sum += exp_x[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): result=TROWEXPANDDIV(exp_x,row_sum); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 32; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v06 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vb8 = vdupq_n_f32(row_sum[_row][0]);
            float32x4_t _vr7 = vdivq_f32(_v06, _vb8);
            vst1q_f32(&result[_row][_col], _vr7);
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            result[_row][_col] = exp_x[_row][_col] / row_sum[_row][0];
            output[_row * 128 + _col] = result[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "softmax_tile"; }
enum { kPtoNumMemrefs = 2 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input",
    "output",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(16384),
    (size_t)(16384),
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
    softmax_tile((float*)args[0], (float*)args[1]);
}
#endif  // PTO_CPU_SMOKE_RUNNER