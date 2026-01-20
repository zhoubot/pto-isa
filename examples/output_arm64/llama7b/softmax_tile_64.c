// PTO Program: softmax_tile_64
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: softmax_tile_64
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 131,584 bytes (128.5 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            65,792 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                64x128     f32     32768   [  3,   5]           <- x
//   result               64x128     f32     32768   [  5,   6]           <- x_shifted
//   row_max              64x1       f32       256   [  1,   2]           -
//   row_sum              64x1       f32       256   [  4,   5]           <- row_max
//   x                    64x128     f32     32768   [  0,   2]           -
//   x_shifted            64x128     f32     32768   [  2,   3]           -
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

void softmax_tile_64(float* input, float* output) {
    float x[64][128];
    float row_max[64][1];
    float x_shifted[64][128];
    float exp_x[64][128];
    float row_sum[64][1];
    float result[64][128];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 64; _row++) {
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
    for (int _row = 0; _row < 64; _row++) {
        float _max = x[_row][0];
        for (int _col = 1; _col < 128; _col++) {
            if (x[_row][_col] > _max) _max = x[_row][_col];
        }
        row_max[_row][0] = _max;}

    // FUSED LOOP (2 ops): x_shifted=TROWEXPANDSUB(x,row_max); exp_x=TEXP(x_shifted)
    for (int _row = 0; _row < 64; _row++) {
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
    for (int _row = 0; _row < 64; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 128; _col++) {
            _sum += exp_x[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): result=TROWEXPANDDIV(exp_x,row_sum); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 64; _row++) {
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