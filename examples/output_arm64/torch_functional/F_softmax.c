// PTO Program: F_softmax
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_softmax(float* input, float* output) {
    float x[8][8];
    float row_max[8][1];
    float x_shifted[8][8];
    float exp_x[8][8];
    float row_sum[8][1];
    float result[8][8];

    // Loop fusion: 0 loop overheads saved

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

    // TROWEXPANDSUB: x_shifted = x - broadcast(row_max)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_max[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            x_shifted[_row][_col] = x[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (1 ops): exp_x=TEXP(x_shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&x_shifted[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_x[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
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

    // TROWEXPANDDIV: result = exp_x / broadcast(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            result[_row][_col] = exp_x[_row][_col] / _broadcast_val;
        }}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs6 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}