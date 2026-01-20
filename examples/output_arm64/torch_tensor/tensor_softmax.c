// PTO Program: tensor_softmax
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_softmax(float* input, float* output) {
    float self[8][8];
    float row_mean[8][1];
    float shifted[8][8];
    float exp_shifted[8][8];
    float row_sum[8][1];
    float result[8][8];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): self=TLOAD(input,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
        }
    }

    // TROWSUM: row_mean = rowsum(self)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += self[_row][_col];
        }
        row_mean[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_mean=TDIVS(row_mean,8.0f)
    float32x4_t _vs1 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v2 = vld1q_f32(&row_mean[_row][_col]);
            float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
            vst1q_f32(&row_mean[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_mean[_row][_col] = row_mean[_row][_col] / 8.0f;
        }
    }

    // TROWEXPANDSUB: shifted = self - broadcast(row_mean)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_mean[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            shifted[_row][_col] = self[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (1 ops): exp_shifted=TEXP(shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_shifted[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            exp_shifted[_row][_col] = expf(shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_shifted)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_shifted[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TROWEXPANDDIV: result = exp_shifted / broadcast(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            result[_row][_col] = exp_shifted[_row][_col] / _broadcast_val;
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