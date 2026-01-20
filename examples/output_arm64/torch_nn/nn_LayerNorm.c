// PTO Program: nn_LayerNorm
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_LayerNorm(float* input, float* output) {
    float x[8][8];
    float row_sum[8][1];
    float mean[8][1];
    float x_minus_mean[8][8];
    float squared[8][8];
    float var_sum[8][1];
    float variance[8][1];
    float var_eps[8][1];
    float std[8][1];
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

    // TROWSUM: row_sum = rowsum(x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += x[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): mean=TDIVS(row_sum,8.0f)
    float32x4_t _vs1 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v2 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
            vst1q_f32(&mean[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            mean[_row][_col] = row_sum[_row][_col] / 8.0f;
        }
    }

    // TROWEXPANDSUB: x_minus_mean = x - broadcast(mean)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = mean[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            x_minus_mean[_row][_col] = x[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (1 ops): squared=TMUL(x_minus_mean,x_minus_mean)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&x_minus_mean[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&x_minus_mean[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v4, _v5);
            vst1q_f32(&squared[_row][_col], _vr6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            squared[_row][_col] = x_minus_mean[_row][_col] * x_minus_mean[_row][_col];
        }
    }

    // TROWSUM: var_sum = rowsum(squared)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += squared[_row][_col];
        }
        var_sum[_row][0] = _sum;}

    // FUSED LOOP (3 ops): variance=TDIVS(var_sum,8.0f); var_eps=TADDS(variance,1e-05f); std=TSQRT(var_eps)
    float32x4_t _vs7 = vdupq_n_f32(8.0f);
    float32x4_t _vs8 = vdupq_n_f32(1e-05f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v9 = vld1q_f32(&var_sum[_row][_col]);
            float32x4_t _vr10 = vdivq_f32(_v9, _vs7);
            vst1q_f32(&variance[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&variance[_row][_col]);
            float32x4_t _vr12 = vaddq_f32(_v11, _vs8);
            vst1q_f32(&var_eps[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&var_eps[_row][_col]);
            float32x4_t _vr14 = vsqrtq_f32(_v13);
            vst1q_f32(&std[_row][_col], _vr14);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            variance[_row][_col] = var_sum[_row][_col] / 8.0f;
            var_eps[_row][_col] = variance[_row][_col] + 1e-05f;
            std[_row][_col] = sqrtf(var_eps[_row][_col]);
        }
    }

    // TROWEXPANDDIV: result = x_minus_mean / broadcast(std)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = std[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            result[_row][_col] = x_minus_mean[_row][_col] / _broadcast_val;
        }}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}