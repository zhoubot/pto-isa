// PTO Program: F_normalize
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_normalize(float* input, float* output) {
    float x[8][8];
    float x_sq[8][8];
    float row_sum[8][1];
    float norm[8][1];
    float result[8][8];

    // Loop fusion: 2 loop overheads saved

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

    // TROWEXPANDDIV: result = x / broadcast(norm)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = norm[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] / _broadcast_val;
        }}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}