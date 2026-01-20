// PTO Program: F_tanh
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_tanh(float* input, float* output) {
    float x[8][8];
    float x_2[8][8];
    float exp_2x[8][8];
    float numerator[8][8];
    float denominator[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); x_2=TMULS(x,2.0f); exp_2x=TEXP(x_2); numerator=TADDS(exp_2x,-1.0f); denominator=TADDS(exp_2x,1.0f); result=TDIV(numerator,denominator); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(2.0f);
    float32x4_t _vs1 = vdupq_n_f32(-1.0f);
    float32x4_t _vs2 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v4, _vs0);
            vst1q_f32(&x_2[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&x_2[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&exp_2x[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&exp_2x[_row][_col]);
            float32x4_t _vr9 = vaddq_f32(_v8, _vs1);
            vst1q_f32(&numerator[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&exp_2x[_row][_col]);
            float32x4_t _vr11 = vaddq_f32(_v10, _vs2);
            vst1q_f32(&denominator[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&numerator[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&denominator[_row][_col]);
            float32x4_t _vr14 = vdivq_f32(_v12, _v13);
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_2[_row][_col] = x[_row][_col] * 2.0f;
            exp_2x[_row][_col] = expf(x_2[_row][_col]);
            numerator[_row][_col] = exp_2x[_row][_col] + -1.0f;
            denominator[_row][_col] = exp_2x[_row][_col] + 1.0f;
            result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}