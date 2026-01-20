// PTO Program: nn_Tanh
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_Tanh(float* input, float* output) {
    float x[8][8];
    float exp_x[8][8];
    float neg_x[8][8];
    float exp_neg_x[8][8];
    float numerator[8][8];
    float denominator[8][8];
    float result[8][8];

    // Loop fusion: 7 loop overheads saved

    // FUSED LOOP (8 ops): x=TLOAD(input,0,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); numerator=TSUB(exp_x,exp_neg_x); denominator=TADD(exp_x,exp_neg_x); result=TDIV(numerator,denominator); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&exp_x[_row][_col], _vr2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vnegq_f32(_v3);
            vst1q_f32(&neg_x[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr6 = _v5;
            vst1q_f32(&exp_neg_x[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr9 = vsubq_f32(_v7, _v8);
            vst1q_f32(&numerator[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr12 = vaddq_f32(_v10, _v11);
            vst1q_f32(&denominator[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&numerator[_row][_col]);
            float32x4_t _v14 = vld1q_f32(&denominator[_row][_col]);
            float32x4_t _vr15 = vdivq_f32(_v13, _v14);
            vst1q_f32(&result[_row][_col], _vr15);
            float32x4_t _vs16 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs16);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            numerator[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
            denominator[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
            result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}