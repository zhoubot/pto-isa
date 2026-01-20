// PTO Program: tensor_asin
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_asin(float* input, float* output) {
    float x[8][8];
    float x2[8][8];
    float x3[8][8];
    float x5[8][8];
    float term1[8][8];
    float term2[8][8];
    float temp[8][8];
    float result[8][8];

    // Loop fusion: 8 loop overheads saved

    // FUSED LOOP (9 ops): x=TLOAD(input,0,0); x2=TMUL(x,x); x3=TMUL(x2,x); x5=TMUL(x3,x2); term1=TDIVS(x3,6.0f); term2=TMULS(x5,0.075f); temp=TADD(x,term1); result=TADD(temp,term2); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(6.0f);
    float32x4_t _vs1 = vdupq_n_f32(0.075f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v3, _v4);
            vst1q_f32(&x2[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr8 = vmulq_f32(_v6, _v7);
            vst1q_f32(&x3[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&x3[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v9, _v10);
            vst1q_f32(&x5[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&x3[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs0);
            vst1q_f32(&term1[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&x5[_row][_col]);
            float32x4_t _vr15 = vmulq_f32(_v14, _vs1);
            vst1q_f32(&term2[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&term1[_row][_col]);
            float32x4_t _vr18 = vaddq_f32(_v16, _v17);
            vst1q_f32(&temp[_row][_col], _vr18);
            float32x4_t _v19 = vld1q_f32(&temp[_row][_col]);
            float32x4_t _v20 = vld1q_f32(&term2[_row][_col]);
            float32x4_t _vr21 = vaddq_f32(_v19, _v20);
            vst1q_f32(&result[_row][_col], _vr21);
            float32x4_t _vs22 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs22);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x2[_row][_col] = x[_row][_col] * x[_row][_col];
            x3[_row][_col] = x2[_row][_col] * x[_row][_col];
            x5[_row][_col] = x3[_row][_col] * x2[_row][_col];
            term1[_row][_col] = x3[_row][_col] / 6.0f;
            term2[_row][_col] = x5[_row][_col] * 0.075f;
            temp[_row][_col] = x[_row][_col] + term1[_row][_col];
            result[_row][_col] = temp[_row][_col] + term2[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}