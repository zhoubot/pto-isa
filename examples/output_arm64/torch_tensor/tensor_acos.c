// PTO Program: tensor_acos
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_acos(float* input, float* output) {
    float x[8][8];
    float x2[8][8];
    float x3[8][8];
    float x5[8][8];
    float term1[8][8];
    float term2[8][8];
    float temp[8][8];
    float asin_val[8][8];
    float pi_half[8][8];
    float result[8][8];

    // Loop fusion: 10 loop overheads saved

    // FUSED LOOP (11 ops): x=TLOAD(input,0,0); x2=TMUL(x,x); x3=TMUL(x2,x); x5=TMUL(x3,x2); term1=TDIVS(x3,6.0f); term2=TMULS(x5,0.075f); temp=TADD(x,term1); asin_val=TADD(temp,term2); pi_half=TEXPANDS(1.5707963267948966f); result=TSUB(pi_half,asin_val); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(6.0f);
    float32x4_t _vs1 = vdupq_n_f32(0.075f);
    float32x4_t _vs2 = vdupq_n_f32(1.5707963267948966f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v4, _v5);
            vst1q_f32(&x2[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr9 = vmulq_f32(_v7, _v8);
            vst1q_f32(&x3[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&x3[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v10, _v11);
            vst1q_f32(&x5[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&x3[_row][_col]);
            float32x4_t _vr14 = vdivq_f32(_v13, _vs0);
            vst1q_f32(&term1[_row][_col], _vr14);
            float32x4_t _v15 = vld1q_f32(&x5[_row][_col]);
            float32x4_t _vr16 = vmulq_f32(_v15, _vs1);
            vst1q_f32(&term2[_row][_col], _vr16);
            float32x4_t _v17 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v18 = vld1q_f32(&term1[_row][_col]);
            float32x4_t _vr19 = vaddq_f32(_v17, _v18);
            vst1q_f32(&temp[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&temp[_row][_col]);
            float32x4_t _v21 = vld1q_f32(&term2[_row][_col]);
            float32x4_t _vr22 = vaddq_f32(_v20, _v21);
            vst1q_f32(&asin_val[_row][_col], _vr22);
            vst1q_f32(&pi_half[_row][_col], _vs2);
            float32x4_t _v23 = vld1q_f32(&pi_half[_row][_col]);
            float32x4_t _v24 = vld1q_f32(&asin_val[_row][_col]);
            float32x4_t _vr25 = vsubq_f32(_v23, _v24);
            vst1q_f32(&result[_row][_col], _vr25);
            float32x4_t _vs26 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs26);
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
            asin_val[_row][_col] = temp[_row][_col] + term2[_row][_col];
            pi_half[_row][_col] = 1.5707963267948966f;
            result[_row][_col] = pi_half[_row][_col] - asin_val[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}