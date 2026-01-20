// PTO Program: tensor_tan
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_tan(float* input, float* output) {
    float x[8][8];
    float x2[8][8];
    float x3[8][8];
    float x4[8][8];
    float x5[8][8];
    float sin_t1[8][8];
    float sin_t2[8][8];
    float sin_temp[8][8];
    float sin_val[8][8];
    float cos_t1[8][8];
    float cos_t2[8][8];
    float ones[8][8];
    float cos_temp[8][8];
    float cos_val[8][8];
    float result[8][8];

    // Loop fusion: 15 loop overheads saved

    // FUSED LOOP (16 ops): x=TLOAD(input,0,0); x2=TMUL(x,x); x3=TMUL(x2,x); x4=TMUL(x2,x2); x5=TMUL(x3,x2); sin_t1=TDIVS(x3,6.0f); sin_t2=TDIVS(x5,120.0f); sin_temp=TSUB(x,sin_t1); sin_val=TADD(sin_temp,sin_t2); cos_t1=TDIVS(x2,2.0f); cos_t2=TDIVS(x4,24.0f); ones=TEXPANDS(1.0f); cos_temp=TSUB(ones,cos_t1); cos_val=TADD(cos_temp,cos_t2); result=TDIV(sin_val,cos_val); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(6.0f);
    float32x4_t _vs1 = vdupq_n_f32(120.0f);
    float32x4_t _vs2 = vdupq_n_f32(2.0f);
    float32x4_t _vs3 = vdupq_n_f32(24.0f);
    float32x4_t _vs4 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl5 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl5);
            float32x4_t _v6 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr8 = vmulq_f32(_v6, _v7);
            vst1q_f32(&x2[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v9, _v10);
            vst1q_f32(&x3[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr14 = vmulq_f32(_v12, _v13);
            vst1q_f32(&x4[_row][_col], _vr14);
            float32x4_t _v15 = vld1q_f32(&x3[_row][_col]);
            float32x4_t _v16 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr17 = vmulq_f32(_v15, _v16);
            vst1q_f32(&x5[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&x3[_row][_col]);
            float32x4_t _vr19 = vdivq_f32(_v18, _vs0);
            vst1q_f32(&sin_t1[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&x5[_row][_col]);
            float32x4_t _vr21 = vdivq_f32(_v20, _vs1);
            vst1q_f32(&sin_t2[_row][_col], _vr21);
            float32x4_t _v22 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v23 = vld1q_f32(&sin_t1[_row][_col]);
            float32x4_t _vr24 = vsubq_f32(_v22, _v23);
            vst1q_f32(&sin_temp[_row][_col], _vr24);
            float32x4_t _v25 = vld1q_f32(&sin_temp[_row][_col]);
            float32x4_t _v26 = vld1q_f32(&sin_t2[_row][_col]);
            float32x4_t _vr27 = vaddq_f32(_v25, _v26);
            vst1q_f32(&sin_val[_row][_col], _vr27);
            float32x4_t _v28 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr29 = vdivq_f32(_v28, _vs2);
            vst1q_f32(&cos_t1[_row][_col], _vr29);
            float32x4_t _v30 = vld1q_f32(&x4[_row][_col]);
            float32x4_t _vr31 = vdivq_f32(_v30, _vs3);
            vst1q_f32(&cos_t2[_row][_col], _vr31);
            vst1q_f32(&ones[_row][_col], _vs4);
            float32x4_t _v32 = vld1q_f32(&ones[_row][_col]);
            float32x4_t _v33 = vld1q_f32(&cos_t1[_row][_col]);
            float32x4_t _vr34 = vsubq_f32(_v32, _v33);
            vst1q_f32(&cos_temp[_row][_col], _vr34);
            float32x4_t _v35 = vld1q_f32(&cos_temp[_row][_col]);
            float32x4_t _v36 = vld1q_f32(&cos_t2[_row][_col]);
            float32x4_t _vr37 = vaddq_f32(_v35, _v36);
            vst1q_f32(&cos_val[_row][_col], _vr37);
            float32x4_t _v38 = vld1q_f32(&sin_val[_row][_col]);
            float32x4_t _v39 = vld1q_f32(&cos_val[_row][_col]);
            float32x4_t _vr40 = vdivq_f32(_v38, _v39);
            vst1q_f32(&result[_row][_col], _vr40);
            float32x4_t _vs41 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs41);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x2[_row][_col] = x[_row][_col] * x[_row][_col];
            x3[_row][_col] = x2[_row][_col] * x[_row][_col];
            x4[_row][_col] = x2[_row][_col] * x2[_row][_col];
            x5[_row][_col] = x3[_row][_col] * x2[_row][_col];
            sin_t1[_row][_col] = x3[_row][_col] / 6.0f;
            sin_t2[_row][_col] = x5[_row][_col] / 120.0f;
            sin_temp[_row][_col] = x[_row][_col] - sin_t1[_row][_col];
            sin_val[_row][_col] = sin_temp[_row][_col] + sin_t2[_row][_col];
            cos_t1[_row][_col] = x2[_row][_col] / 2.0f;
            cos_t2[_row][_col] = x4[_row][_col] / 24.0f;
            ones[_row][_col] = 1.0f;
            cos_temp[_row][_col] = ones[_row][_col] - cos_t1[_row][_col];
            cos_val[_row][_col] = cos_temp[_row][_col] + cos_t2[_row][_col];
            result[_row][_col] = sin_val[_row][_col] / cos_val[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}