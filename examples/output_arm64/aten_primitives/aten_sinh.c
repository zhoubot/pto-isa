// PTO Program: aten_sinh
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_sinh(float* input, float* output) {
    float x[8][8];
    float x_squared[8][8];
    float term[8][8];
    float result[8][8];

    // Loop fusion: 16 loop overheads saved

    // FUSED LOOP (17 ops): x=TLOAD(input,0,0); result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    float32x4_t _vs1 = vdupq_n_f32(6.0f);
    float32x4_t _vs2 = vdupq_n_f32(20.0f);
    float32x4_t _vs3 = vdupq_n_f32(42.0f);
    float32x4_t _vs4 = vdupq_n_f32(72.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl5 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl5);
            float32x4_t _v6 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v6, _vs0);
            vst1q_f32(&result[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr10 = vmulq_f32(_v8, _v9);
            vst1q_f32(&x_squared[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v11, _vs0);
            vst1q_f32(&term[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v14 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr15 = vmulq_f32(_v13, _v14);
            vst1q_f32(&term[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr17 = vdivq_f32(_v16, _vs1);
            vst1q_f32(&term[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v19 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr20 = vaddq_f32(_v18, _v19);
            vst1q_f32(&result[_row][_col], _vr20);
            float32x4_t _v21 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v22 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr23 = vmulq_f32(_v21, _v22);
            vst1q_f32(&term[_row][_col], _vr23);
            float32x4_t _v24 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr25 = vdivq_f32(_v24, _vs2);
            vst1q_f32(&term[_row][_col], _vr25);
            float32x4_t _v26 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v27 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr28 = vaddq_f32(_v26, _v27);
            vst1q_f32(&result[_row][_col], _vr28);
            float32x4_t _v29 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v30 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr31 = vmulq_f32(_v29, _v30);
            vst1q_f32(&term[_row][_col], _vr31);
            float32x4_t _v32 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr33 = vdivq_f32(_v32, _vs3);
            vst1q_f32(&term[_row][_col], _vr33);
            float32x4_t _v34 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v35 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr36 = vaddq_f32(_v34, _v35);
            vst1q_f32(&result[_row][_col], _vr36);
            float32x4_t _v37 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v38 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr39 = vmulq_f32(_v37, _v38);
            vst1q_f32(&term[_row][_col], _vr39);
            float32x4_t _v40 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr41 = vdivq_f32(_v40, _vs4);
            vst1q_f32(&term[_row][_col], _vr41);
            float32x4_t _v42 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v43 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr44 = vaddq_f32(_v42, _v43);
            vst1q_f32(&result[_row][_col], _vr44);
            float32x4_t _vs45 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs45);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            result[_row][_col] = x[_row][_col] * 1.0f;
            x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
            term[_row][_col] = x[_row][_col] * 1.0f;
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 6.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 20.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 42.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 72.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}