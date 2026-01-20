// PTO Program: sinh_taylor
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void sinh_taylor(float* input, float* output) {
    float x[8][8];
    float x_squared[8][8];
    float term[8][8];
    float result[8][8];

    // Loop fusion: 22 loop overheads saved

    // FUSED LOOP (23 ops): x=TLOAD(input,0,0); result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,110.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,156.0f); result=TADD(result,term); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    float32x4_t _vs1 = vdupq_n_f32(6.0f);
    float32x4_t _vs2 = vdupq_n_f32(20.0f);
    float32x4_t _vs3 = vdupq_n_f32(42.0f);
    float32x4_t _vs4 = vdupq_n_f32(72.0f);
    float32x4_t _vs5 = vdupq_n_f32(110.0f);
    float32x4_t _vs6 = vdupq_n_f32(156.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl7 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl7);
            float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr9 = vmulq_f32(_v8, _vs0);
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v10, _v11);
            vst1q_f32(&x_squared[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr14 = vmulq_f32(_v13, _vs0);
            vst1q_f32(&term[_row][_col], _vr14);
            float32x4_t _v15 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v16 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr17 = vmulq_f32(_v15, _v16);
            vst1q_f32(&term[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr19 = vdivq_f32(_v18, _vs1);
            vst1q_f32(&term[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v21 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr22 = vaddq_f32(_v20, _v21);
            vst1q_f32(&result[_row][_col], _vr22);
            float32x4_t _v23 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v24 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr25 = vmulq_f32(_v23, _v24);
            vst1q_f32(&term[_row][_col], _vr25);
            float32x4_t _v26 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr27 = vdivq_f32(_v26, _vs2);
            vst1q_f32(&term[_row][_col], _vr27);
            float32x4_t _v28 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v29 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr30 = vaddq_f32(_v28, _v29);
            vst1q_f32(&result[_row][_col], _vr30);
            float32x4_t _v31 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v32 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr33 = vmulq_f32(_v31, _v32);
            vst1q_f32(&term[_row][_col], _vr33);
            float32x4_t _v34 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr35 = vdivq_f32(_v34, _vs3);
            vst1q_f32(&term[_row][_col], _vr35);
            float32x4_t _v36 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v37 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr38 = vaddq_f32(_v36, _v37);
            vst1q_f32(&result[_row][_col], _vr38);
            float32x4_t _v39 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v40 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr41 = vmulq_f32(_v39, _v40);
            vst1q_f32(&term[_row][_col], _vr41);
            float32x4_t _v42 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr43 = vdivq_f32(_v42, _vs4);
            vst1q_f32(&term[_row][_col], _vr43);
            float32x4_t _v44 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v45 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr46 = vaddq_f32(_v44, _v45);
            vst1q_f32(&result[_row][_col], _vr46);
            float32x4_t _v47 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v48 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr49 = vmulq_f32(_v47, _v48);
            vst1q_f32(&term[_row][_col], _vr49);
            float32x4_t _v50 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr51 = vdivq_f32(_v50, _vs5);
            vst1q_f32(&term[_row][_col], _vr51);
            float32x4_t _v52 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v53 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr54 = vaddq_f32(_v52, _v53);
            vst1q_f32(&result[_row][_col], _vr54);
            float32x4_t _v55 = vld1q_f32(&term[_row][_col]);
            float32x4_t _v56 = vld1q_f32(&x_squared[_row][_col]);
            float32x4_t _vr57 = vmulq_f32(_v55, _v56);
            vst1q_f32(&term[_row][_col], _vr57);
            float32x4_t _v58 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr59 = vdivq_f32(_v58, _vs6);
            vst1q_f32(&term[_row][_col], _vr59);
            float32x4_t _v60 = vld1q_f32(&result[_row][_col]);
            float32x4_t _v61 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr62 = vaddq_f32(_v60, _v61);
            vst1q_f32(&result[_row][_col], _vr62);
            float32x4_t _vs63 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs63);
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
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 110.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
            term[_row][_col] = term[_row][_col] / 156.0f;
            result[_row][_col] = result[_row][_col] + term[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}