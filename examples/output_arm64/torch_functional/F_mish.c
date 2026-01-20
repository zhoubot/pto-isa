// PTO Program: F_mish
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_mish(float* input, float* output) {
    float x[8][8];
    float exp_x[8][8];
    float one_plus_exp[8][8];
    float softplus[8][8];
    float sp_2[8][8];
    float exp_2sp[8][8];
    float tanh_num[8][8];
    float tanh_den[8][8];
    float tanh_out[8][8];
    float result[8][8];

    // Loop fusion: 10 loop overheads saved

    // FUSED LOOP (11 ops): x=TLOAD(input,0,0); exp_x=TEXP(x); one_plus_exp=TADDS(exp_x,1.0f); softplus=TLOG(one_plus_exp); sp_2=TMULS(softplus,2.0f); exp_2sp=TEXP(sp_2); tanh_num=TADDS(exp_2sp,-1.0f); tanh_den=TADDS(exp_2sp,1.0f); tanh_out=TDIV(tanh_num,tanh_den); result=TMUL(x,tanh_out); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    float32x4_t _vs1 = vdupq_n_f32(2.0f);
    float32x4_t _vs2 = vdupq_n_f32(-1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_x[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&one_plus_exp[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_plus_exp[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&softplus[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&softplus[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v10, _vs1);
            vst1q_f32(&sp_2[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&sp_2[_row][_col]);
            float32x4_t _vr13 = _v12;
            vst1q_f32(&exp_2sp[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&exp_2sp[_row][_col]);
            float32x4_t _vr15 = vaddq_f32(_v14, _vs2);
            vst1q_f32(&tanh_num[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&exp_2sp[_row][_col]);
            float32x4_t _vr17 = vaddq_f32(_v16, _vs0);
            vst1q_f32(&tanh_den[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&tanh_num[_row][_col]);
            float32x4_t _v19 = vld1q_f32(&tanh_den[_row][_col]);
            float32x4_t _vr20 = vdivq_f32(_v18, _v19);
            vst1q_f32(&tanh_out[_row][_col], _vr20);
            float32x4_t _v21 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v22 = vld1q_f32(&tanh_out[_row][_col]);
            float32x4_t _vr23 = vmulq_f32(_v21, _v22);
            vst1q_f32(&result[_row][_col], _vr23);
            float32x4_t _vs24 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs24);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            one_plus_exp[_row][_col] = exp_x[_row][_col] + 1.0f;
            softplus[_row][_col] = logf(one_plus_exp[_row][_col]);
            sp_2[_row][_col] = softplus[_row][_col] * 2.0f;
            exp_2sp[_row][_col] = expf(sp_2[_row][_col]);
            tanh_num[_row][_col] = exp_2sp[_row][_col] + -1.0f;
            tanh_den[_row][_col] = exp_2sp[_row][_col] + 1.0f;
            tanh_out[_row][_col] = tanh_num[_row][_col] / tanh_den[_row][_col];
            result[_row][_col] = x[_row][_col] * tanh_out[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}