// PTO Program: F_softplus
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_softplus(float* input, float* output) {
    float x[8][8];
    float beta_x[8][8];
    float exp_bx[8][8];
    float one_plus[8][8];
    float log_val[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); beta_x=TMULS(x,1.0f); exp_bx=TEXP(beta_x); one_plus=TADDS(exp_bx,1.0f); log_val=TLOG(one_plus); result=TDIVS(log_val,1.0f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vmulq_f32(_v2, _vs0);
            vst1q_f32(&beta_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&beta_x[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_bx[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&exp_bx[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&one_plus[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&log_val[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&log_val[_row][_col]);
            float32x4_t _vr11 = vdivq_f32(_v10, _vs0);
            vst1q_f32(&result[_row][_col], _vr11);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            beta_x[_row][_col] = x[_row][_col] * 1.0f;
            exp_bx[_row][_col] = expf(beta_x[_row][_col]);
            one_plus[_row][_col] = exp_bx[_row][_col] + 1.0f;
            log_val[_row][_col] = logf(one_plus[_row][_col]);
            result[_row][_col] = log_val[_row][_col] / 1.0f;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}