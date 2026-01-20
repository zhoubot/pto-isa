// PTO Program: F_silu
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_silu(float* input, float* output) {
    float x[8][8];
    float neg_x[8][8];
    float exp_neg[8][8];
    float one_plus[8][8];
    float sigmoid[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); neg_x=TNEG(x); exp_neg=TEXP(neg_x); one_plus=TADDS(exp_neg,1.0f); sigmoid=TRECIP(one_plus); result=TMUL(x,sigmoid); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vnegq_f32(_v2);
            vst1q_f32(&neg_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_neg[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&exp_neg[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&one_plus[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&sigmoid[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&sigmoid[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v10, _v11);
            vst1q_f32(&result[_row][_col], _vr12);
            float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs13);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg[_row][_col] = expf(neg_x[_row][_col]);
            one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
            sigmoid[_row][_col] = 1.0f / one_plus[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}