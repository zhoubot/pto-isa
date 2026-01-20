// PTO Program: tensor_logit
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_logit(float* input, float* output) {
    float self[8][8];
    float one_minus[8][8];
    float ratio[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): self=TLOAD(input,0,0); one_minus=TMULS(self,-1.0f); one_minus=TADDS(one_minus,1.0f); one_minus=TADDS(one_minus,1e-06f); ratio=TDIV(self,one_minus); result=TLOG(ratio); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1.0f);
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    float32x4_t _vs2 = vdupq_n_f32(1e-06f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v4, _vs0);
            vst1q_f32(&one_minus[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&one_minus[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs1);
            vst1q_f32(&one_minus[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_minus[_row][_col]);
            float32x4_t _vr9 = vaddq_f32(_v8, _vs2);
            vst1q_f32(&one_minus[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&one_minus[_row][_col]);
            float32x4_t _vr12 = vdivq_f32(_v10, _v11);
            vst1q_f32(&ratio[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&ratio[_row][_col]);
            float32x4_t _vr14 = _v13;
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            one_minus[_row][_col] = self[_row][_col] * -1.0f;
            one_minus[_row][_col] = one_minus[_row][_col] + 1.0f;
            one_minus[_row][_col] = one_minus[_row][_col] + 1e-06f;
            ratio[_row][_col] = self[_row][_col] / one_minus[_row][_col];
            result[_row][_col] = logf(ratio[_row][_col]);
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}