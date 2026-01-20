// PTO Program: tensor_expm1
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_expm1(float* input, float* output) {
    float self[8][8];
    float exp_val[8][8];
    float result[8][8];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (4 ops): self=TLOAD(input,0,0); exp_val=TEXP(self); result=TADDS(exp_val,-1.0f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&exp_val[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&exp_val[_row][_col]);
            float32x4_t _vr5 = vaddq_f32(_v4, _vs0);
            vst1q_f32(&result[_row][_col], _vr5);
            float32x4_t _vs6 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            exp_val[_row][_col] = expf(self[_row][_col]);
            result[_row][_col] = exp_val[_row][_col] + -1.0f;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}