// PTO Program: tensor_sign
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_sign(float* input, float* output) {
    float self[8][8];
    float abs_self[8][8];
    float abs_plus_eps[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): self=TLOAD(input,0,0); abs_self=TABS(self); abs_plus_eps=TADDS(abs_self,1e-07f); result=TDIV(self,abs_plus_eps); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1e-07f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr3 = vabsq_f32(_v2);
            vst1q_f32(&abs_self[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&abs_self[_row][_col]);
            float32x4_t _vr5 = vaddq_f32(_v4, _vs0);
            vst1q_f32(&abs_plus_eps[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&abs_plus_eps[_row][_col]);
            float32x4_t _vr8 = vdivq_f32(_v6, _v7);
            vst1q_f32(&result[_row][_col], _vr8);
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            abs_self[_row][_col] = fabsf(self[_row][_col]);
            abs_plus_eps[_row][_col] = abs_self[_row][_col] + 1e-07f;
            result[_row][_col] = self[_row][_col] / abs_plus_eps[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}