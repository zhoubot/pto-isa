// PTO Program: tensor_xlogy
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_xlogy(float* input_self, float* input_other, float* output) {
    float self[8][8];
    float other[8][8];
    float log_other[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): self=TLOAD(input_self,0,0); other=TLOAD(input_other,0,0); log_other=TLOG(other); result=TMUL(self,log_other); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_self[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_other[_row * 8 + _col]);
            vst1q_f32(&other[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&other[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&log_other[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&log_other[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v4, _v5);
            vst1q_f32(&result[_row][_col], _vr6);
            float32x4_t _vs7 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input_self[_row * 8 + _col];
            other[_row][_col] = input_other[_row * 8 + _col];
            log_other[_row][_col] = logf(other[_row][_col]);
            result[_row][_col] = self[_row][_col] * log_other[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}