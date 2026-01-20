// PTO Program: tensor_hypot
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_hypot(float* input_self, float* input_other, float* output) {
    float self[8][8];
    float other[8][8];
    float self_sq[8][8];
    float other_sq[8][8];
    float sum_sq[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): self=TLOAD(input_self,0,0); other=TLOAD(input_other,0,0); self_sq=TMUL(self,self); other_sq=TMUL(other,other); sum_sq=TADD(self_sq,other_sq); result=TSQRT(sum_sq); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_self[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_other[_row * 8 + _col]);
            vst1q_f32(&other[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v2, _v3);
            vst1q_f32(&self_sq[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&other[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&other[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v5, _v6);
            vst1q_f32(&other_sq[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&self_sq[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&other_sq[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v8, _v9);
            vst1q_f32(&sum_sq[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&sum_sq[_row][_col]);
            float32x4_t _vr12 = vsqrtq_f32(_v11);
            vst1q_f32(&result[_row][_col], _vr12);
            float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs13);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input_self[_row * 8 + _col];
            other[_row][_col] = input_other[_row * 8 + _col];
            self_sq[_row][_col] = self[_row][_col] * self[_row][_col];
            other_sq[_row][_col] = other[_row][_col] * other[_row][_col];
            sum_sq[_row][_col] = self_sq[_row][_col] + other_sq[_row][_col];
            result[_row][_col] = sqrtf(sum_sq[_row][_col]);
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}