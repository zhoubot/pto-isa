// PTO Program: tensor_lerp
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_lerp(float* input_self, float* input_end, float* output) {
    float self[8][8];
    float end[8][8];
    float diff[8][8];
    float scaled[8][8];
    float result[8][8];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): self=TLOAD(input_self,0,0); end=TLOAD(input_end,0,0); diff=TSUB(end,self); scaled=TMULS(diff,0.5f); result=TADD(self,scaled); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.5f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_self[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&input_end[_row * 8 + _col]);
            vst1q_f32(&end[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&end[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr5 = vsubq_f32(_v3, _v4);
            vst1q_f32(&diff[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v6, _vs0);
            vst1q_f32(&scaled[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v8, _v9);
            vst1q_f32(&result[_row][_col], _vr10);
            float32x4_t _vs11 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs11);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input_self[_row * 8 + _col];
            end[_row][_col] = input_end[_row * 8 + _col];
            diff[_row][_col] = end[_row][_col] - self[_row][_col];
            scaled[_row][_col] = diff[_row][_col] * 0.5f;
            result[_row][_col] = self[_row][_col] + scaled[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}