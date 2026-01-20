// PTO Program: tensor_addcmul
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_addcmul(float* input_self, float* input_t1, float* input_t2, float* output) {
    float self[8][8];
    float tensor1[8][8];
    float tensor2[8][8];
    float prod[8][8];
    float scaled[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): self=TLOAD(input_self,0,0); tensor1=TLOAD(input_t1,0,0); tensor2=TLOAD(input_t2,0,0); prod=TMUL(tensor1,tensor2); scaled=TMULS(prod,1.0f); result=TADD(self,scaled); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_self[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&input_t1[_row * 8 + _col]);
            vst1q_f32(&tensor1[_row][_col], _vl2);
            float32x4_t _vl3 = vld1q_f32(&input_t2[_row * 8 + _col]);
            vst1q_f32(&tensor2[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&tensor1[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&tensor2[_row][_col]);
            float32x4_t _vr6 = vmulq_f32(_v4, _v5);
            vst1q_f32(&prod[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&prod[_row][_col]);
            float32x4_t _vr8 = vmulq_f32(_v7, _vs0);
            vst1q_f32(&scaled[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr11 = vaddq_f32(_v9, _v10);
            vst1q_f32(&result[_row][_col], _vr11);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input_self[_row * 8 + _col];
            tensor1[_row][_col] = input_t1[_row * 8 + _col];
            tensor2[_row][_col] = input_t2[_row * 8 + _col];
            prod[_row][_col] = tensor1[_row][_col] * tensor2[_row][_col];
            scaled[_row][_col] = prod[_row][_col] * 1.0f;
            result[_row][_col] = self[_row][_col] + scaled[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}