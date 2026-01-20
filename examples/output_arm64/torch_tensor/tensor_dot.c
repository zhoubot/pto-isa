// PTO Program: tensor_dot
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_dot(float* input_self, float* input_other, float* output) {
    float self[1][64];
    float other[1][64];
    float prod[1][64];
    float result[1][1];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): self=TLOAD(input_self,0,0); other=TLOAD(input_other,0,0); prod=TMUL(self,other)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 64; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_self[_row * 64 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_other[_row * 64 + _col]);
            vst1q_f32(&other[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&other[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v2, _v3);
            vst1q_f32(&prod[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 64; _col++) {
            self[_row][_col] = input_self[_row * 64 + _col];
            other[_row][_col] = input_other[_row * 64 + _col];
            prod[_row][_col] = self[_row][_col] * other[_row][_col];
        }
    }

    // TROWSUM: result = rowsum(prod)
    for (int _row = 0; _row < 1; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 64; _col++) {
            _sum += prod[_row][_col];
        }
        result[_row][0] = _sum;}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}