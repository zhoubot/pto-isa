// PTO Program: tensor_mv
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_mv(float* input_self, float* input_vec, float* output) {
    float self[8][8];
    float vec[8][1];
    float result[8][1];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): self=TLOAD(input_self,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_self[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input_self[_row * 8 + _col];
        }
    }

    // FUSED LOOP (1 ops): vec=TLOAD(input_vec,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_vec[_row * 1 + _col]);
            vst1q_f32(&vec[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            vec[_row][_col] = input_vec[_row * 1 + _col];
        }
    }

    // TMATMUL: result = self @ vec
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 1; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += self[_i][_k] * vec[_k][_j];}
            result[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}