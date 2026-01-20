// PTO Program: F_bilinear
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_bilinear(float* input1, float* input2, float* weight_mem, float* output_mem) {
    float x1[8][8];
    float x2[8][8];
    float weight[8][8];
    float temp[8][8];
    float output[8][8];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (3 ops): x1=TLOAD(input1,0,0); x2=TLOAD(input2,0,0); weight=TLOAD(weight_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input1[_row * 8 + _col]);
            vst1q_f32(&x1[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input2[_row * 8 + _col]);
            vst1q_f32(&x2[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&weight_mem[_row * 8 + _col]);
            vst1q_f32(&weight[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x1[_row][_col] = input1[_row * 8 + _col];
            x2[_row][_col] = input2[_row * 8 + _col];
            weight[_row][_col] = weight_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: temp = x1 @ weight
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += x1[_i][_k] * weight[_k][_j];}
            temp[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): output=TMUL(temp,x2); output_mem=TSTORE(output,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&temp[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v3, _v4);
            vst1q_f32(&output[_row][_col], _vr5);
            float32x4_t _vs6 = vld1q_f32(&output[_row][_col]);
            vst1q_f32(&output_mem[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row][_col] = temp[_row][_col] * x2[_row][_col];
            output_mem[_row * 8 + _col] = output[_row][_col];
        }
    }

}