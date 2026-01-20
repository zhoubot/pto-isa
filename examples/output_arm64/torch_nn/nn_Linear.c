// PTO Program: nn_Linear
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_Linear(float* input, float* weight_mem, float* bias_mem, float* output) {
    float x[8][8];
    float weight[8][8];
    float bias[8][8];
    float mm_result[8][8];
    float result[8][8];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (3 ops): x=TLOAD(input,0,0); weight=TLOAD(weight_mem,0,0); bias=TLOAD(bias_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&weight_mem[_row * 8 + _col]);
            vst1q_f32(&weight[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&bias_mem[_row * 8 + _col]);
            vst1q_f32(&bias[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            weight[_row][_col] = weight_mem[_row * 8 + _col];
            bias[_row][_col] = bias_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: mm_result = x @ weight
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += x[_i][_k] * weight[_k][_j];}
            mm_result[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): result=TADD(mm_result,bias); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&mm_result[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&bias[_row][_col]);
            float32x4_t _vr5 = vaddq_f32(_v3, _v4);
            vst1q_f32(&result[_row][_col], _vr5);
            float32x4_t _vs6 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = mm_result[_row][_col] + bias[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}