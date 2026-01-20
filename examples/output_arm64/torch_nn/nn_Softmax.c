// PTO Program: nn_Softmax
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_Softmax(float* input, float* output) {
    float x[8][8];
    float exp_x[8][8];
    float sum_exp[8][1];
    float result[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (2 ops): x=TLOAD(input,0,0); exp_x=TEXP(x)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&exp_x[_row][_col], _vr2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
        }
    }

    // TROWSUM: sum_exp = rowsum(exp_x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_x[_row][_col];
        }
        sum_exp[_row][0] = _sum;}

    // FUSED LOOP (2 ops): result=TDIV(exp_x,sum_exp); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&sum_exp[_row][_col]);
            float32x4_t _vr5 = vdivq_f32(_v3, _v4);
            vst1q_f32(&result[_row][_col], _vr5);
            float32x4_t _vs6 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = exp_x[_row][_col] / sum_exp[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}