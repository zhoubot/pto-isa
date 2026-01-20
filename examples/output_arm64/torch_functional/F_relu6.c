// PTO Program: F_relu6
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_relu6(float* input, float* output) {
    float x[8][8];
    float relu_out[8][8];
    float six[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): x=TLOAD(input,0,0); relu_out=TRELU(x); six=TEXPANDS(6.0f); result=TMIN(relu_out,six); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(6.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vmaxq_f32(_v2, vdupq_n_f32(0.0f));
            vst1q_f32(&relu_out[_row][_col], _vr3);
            vst1q_f32(&six[_row][_col], _vs0);
            float32x4_t _v4 = vld1q_f32(&relu_out[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&six[_row][_col]);
            float32x4_t _vr6 = vminq_f32(_v4, _v5);
            vst1q_f32(&result[_row][_col], _vr6);
            float32x4_t _vs7 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            relu_out[_row][_col] = fmaxf(x[_row][_col], 0.0f);
            six[_row][_col] = 6.0f;
            result[_row][_col] = relu_out[_row][_col] + six[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}