// PTO Program: nn_Hardswish
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_Hardswish(float* input, float* output) {
    float x[8][8];
    float x_plus_3[8][8];
    float relu_out[8][8];
    float six[8][8];
    float relu6_out[8][8];
    float scaled[8][8];
    float result[8][8];

    // Loop fusion: 7 loop overheads saved

    // FUSED LOOP (8 ops): x=TLOAD(input,0,0); x_plus_3=TADDS(x,3.0f); relu_out=TRELU(x_plus_3); six=TEXPANDS(6.0f); relu6_out=TMIN(relu_out,six); scaled=TMUL(x,relu6_out); result=TDIVS(scaled,6.0f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(3.0f);
    float32x4_t _vs1 = vdupq_n_f32(6.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vaddq_f32(_v3, _vs0);
            vst1q_f32(&x_plus_3[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&x_plus_3[_row][_col]);
            float32x4_t _vr6 = vmaxq_f32(_v5, vdupq_n_f32(0.0f));
            vst1q_f32(&relu_out[_row][_col], _vr6);
            vst1q_f32(&six[_row][_col], _vs1);
            float32x4_t _v7 = vld1q_f32(&relu_out[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&six[_row][_col]);
            float32x4_t _vr9 = vminq_f32(_v7, _v8);
            vst1q_f32(&relu6_out[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&relu6_out[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v10, _v11);
            vst1q_f32(&scaled[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr14 = vdivq_f32(_v13, _vs1);
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_plus_3[_row][_col] = x[_row][_col] + 3.0f;
            relu_out[_row][_col] = fmaxf(x_plus_3[_row][_col], 0.0f);
            six[_row][_col] = 6.0f;
            relu6_out[_row][_col] = relu_out[_row][_col] + six[_row][_col];
            scaled[_row][_col] = x[_row][_col] * relu6_out[_row][_col];
            result[_row][_col] = scaled[_row][_col] / 6.0f;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}