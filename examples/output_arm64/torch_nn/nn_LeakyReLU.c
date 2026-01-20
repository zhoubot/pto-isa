// PTO Program: nn_LeakyReLU
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_LeakyReLU(float* input, float* output) {
    float x[8][8];
    float pos_part[8][8];
    float neg_x[8][8];
    float neg_relu[8][8];
    float neg_part[8][8];
    float scaled_neg[8][8];
    float result[8][8];

    // Loop fusion: 7 loop overheads saved

    // FUSED LOOP (8 ops): x=TLOAD(input,0,0); pos_part=TRELU(x); neg_x=TNEG(x); neg_relu=TRELU(neg_x); neg_part=TNEG(neg_relu); scaled_neg=TMULS(neg_part,0.01f); result=TADD(pos_part,scaled_neg); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.01f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vmaxq_f32(_v2, vdupq_n_f32(0.0f));
            vst1q_f32(&pos_part[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = vnegq_f32(_v4);
            vst1q_f32(&neg_x[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr7 = vmaxq_f32(_v6, vdupq_n_f32(0.0f));
            vst1q_f32(&neg_relu[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&neg_relu[_row][_col]);
            float32x4_t _vr9 = vnegq_f32(_v8);
            vst1q_f32(&neg_part[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&neg_part[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v10, _vs0);
            vst1q_f32(&scaled_neg[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&pos_part[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&scaled_neg[_row][_col]);
            float32x4_t _vr14 = vaddq_f32(_v12, _v13);
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
            neg_x[_row][_col] = -x[_row][_col];
            neg_relu[_row][_col] = fmaxf(neg_x[_row][_col], 0.0f);
            neg_part[_row][_col] = -neg_relu[_row][_col];
            scaled_neg[_row][_col] = neg_part[_row][_col] * 0.01f;
            result[_row][_col] = pos_part[_row][_col] + scaled_neg[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}