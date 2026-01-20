// PTO Program: nn_ELU
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_ELU(float* input, float* output) {
    float x[8][8];
    float pos_part[8][8];
    float exp_x[8][8];
    float exp_minus_1[8][8];
    float scaled[8][8];
    float neg_x[8][8];
    float neg_relu[8][8];
    float neg_part[8][8];
    float neg_contrib[8][8];
    float result[8][8];

    // Loop fusion: 9 loop overheads saved

    // FUSED LOOP (10 ops): x=TLOAD(input,0,0); pos_part=TRELU(x); exp_x=TEXP(x); exp_minus_1=TADDS(exp_x,-1.0f); scaled=TMULS(exp_minus_1,1.0f); neg_x=TNEG(scaled); neg_relu=TRELU(neg_x); neg_contrib=TNEG(neg_relu); result=TADD(pos_part,neg_contrib); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1.0f);
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vmaxq_f32(_v3, vdupq_n_f32(0.0f));
            vst1q_f32(&pos_part[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr6 = _v5;
            vst1q_f32(&exp_x[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vr8 = vaddq_f32(_v7, _vs0);
            vst1q_f32(&exp_minus_1[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&exp_minus_1[_row][_col]);
            float32x4_t _vr10 = vmulq_f32(_v9, _vs1);
            vst1q_f32(&scaled[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr12 = vnegq_f32(_v11);
            vst1q_f32(&neg_x[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr14 = vmaxq_f32(_v13, vdupq_n_f32(0.0f));
            vst1q_f32(&neg_relu[_row][_col], _vr14);
            float32x4_t _v15 = vld1q_f32(&neg_relu[_row][_col]);
            float32x4_t _vr16 = vnegq_f32(_v15);
            vst1q_f32(&neg_contrib[_row][_col], _vr16);
            float32x4_t _v17 = vld1q_f32(&pos_part[_row][_col]);
            float32x4_t _v18 = vld1q_f32(&neg_contrib[_row][_col]);
            float32x4_t _vr19 = vaddq_f32(_v17, _v18);
            vst1q_f32(&result[_row][_col], _vr19);
            float32x4_t _vs20 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs20);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
            exp_x[_row][_col] = expf(x[_row][_col]);
            exp_minus_1[_row][_col] = exp_x[_row][_col] + -1.0f;
            scaled[_row][_col] = exp_minus_1[_row][_col] * 1.0f;
            neg_x[_row][_col] = -scaled[_row][_col];
            neg_relu[_row][_col] = fmaxf(neg_x[_row][_col], 0.0f);
            neg_contrib[_row][_col] = -neg_relu[_row][_col];
            result[_row][_col] = pos_part[_row][_col] + neg_contrib[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}