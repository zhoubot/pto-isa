// PTO Program: F_selu
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_selu(float* input, float* output) {
    float x[8][8];
    float pos_part[8][8];
    float exp_x[8][8];
    float exp_minus_one[8][8];
    float alpha_scaled[8][8];
    float zeros[8][8];
    float neg_part[8][8];
    float elu_result[8][8];
    float result[8][8];

    // Loop fusion: 9 loop overheads saved

    // FUSED LOOP (10 ops): x=TLOAD(input,0,0); pos_part=TRELU(x); exp_x=TEXP(x); exp_minus_one=TADDS(exp_x,-1.0f); alpha_scaled=TMULS(exp_minus_one,1.6732632423543772f); zeros=TEXPANDS(0.0f); neg_part=TMIN(alpha_scaled,zeros); elu_result=TADD(pos_part,neg_part); result=TMULS(elu_result,1.0507009873554805f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(-1.0f);
    float32x4_t _vs1 = vdupq_n_f32(1.6732632423543772f);
    float32x4_t _vs2 = vdupq_n_f32(0.0f);
    float32x4_t _vs3 = vdupq_n_f32(1.0507009873554805f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl4 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl4);
            float32x4_t _v5 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr6 = vmaxq_f32(_v5, vdupq_n_f32(0.0f));
            vst1q_f32(&pos_part[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr8 = _v7;
            vst1q_f32(&exp_x[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v9, _vs0);
            vst1q_f32(&exp_minus_one[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&exp_minus_one[_row][_col]);
            float32x4_t _vr12 = vmulq_f32(_v11, _vs1);
            vst1q_f32(&alpha_scaled[_row][_col], _vr12);
            vst1q_f32(&zeros[_row][_col], _vs2);
            float32x4_t _v13 = vld1q_f32(&alpha_scaled[_row][_col]);
            float32x4_t _v14 = vld1q_f32(&zeros[_row][_col]);
            float32x4_t _vr15 = vminq_f32(_v13, _v14);
            vst1q_f32(&neg_part[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&pos_part[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&neg_part[_row][_col]);
            float32x4_t _vr18 = vaddq_f32(_v16, _v17);
            vst1q_f32(&elu_result[_row][_col], _vr18);
            float32x4_t _v19 = vld1q_f32(&elu_result[_row][_col]);
            float32x4_t _vr20 = vmulq_f32(_v19, _vs3);
            vst1q_f32(&result[_row][_col], _vr20);
            float32x4_t _vs21 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs21);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            pos_part[_row][_col] = fmaxf(x[_row][_col], 0.0f);
            exp_x[_row][_col] = expf(x[_row][_col]);
            exp_minus_one[_row][_col] = exp_x[_row][_col] + -1.0f;
            alpha_scaled[_row][_col] = exp_minus_one[_row][_col] * 1.6732632423543772f;
            zeros[_row][_col] = 0.0f;
            neg_part[_row][_col] = alpha_scaled[_row][_col] + zeros[_row][_col];
            elu_result[_row][_col] = pos_part[_row][_col] + neg_part[_row][_col];
            result[_row][_col] = elu_result[_row][_col] * 1.0507009873554805f;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}