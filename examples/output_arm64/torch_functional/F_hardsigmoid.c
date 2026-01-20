// PTO Program: F_hardsigmoid
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_hardsigmoid(float* input, float* output) {
    float x[8][8];
    float x_plus_3[8][8];
    float scaled[8][8];
    float zeros[8][8];
    float ones[8][8];
    float clamp_low[8][8];
    float result[8][8];

    // Loop fusion: 7 loop overheads saved

    // FUSED LOOP (8 ops): x=TLOAD(input,0,0); x_plus_3=TADDS(x,3.0f); scaled=TDIVS(x_plus_3,6.0f); zeros=TEXPANDS(0.0f); ones=TEXPANDS(1.0f); clamp_low=TMAX(scaled,zeros); result=TMIN(clamp_low,ones); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(3.0f);
    float32x4_t _vs1 = vdupq_n_f32(6.0f);
    float32x4_t _vs2 = vdupq_n_f32(0.0f);
    float32x4_t _vs3 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl4 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl4);
            float32x4_t _v5 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr6 = vaddq_f32(_v5, _vs0);
            vst1q_f32(&x_plus_3[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&x_plus_3[_row][_col]);
            float32x4_t _vr8 = vdivq_f32(_v7, _vs1);
            vst1q_f32(&scaled[_row][_col], _vr8);
            vst1q_f32(&zeros[_row][_col], _vs2);
            vst1q_f32(&ones[_row][_col], _vs3);
            float32x4_t _v9 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&zeros[_row][_col]);
            float32x4_t _vr11 = vmaxq_f32(_v9, _v10);
            vst1q_f32(&clamp_low[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&clamp_low[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&ones[_row][_col]);
            float32x4_t _vr14 = vminq_f32(_v12, _v13);
            vst1q_f32(&result[_row][_col], _vr14);
            float32x4_t _vs15 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            x_plus_3[_row][_col] = x[_row][_col] + 3.0f;
            scaled[_row][_col] = x_plus_3[_row][_col] / 6.0f;
            zeros[_row][_col] = 0.0f;
            ones[_row][_col] = 1.0f;
            clamp_low[_row][_col] = scaled[_row][_col] + zeros[_row][_col];
            result[_row][_col] = clamp_low[_row][_col] + ones[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}