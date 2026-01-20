// PTO Program: F_threshold
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_threshold(float* input, float* output) {
    float x[8][8];
    float thresh_tile[8][8];
    float value_tile[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): x=TLOAD(input,0,0); thresh_tile=TEXPANDS(0.0f); value_tile=TEXPANDS(0.0f); result=TMAX(x,thresh_tile); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            vst1q_f32(&thresh_tile[_row][_col], _vs0);
            vst1q_f32(&value_tile[_row][_col], _vs0);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&thresh_tile[_row][_col]);
            float32x4_t _vr4 = vmaxq_f32(_v2, _v3);
            vst1q_f32(&result[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            thresh_tile[_row][_col] = 0.0f;
            value_tile[_row][_col] = 0.0f;
            result[_row][_col] = x[_row][_col] + thresh_tile[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}