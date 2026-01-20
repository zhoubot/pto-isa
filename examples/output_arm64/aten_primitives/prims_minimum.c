// PTO Program: prims_minimum
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void prims_minimum(float* input_a, float* input_b, float* output) {
    float a[8][8];
    float b[8][8];
    float result[8][8];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (4 ops): a=TLOAD(input_a,0,0); b=TLOAD(input_b,0,0); result=TMIN(a,b); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_a[_row * 8 + _col]);
            vst1q_f32(&a[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_b[_row * 8 + _col]);
            vst1q_f32(&b[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&a[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&b[_row][_col]);
            float32x4_t _vr4 = vminq_f32(_v2, _v3);
            vst1q_f32(&result[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            a[_row][_col] = input_a[_row * 8 + _col];
            b[_row][_col] = input_b[_row * 8 + _col];
            result[_row][_col] = a[_row][_col] + b[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}