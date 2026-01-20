// PTO Program: aten_relu
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_relu(float* input, float* output) {
    float x[8][8];
    float result[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): x=TLOAD(input,0,0); result=TRELU(x); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = vmaxq_f32(_v1, vdupq_n_f32(0.0f));
            vst1q_f32(&result[_row][_col], _vr2);
            float32x4_t _vs3 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            result[_row][_col] = fmaxf(x[_row][_col], 0.0f);
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}