// PTO Program: aten_mm
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_mm(float* input_a, float* input_b, float* output) {
    float a[8][8];
    float b[8][8];
    float result[8][8];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (2 ops): a=TLOAD(input_a,0,0); b=TLOAD(input_b,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_a[_row * 8 + _col]);
            vst1q_f32(&a[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_b[_row * 8 + _col]);
            vst1q_f32(&b[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            a[_row][_col] = input_a[_row * 8 + _col];
            b[_row][_col] = input_b[_row * 8 + _col];
        }
    }

    // TMATMUL: result = a @ b
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += a[_i][_k] * b[_k][_j];}
            result[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}