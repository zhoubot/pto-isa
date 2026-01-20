// PTO Program: F_batch_norm
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_batch_norm(float* input, float* mean_mem, float* var_mem, float* output) {
    float x[8][8];
    float mean[1][8];
    float var[1][8];
    float std[1][8];
    float centered[8][8];
    float result[8][8];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
        }
    }

    // FUSED LOOP (2 ops): mean=TLOAD(mean_mem,0,0); var=TLOAD(var_mem,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&mean_mem[_row * 8 + _col]);
            vst1q_f32(&mean[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&var_mem[_row * 8 + _col]);
            vst1q_f32(&var[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            mean[_row][_col] = mean_mem[_row * 8 + _col];
            var[_row][_col] = var_mem[_row * 8 + _col];
        }
    }

    // TCOLSUM: centered = colsum(x)
    for (int _col = 0; _col < 8; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += x[_row][_col];
        }
        centered[0][_col] = _sum;}

    // FUSED LOOP (2 ops): var=TADDS(var,1e-05f); std=TSQRT(var)
    float32x4_t _vs3 = vdupq_n_f32(1e-05f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&var[_row][_col]);
            float32x4_t _vr5 = vaddq_f32(_v4, _vs3);
            vst1q_f32(&var[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&var[_row][_col]);
            float32x4_t _vr7 = vsqrtq_f32(_v6);
            vst1q_f32(&std[_row][_col], _vr7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            var[_row][_col] = var[_row][_col] + 1e-05f;
            std[_row][_col] = sqrtf(var[_row][_col]);
        }
    }

    // FUSED LOOP (2 ops): result=TDIV(x,std); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&std[_row][_col]);
            float32x4_t _vr10 = vdivq_f32(_v8, _v9);
            vst1q_f32(&result[_row][_col], _vr10);
            float32x4_t _vs11 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs11);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] / std[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}