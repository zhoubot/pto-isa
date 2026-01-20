// PTO Program: tensor_prod
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_prod(float* input, float* output) {
    float self[8][8];
    float log_self[8][8];
    float row_sum[8][1];
    float total[1][1];
    float result[1][1];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (2 ops): self=TLOAD(input,0,0); log_self=TLOG(self)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&log_self[_row][_col], _vr2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            log_self[_row][_col] = logf(self[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(log_self)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += log_self[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TEXP(total); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&total[_row][_col]);
            float32x4_t _vr4 = _v3;
            vst1q_f32(&result[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = expf(total[_row][_col]);
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}