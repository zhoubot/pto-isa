// PTO Program: nn_LogSoftmax
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_LogSoftmax(float* input, float* output) {
    float x[8][8];
    float exp_x[8][8];
    float sum_exp[8][1];
    float log_sum[8][1];
    float result[8][8];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (2 ops): x=TLOAD(input,0,0); exp_x=TEXP(x)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&exp_x[_row][_col], _vr2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
        }
    }

    // TROWSUM: sum_exp = rowsum(exp_x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_x[_row][_col];
        }
        sum_exp[_row][0] = _sum;}

    // FUSED LOOP (1 ops): log_sum=TLOG(sum_exp)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&sum_exp[_row][_col]);
            float32x4_t _vr4 = _v3;
            vst1q_f32(&log_sum[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            log_sum[_row][_col] = logf(sum_exp[_row][_col]);
        }
    }

    // TROWEXPANDSUB: result = x - broadcast(log_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = log_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}