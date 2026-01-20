// PTO Program: F_nll_loss
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_nll_loss(float* input, float* target_mem, float* output) {
    float log_probs[8][8];
    float target[8][8];
    float weighted[8][8];
    float row_sum[8][1];
    float result[1][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (3 ops): log_probs=TLOAD(input,0,0); target=TLOAD(target_mem,0,0); weighted=TMUL(target,log_probs)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&log_probs[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&log_probs[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v2, _v3);
            vst1q_f32(&weighted[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            log_probs[_row][_col] = input[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            weighted[_row][_col] = target[_row][_col] * log_probs[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(weighted)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += weighted[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: result = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        result[0][_col] = _sum;}

    // FUSED LOOP (3 ops): result=TNEG(result); result=TDIVS(result,8.0f); output=TSTORE(result,0,0)
    float32x4_t _vs5 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v6 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr7 = vnegq_f32(_v6);
            vst1q_f32(&result[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr9 = vdivq_f32(_v8, _vs5);
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs10);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = -result[_row][_col];
            result[_row][_col] = result[_row][_col] / 8.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}