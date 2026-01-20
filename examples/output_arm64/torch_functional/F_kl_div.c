// PTO Program: F_kl_div
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_kl_div(float* input, float* target_mem, float* output) {
    float log_pred[8][8];
    float target[8][8];
    float log_target[8][8];
    float diff[8][8];
    float kl[8][8];
    float row_sum[8][1];
    float result[1][1];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (5 ops): log_pred=TLOAD(input,0,0); target=TLOAD(target_mem,0,0); log_target=TLOG(target); diff=TSUB(log_target,log_pred); kl=TMUL(target,diff)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&log_pred[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&target[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&log_target[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&log_target[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&log_pred[_row][_col]);
            float32x4_t _vr6 = vsubq_f32(_v4, _v5);
            vst1q_f32(&diff[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr9 = vmulq_f32(_v7, _v8);
            vst1q_f32(&kl[_row][_col], _vr9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            log_pred[_row][_col] = input[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            log_target[_row][_col] = logf(target[_row][_col]);
            diff[_row][_col] = log_target[_row][_col] - log_pred[_row][_col];
            kl[_row][_col] = target[_row][_col] * diff[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(kl)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += kl[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: result = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        result[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(result,64.0f); output=TSTORE(result,0,0)
    float32x4_t _vs10 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v11 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr12 = vdivq_f32(_v11, _vs10);
            vst1q_f32(&result[_row][_col], _vr12);
            float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs13);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = result[_row][_col] / 64.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}