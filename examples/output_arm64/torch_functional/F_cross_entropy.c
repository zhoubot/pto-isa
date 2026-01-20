// PTO Program: F_cross_entropy
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_cross_entropy(float* input, float* target_mem, float* output) {
    float logits[8][8];
    float target[8][8];
    float row_mean[8][1];
    float shifted[8][8];
    float exp_shifted[8][8];
    float row_sum[8][1];
    float log_sum[8][1];
    float log_softmax[8][8];
    float ce[8][8];
    float ce_row[8][1];
    float result[1][1];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (2 ops): logits=TLOAD(input,0,0); target=TLOAD(target_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&logits[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            logits[_row][_col] = input[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
        }
    }

    // TROWSUM: row_mean = rowsum(logits)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += logits[_row][_col];
        }
        row_mean[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_mean=TDIVS(row_mean,8.0f)
    float32x4_t _vs2 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&row_mean[_row][_col]);
            float32x4_t _vr4 = vdivq_f32(_v3, _vs2);
            vst1q_f32(&row_mean[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_mean[_row][_col] = row_mean[_row][_col] / 8.0f;
        }
    }

    // TROWEXPANDSUB: shifted = logits - broadcast(row_mean)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_mean[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            shifted[_row][_col] = logits[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (1 ops): exp_shifted=TEXP(shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v5 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr6 = _v5;
            vst1q_f32(&exp_shifted[_row][_col], _vr6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            exp_shifted[_row][_col] = expf(shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_shifted)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_shifted[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): log_sum=TLOG(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v7 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr8 = _v7;
            vst1q_f32(&log_sum[_row][_col], _vr8);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            log_sum[_row][_col] = logf(row_sum[_row][_col]);
        }
    }

    // TROWEXPANDSUB: log_softmax = shifted - broadcast(log_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = log_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            log_softmax[_row][_col] = shifted[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (2 ops): ce=TMUL(target,log_softmax); ce=TNEG(ce)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v9 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&log_softmax[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v9, _v10);
            vst1q_f32(&ce[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&ce[_row][_col]);
            float32x4_t _vr13 = vnegq_f32(_v12);
            vst1q_f32(&ce[_row][_col], _vr13);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            ce[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
            ce[_row][_col] = -ce[_row][_col];
        }
    }

    // TROWSUM: ce_row = rowsum(ce)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += ce[_row][_col];
        }
        ce_row[_row][0] = _sum;}

    // TCOLSUM: result = colsum(ce_row)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += ce_row[_row][_col];
        }
        result[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(result,8.0f); output=TSTORE(result,0,0)
    float32x4_t _vs14 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v15 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr16 = vdivq_f32(_v15, _vs14);
            vst1q_f32(&result[_row][_col], _vr16);
            float32x4_t _vs17 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs17);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = result[_row][_col] / 8.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}