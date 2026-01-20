// PTO Program: score_to_weight
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void score_to_weight(float* scores_mem, float* weights_mem) {
    float scores[8][8];
    float row_sum[8][1];
    float shifted[8][8];
    float exp_scores[8][8];
    float weights[8][8];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): scores=TLOAD(scores_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&scores_mem[_row * 8 + _col]);
            vst1q_f32(&scores[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            scores[_row][_col] = scores_mem[_row * 8 + _col];
        }
    }

    // TROWSUM: row_sum = rowsum(scores)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += scores[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_sum=TDIVS(row_sum,8.0f)
    float32x4_t _vs1 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v2 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
            vst1q_f32(&row_sum[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
        }
    }

    // TROWEXPANDSUB: shifted = scores - broadcast(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            shifted[_row][_col] = scores[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (1 ops): exp_scores=TEXP(shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_scores[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            exp_scores[_row][_col] = expf(shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_scores)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_scores[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TROWEXPANDDIV: weights = exp_scores / broadcast(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            weights[_row][_col] = exp_scores[_row][_col] / _broadcast_val;
        }}

    // FUSED LOOP (1 ops): weights_mem=TSTORE(weights,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs6 = vld1q_f32(&weights[_row][_col]);
            vst1q_f32(&weights_mem[_row * 8 + _col], _vs6);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            weights_mem[_row * 8 + _col] = weights[_row][_col];
        }
    }

}