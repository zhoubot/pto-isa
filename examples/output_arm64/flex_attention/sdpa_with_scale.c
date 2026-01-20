// PTO Program: sdpa_with_scale
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void sdpa_with_scale(float* Q_mem, float* K_mem, float* V_mem, float* output_mem) {
    float Q[8][8];
    float K[8][8];
    float V[8][8];
    float scores[8][8];
    float scaled[8][8];
    float row_sum[8][1];
    float shifted[8][8];
    float exp_scores[8][8];
    float attn[8][8];
    float output[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): Q=TLOAD(Q_mem,0,0); K=TLOAD(K_mem,0,0); V=TLOAD(V_mem,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&Q_mem[_row * 8 + _col]);
            vst1q_f32(&Q[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&K_mem[_row * 8 + _col]);
            vst1q_f32(&K[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&V_mem[_row * 8 + _col]);
            vst1q_f32(&V[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            Q[_row][_col] = Q_mem[_row * 8 + _col];
            K[_row][_col] = K_mem[_row * 8 + _col];
            V[_row][_col] = V_mem[_row * 8 + _col];
        }
    }

    // TMATMUL: scores = Q @ K
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += Q[_i][_k] * K[_k][_j];}
            scores[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): scaled=TMULS(scores,0.35355339059327373f)
    float32x4_t _vs3 = vdupq_n_f32(0.35355339059327373f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&scores[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v4, _vs3);
            vst1q_f32(&scaled[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            scaled[_row][_col] = scores[_row][_col] * 0.35355339059327373f;
        }
    }

    // TROWSUM: row_sum = rowsum(scaled)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += scaled[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): row_sum=TDIVS(row_sum,8.0f)
    float32x4_t _vs6 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v7 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr8 = vdivq_f32(_v7, _vs6);
            vst1q_f32(&row_sum[_row][_col], _vr8);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_sum[_row][_col] = row_sum[_row][_col] / 8.0f;
        }
    }

    // TROWEXPANDSUB: shifted = scaled - broadcast(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            shifted[_row][_col] = scaled[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (1 ops): exp_scores=TEXP(shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v9 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr10 = _v9;
            vst1q_f32(&exp_scores[_row][_col], _vr10);
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

    // TROWEXPANDDIV: attn = exp_scores / broadcast(row_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            attn[_row][_col] = exp_scores[_row][_col] / _broadcast_val;
        }}

    // TMATMUL: output = attn @ V
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += attn[_i][_k] * V[_k][_j];}
            output[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output_mem=TSTORE(output,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs11 = vld1q_f32(&output[_row][_col]);
            vst1q_f32(&output_mem[_row * 8 + _col], _vs11);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output_mem[_row * 8 + _col] = output[_row][_col];
        }
    }

}