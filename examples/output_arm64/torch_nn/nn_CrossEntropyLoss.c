// PTO Program: nn_CrossEntropyLoss
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_CrossEntropyLoss(float* pred_mem, float* target_mem, float* output) {
    float pred[8][8];
    float target[8][8];
    float exp_pred[8][8];
    float sum_exp[8][1];
    float log_sum[8][1];
    float log_softmax[8][8];
    float weighted[8][8];
    float neg_weighted[8][8];
    float row_sum[8][1];
    float total_sum[1][1];
    float result[1][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (3 ops): pred=TLOAD(pred_mem,0,0); target=TLOAD(target_mem,0,0); exp_pred=TEXP(pred)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&pred_mem[_row * 8 + _col]);
            vst1q_f32(&pred[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&exp_pred[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            pred[_row][_col] = pred_mem[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            exp_pred[_row][_col] = expf(pred[_row][_col]);
        }
    }

    // TROWSUM: sum_exp = rowsum(exp_pred)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_pred[_row][_col];
        }
        sum_exp[_row][0] = _sum;}

    // FUSED LOOP (1 ops): log_sum=TLOG(sum_exp)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v4 = vld1q_f32(&sum_exp[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&log_sum[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            log_sum[_row][_col] = logf(sum_exp[_row][_col]);
        }
    }

    // TROWEXPANDSUB: log_softmax = pred - broadcast(log_sum)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = log_sum[_row][0];
        for (int _col = 0; _col < 8; _col++) {
            log_softmax[_row][_col] = pred[_row][_col] - _broadcast_val;
        }}

    // FUSED LOOP (2 ops): weighted=TMUL(target,log_softmax); neg_weighted=TNEG(weighted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v6 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&log_softmax[_row][_col]);
            float32x4_t _vr8 = vmulq_f32(_v6, _v7);
            vst1q_f32(&weighted[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&weighted[_row][_col]);
            float32x4_t _vr10 = vnegq_f32(_v9);
            vst1q_f32(&neg_weighted[_row][_col], _vr10);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            weighted[_row][_col] = target[_row][_col] * log_softmax[_row][_col];
            neg_weighted[_row][_col] = -weighted[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(neg_weighted)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += neg_weighted[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total_sum = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total_sum[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(total_sum,8.0f); output=TSTORE(result,0,0)
    float32x4_t _vs11 = vdupq_n_f32(8.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v12 = vld1q_f32(&total_sum[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs11);
            vst1q_f32(&result[_row][_col], _vr13);
            float32x4_t _vs14 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs14);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = total_sum[_row][_col] / 8.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}