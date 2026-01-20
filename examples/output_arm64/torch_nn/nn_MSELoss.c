// PTO Program: nn_MSELoss
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void nn_MSELoss(float* pred_mem, float* target_mem, float* output) {
    float pred[8][8];
    float target[8][8];
    float diff[8][8];
    float squared[8][8];
    float row_sum[8][1];
    float total_sum[1][1];
    float result[1][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (4 ops): pred=TLOAD(pred_mem,0,0); target=TLOAD(target_mem,0,0); diff=TSUB(pred,target); squared=TMUL(diff,diff)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&pred_mem[_row * 8 + _col]);
            vst1q_f32(&pred[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&target[_row][_col]);
            float32x4_t _vr4 = vsubq_f32(_v2, _v3);
            vst1q_f32(&diff[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v5, _v6);
            vst1q_f32(&squared[_row][_col], _vr7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            pred[_row][_col] = pred_mem[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            diff[_row][_col] = pred[_row][_col] - target[_row][_col];
            squared[_row][_col] = diff[_row][_col] * diff[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(squared)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += squared[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total_sum = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total_sum[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(total_sum,64.0f); output=TSTORE(result,0,0)
    float32x4_t _vs8 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v9 = vld1q_f32(&total_sum[_row][_col]);
            float32x4_t _vr10 = vdivq_f32(_v9, _vs8);
            vst1q_f32(&result[_row][_col], _vr10);
            float32x4_t _vs11 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs11);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = total_sum[_row][_col] / 64.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}