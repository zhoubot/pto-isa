// PTO Program: F_smooth_l1_loss
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_smooth_l1_loss(float* input, float* target_mem, float* output) {
    float pred[8][8];
    float target[8][8];
    float diff[8][8];
    float abs_diff[8][8];
    float sq_diff[8][8];
    float l2_part[8][8];
    float l1_part[8][8];
    float beta_tile[8][8];
    float loss[8][8];
    float row_sum[8][1];
    float result[1][1];

    // Loop fusion: 8 loop overheads saved

    // FUSED LOOP (8 ops): pred=TLOAD(input,0,0); target=TLOAD(target_mem,0,0); diff=TSUB(pred,target); abs_diff=TABS(diff); sq_diff=TMUL(diff,diff); l2_part=TDIVS(sq_diff,2.0f); l1_part=TADDS(abs_diff,-0.5f); loss=TMIN(l2_part,l1_part)
    float32x4_t _vs0 = vdupq_n_f32(2.0f);
    float32x4_t _vs1 = vdupq_n_f32(-0.5f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&pred[_row][_col], _vl2);
            float32x4_t _vl3 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _v5 = vld1q_f32(&target[_row][_col]);
            float32x4_t _vr6 = vsubq_f32(_v4, _v5);
            vst1q_f32(&diff[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr8 = vabsq_f32(_v7);
            vst1q_f32(&abs_diff[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v9, _v10);
            vst1q_f32(&sq_diff[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&sq_diff[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs0);
            vst1q_f32(&l2_part[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&abs_diff[_row][_col]);
            float32x4_t _vr15 = vaddq_f32(_v14, _vs1);
            vst1q_f32(&l1_part[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&l2_part[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&l1_part[_row][_col]);
            float32x4_t _vr18 = vminq_f32(_v16, _v17);
            vst1q_f32(&loss[_row][_col], _vr18);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            pred[_row][_col] = input[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            diff[_row][_col] = pred[_row][_col] - target[_row][_col];
            abs_diff[_row][_col] = fabsf(diff[_row][_col]);
            sq_diff[_row][_col] = diff[_row][_col] * diff[_row][_col];
            l2_part[_row][_col] = sq_diff[_row][_col] / 2.0f;
            l1_part[_row][_col] = abs_diff[_row][_col] + -0.5f;
            loss[_row][_col] = l2_part[_row][_col] + l1_part[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(loss)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += loss[_row][_col];
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
    float32x4_t _vs19 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v20 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr21 = vdivq_f32(_v20, _vs19);
            vst1q_f32(&result[_row][_col], _vr21);
            float32x4_t _vs22 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs22);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = result[_row][_col] / 64.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}