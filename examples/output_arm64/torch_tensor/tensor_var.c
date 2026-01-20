// PTO Program: tensor_var
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_var(float* input, float* output) {
    float self[8][8];
    float row_sum[8][1];
    float total[1][1];
    float mean_val[8][8];
    float centered[8][8];
    float sq_centered[8][8];
    float sq_row_sum[8][1];
    float var_total[1][1];
    float result[1][1];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (1 ops): self=TLOAD(input,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
        }
    }

    // TROWSUM: row_sum = rowsum(self)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += self[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TCOLSUM: total = colsum(row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += row_sum[_row][_col];
        }
        total[0][_col] = _sum;}

    // FUSED LOOP (1 ops): total=TDIVS(total,64.0f)
    float32x4_t _vs1 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v2 = vld1q_f32(&total[_row][_col]);
            float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
            vst1q_f32(&total[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            total[_row][_col] = total[_row][_col] / 64.0f;
        }
    }

    // FUSED LOOP (3 ops): mean_val=TEXPANDS(0.0f); centered=TSUB(self,mean_val); sq_centered=TMUL(centered,centered)
    float32x4_t _vs4 = vdupq_n_f32(0.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            vst1q_f32(&mean_val[_row][_col], _vs4);
            float32x4_t _v5 = vld1q_f32(&self[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&mean_val[_row][_col]);
            float32x4_t _vr7 = vsubq_f32(_v5, _v6);
            vst1q_f32(&centered[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&centered[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&centered[_row][_col]);
            float32x4_t _vr10 = vmulq_f32(_v8, _v9);
            vst1q_f32(&sq_centered[_row][_col], _vr10);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            mean_val[_row][_col] = 0.0f;
            centered[_row][_col] = self[_row][_col] - mean_val[_row][_col];
            sq_centered[_row][_col] = centered[_row][_col] * centered[_row][_col];
        }
    }

    // TROWSUM: sq_row_sum = rowsum(sq_centered)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += sq_centered[_row][_col];
        }
        sq_row_sum[_row][0] = _sum;}

    // TCOLSUM: var_total = colsum(sq_row_sum)
    for (int _col = 0; _col < 1; _col++) {
        float _sum = 0.0f;
        for (int _row = 0; _row < 8; _row++) {
            _sum += sq_row_sum[_row][_col];
        }
        var_total[0][_col] = _sum;}

    // FUSED LOOP (2 ops): result=TDIVS(var_total,64.0f); output=TSTORE(result,0,0)
    float32x4_t _vs11 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v12 = vld1q_f32(&var_total[_row][_col]);
            float32x4_t _vr13 = vdivq_f32(_v12, _vs11);
            vst1q_f32(&result[_row][_col], _vr13);
            float32x4_t _vs14 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs14);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = var_total[_row][_col] / 64.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}