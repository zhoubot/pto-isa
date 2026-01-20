// PTO Program: F_pairwise_distance
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_pairwise_distance(float* input1, float* input2, float* output) {
    float x1[8][8];
    float x2[8][8];
    float diff[8][8];
    float sq_diff[8][8];
    float row_sum[8][1];
    float result[8][1];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (4 ops): x1=TLOAD(input1,0,0); x2=TLOAD(input2,0,0); diff=TSUB(x1,x2); sq_diff=TMUL(diff,diff)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input1[_row * 8 + _col]);
            vst1q_f32(&x1[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input2[_row * 8 + _col]);
            vst1q_f32(&x2[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x1[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr4 = vsubq_f32(_v2, _v3);
            vst1q_f32(&diff[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v5, _v6);
            vst1q_f32(&sq_diff[_row][_col], _vr7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x1[_row][_col] = input1[_row * 8 + _col];
            x2[_row][_col] = input2[_row * 8 + _col];
            diff[_row][_col] = x1[_row][_col] - x2[_row][_col];
            sq_diff[_row][_col] = diff[_row][_col] * diff[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(sq_diff)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += sq_diff[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): result=TSQRT(row_sum); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v8 = vld1q_f32(&row_sum[_row][_col]);
            float32x4_t _vr9 = vsqrtq_f32(_v8);
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs10);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = sqrtf(row_sum[_row][_col]);
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}