// PTO Program: F_cosine_similarity
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_cosine_similarity(float* input1, float* input2, float* output) {
    float x1[8][8];
    float x2[8][8];
    float dot_prod[8][8];
    float x1_sq[8][8];
    float x2_sq[8][8];
    float dot_sum[8][1];
    float x1_norm_sq[8][1];
    float x2_norm_sq[8][1];
    float x1_norm[8][1];
    float x2_norm[8][1];
    float norm_prod[8][1];
    float result[8][1];

    // Loop fusion: 8 loop overheads saved

    // FUSED LOOP (3 ops): x1=TLOAD(input1,0,0); x2=TLOAD(input2,0,0); dot_prod=TMUL(x1,x2)
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
            float32x4_t _vr4 = vmulq_f32(_v2, _v3);
            vst1q_f32(&dot_prod[_row][_col], _vr4);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x1[_row][_col] = input1[_row * 8 + _col];
            x2[_row][_col] = input2[_row * 8 + _col];
            dot_prod[_row][_col] = x1[_row][_col] * x2[_row][_col];
        }
    }

    // TROWSUM: dot_sum = rowsum(dot_prod)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += dot_prod[_row][_col];
        }
        dot_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): x1_sq=TMUL(x1,x1); x2_sq=TMUL(x2,x2)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v5 = vld1q_f32(&x1[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&x1[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v5, _v6);
            vst1q_f32(&x1_sq[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&x2[_row][_col]);
            float32x4_t _vr10 = vmulq_f32(_v8, _v9);
            vst1q_f32(&x2_sq[_row][_col], _vr10);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x1_sq[_row][_col] = x1[_row][_col] * x1[_row][_col];
            x2_sq[_row][_col] = x2[_row][_col] * x2[_row][_col];
        }
    }

    // TROWSUM: x1_norm_sq = rowsum(x1_sq)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += x1_sq[_row][_col];
        }
        x1_norm_sq[_row][0] = _sum;}

    // TROWSUM: x2_norm_sq = rowsum(x2_sq)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += x2_sq[_row][_col];
        }
        x2_norm_sq[_row][0] = _sum;}

    // FUSED LOOP (6 ops): x1_norm=TSQRT(x1_norm_sq); x2_norm=TSQRT(x2_norm_sq); norm_prod=TMUL(x1_norm,x2_norm); norm_prod=TADDS(norm_prod,1e-08f); result=TDIV(dot_sum,norm_prod); output=TSTORE(result,0,0)
    float32x4_t _vs11 = vdupq_n_f32(1e-08f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v12 = vld1q_f32(&x1_norm_sq[_row][_col]);
            float32x4_t _vr13 = vsqrtq_f32(_v12);
            vst1q_f32(&x1_norm[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&x2_norm_sq[_row][_col]);
            float32x4_t _vr15 = vsqrtq_f32(_v14);
            vst1q_f32(&x2_norm[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&x1_norm[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&x2_norm[_row][_col]);
            float32x4_t _vr18 = vmulq_f32(_v16, _v17);
            vst1q_f32(&norm_prod[_row][_col], _vr18);
            float32x4_t _v19 = vld1q_f32(&norm_prod[_row][_col]);
            float32x4_t _vr20 = vaddq_f32(_v19, _vs11);
            vst1q_f32(&norm_prod[_row][_col], _vr20);
            float32x4_t _v21 = vld1q_f32(&dot_sum[_row][_col]);
            float32x4_t _v22 = vld1q_f32(&norm_prod[_row][_col]);
            float32x4_t _vr23 = vdivq_f32(_v21, _v22);
            vst1q_f32(&result[_row][_col], _vr23);
            float32x4_t _vs24 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs24);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            x1_norm[_row][_col] = sqrtf(x1_norm_sq[_row][_col]);
            x2_norm[_row][_col] = sqrtf(x2_norm_sq[_row][_col]);
            norm_prod[_row][_col] = x1_norm[_row][_col] * x2_norm[_row][_col];
            norm_prod[_row][_col] = norm_prod[_row][_col] + 1e-08f;
            result[_row][_col] = dot_sum[_row][_col] / norm_prod[_row][_col];
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}