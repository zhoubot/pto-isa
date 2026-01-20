// PTO Program: F_binary_cross_entropy
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void F_binary_cross_entropy(float* input, float* target_mem, float* output) {
    float pred[8][8];
    float target[8][8];
    float log_pred[8][8];
    float one_minus_pred[8][8];
    float log_one_minus[8][8];
    float one_minus_target[8][8];
    float term1[8][8];
    float term2[8][8];
    float bce[8][8];
    float row_sum[8][1];
    float result[1][1];

    // Loop fusion: 12 loop overheads saved

    // FUSED LOOP (12 ops): pred=TLOAD(input,0,0); target=TLOAD(target_mem,0,0); log_pred=TLOG(pred); one_minus_pred=TMULS(pred,-1.0f); one_minus_pred=TADDS(one_minus_pred,1.0f); log_one_minus=TLOG(one_minus_pred); one_minus_target=TMULS(target,-1.0f); one_minus_target=TADDS(one_minus_target,1.0f); term1=TMUL(target,log_pred); term2=TMUL(one_minus_target,log_one_minus); bce=TADD(term1,term2); bce=TNEG(bce)
    float32x4_t _vs0 = vdupq_n_f32(-1.0f);
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&pred[_row][_col], _vl2);
            float32x4_t _vl3 = vld1q_f32(&target_mem[_row * 8 + _col]);
            vst1q_f32(&target[_row][_col], _vl3);
            float32x4_t _v4 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&log_pred[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&pred[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v6, _vs0);
            vst1q_f32(&one_minus_pred[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&one_minus_pred[_row][_col]);
            float32x4_t _vr9 = vaddq_f32(_v8, _vs1);
            vst1q_f32(&one_minus_pred[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&one_minus_pred[_row][_col]);
            float32x4_t _vr11 = _v10;
            vst1q_f32(&log_one_minus[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&target[_row][_col]);
            float32x4_t _vr13 = vmulq_f32(_v12, _vs0);
            vst1q_f32(&one_minus_target[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&one_minus_target[_row][_col]);
            float32x4_t _vr15 = vaddq_f32(_v14, _vs1);
            vst1q_f32(&one_minus_target[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&target[_row][_col]);
            float32x4_t _v17 = vld1q_f32(&log_pred[_row][_col]);
            float32x4_t _vr18 = vmulq_f32(_v16, _v17);
            vst1q_f32(&term1[_row][_col], _vr18);
            float32x4_t _v19 = vld1q_f32(&one_minus_target[_row][_col]);
            float32x4_t _v20 = vld1q_f32(&log_one_minus[_row][_col]);
            float32x4_t _vr21 = vmulq_f32(_v19, _v20);
            vst1q_f32(&term2[_row][_col], _vr21);
            float32x4_t _v22 = vld1q_f32(&term1[_row][_col]);
            float32x4_t _v23 = vld1q_f32(&term2[_row][_col]);
            float32x4_t _vr24 = vaddq_f32(_v22, _v23);
            vst1q_f32(&bce[_row][_col], _vr24);
            float32x4_t _v25 = vld1q_f32(&bce[_row][_col]);
            float32x4_t _vr26 = vnegq_f32(_v25);
            vst1q_f32(&bce[_row][_col], _vr26);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            pred[_row][_col] = input[_row * 8 + _col];
            target[_row][_col] = target_mem[_row * 8 + _col];
            log_pred[_row][_col] = logf(pred[_row][_col]);
            one_minus_pred[_row][_col] = pred[_row][_col] * -1.0f;
            one_minus_pred[_row][_col] = one_minus_pred[_row][_col] + 1.0f;
            log_one_minus[_row][_col] = logf(one_minus_pred[_row][_col]);
            one_minus_target[_row][_col] = target[_row][_col] * -1.0f;
            one_minus_target[_row][_col] = one_minus_target[_row][_col] + 1.0f;
            term1[_row][_col] = target[_row][_col] * log_pred[_row][_col];
            term2[_row][_col] = one_minus_target[_row][_col] * log_one_minus[_row][_col];
            bce[_row][_col] = term1[_row][_col] + term2[_row][_col];
            bce[_row][_col] = -bce[_row][_col];
        }
    }

    // TROWSUM: row_sum = rowsum(bce)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += bce[_row][_col];
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
    float32x4_t _vs27 = vdupq_n_f32(64.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v28 = vld1q_f32(&result[_row][_col]);
            float32x4_t _vr29 = vdivq_f32(_v28, _vs27);
            vst1q_f32(&result[_row][_col], _vr29);
            float32x4_t _vs30 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs30);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            result[_row][_col] = result[_row][_col] / 64.0f;
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}