// PTO Program: tensor_pow
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tensor_pow(float* input, float* output) {
    float self[8][8];
    float log_self[8][8];
    float scaled[8][8];
    float result[8][8];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (5 ops): self=TLOAD(input,0,0); log_self=TLOG(self); scaled=TMULS(log_self,0.5f); result=TEXP(scaled); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(0.5f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&self[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&self[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&log_self[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&log_self[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v4, _vs0);
            vst1q_f32(&scaled[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&scaled[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&result[_row][_col], _vr7);
            float32x4_t _vs8 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs8);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            self[_row][_col] = input[_row * 8 + _col];
            log_self[_row][_col] = logf(self[_row][_col]);
            scaled[_row][_col] = log_self[_row][_col] * 0.5f;
            result[_row][_col] = expf(scaled[_row][_col]);
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}